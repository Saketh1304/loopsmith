import pytest

from loopsmith.llm import Message, ModelError, ToolCall, ToolResult, ToolSpec, Usage
from loopsmith.llm.gemini import GeminiClient, build_request_body, format_wait, parse_reply


class RecordingTransport:
    """A fake transport: returns a canned response and records the request."""

    def __init__(self, status: int, body: dict):
        self.status = status
        self.body = body
        self.requests: list[tuple] = []

    def __call__(self, method, url, headers, body):
        self.requests.append((method, url, headers, body))
        return self.status, self.body


def reply_body(*texts: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": t} for t in texts]}}]}


def test_request_goes_to_the_model_url_with_the_key_header():
    transport = RecordingTransport(200, reply_body("hi"))
    client = GeminiClient("secret-key", "some-model", transport)

    client.generate([Message("user", "hello")])

    method, url, headers, _ = transport.requests[0]
    assert method == "POST"
    assert url.endswith("/models/some-model:generateContent")
    assert headers == {"x-goog-api-key": "secret-key"}
    assert "secret-key" not in url


def test_messages_are_translated_to_gemini_format():
    messages = [
        Message("user", "hello"),
        Message("assistant", "hi there"),
        Message("user", "how are you?"),
    ]

    body = build_request_body(messages, system="Be brief.")

    assert body == {
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi there"}]},
            {"role": "user", "parts": [{"text": "how are you?"}]},
        ],
        "system_instruction": {"parts": [{"text": "Be brief."}]},
    }


def test_system_instruction_is_left_out_when_not_given():
    body = build_request_body([Message("user", "hello")], system=None)

    assert "system_instruction" not in body


def test_reply_text_parts_are_joined():
    assert parse_reply(reply_body("Hello, ", "world")).text == "Hello, world"


def test_thought_parts_are_not_part_of_the_answer():
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "let me think...", "thought": True},
                        {"text": "42"},
                    ]
                }
            }
        ]
    }

    assert parse_reply(body).text == "42"


def test_no_candidates_raises_with_feedback():
    body = {"promptFeedback": {"blockReason": "SAFETY"}}

    with pytest.raises(ModelError, match="SAFETY"):
        parse_reply(body)


@pytest.mark.parametrize(
    ("status", "expected_text", "retryable"),
    [
        (404, "--list-models", False),
        (429, "Rate limit", True),
        (403, "API key", False),
        (503, "Try again", True),
    ],
)
def test_http_errors_become_helpful_model_errors(status, expected_text, retryable):
    transport = RecordingTransport(status, {"error": {"message": "server detail"}})
    client = GeminiClient("key", "some-model", transport)

    with pytest.raises(ModelError) as error_info:
        client.generate([Message("user", "hello")])

    error = error_info.value
    assert expected_text in str(error)
    assert "server detail" in str(error)
    assert error.status == status
    assert error.retryable is retryable


def test_list_models_keeps_only_text_generation_models():
    transport = RecordingTransport(
        200,
        {
            "models": [
                {"name": "models/zeta", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/embedder", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/alpha", "supportedGenerationMethods": ["generateContent"]},
            ]
        },
    )
    client = GeminiClient("key", "some-model", transport)

    assert client.list_models() == ["alpha", "zeta"]
    assert transport.requests[0][0] == "GET"


# --- Tool calling ---

READ_SPEC = ToolSpec(
    name="read_file",
    description="Read a file.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)

# What Gemini sends when it wants a tool: a functionCall part that also
# carries an encrypted thoughtSignature.
FUNCTION_CALL_PARTS = [
    {
        "functionCall": {"name": "read_file", "args": {"path": "app.py"}},
        "thoughtSignature": "SIGNATURE-A",
    }
]


def test_tools_are_declared_in_the_request():
    body = build_request_body([Message("user", "hi")], tools=[READ_SPEC])

    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": READ_SPEC.parameters,
                }
            ]
        }
    ]


def test_no_tools_key_when_no_tools_given():
    assert "tools" not in build_request_body([Message("user", "hi")])


def test_function_call_is_parsed_into_a_tool_call():
    reply = parse_reply({"candidates": [{"content": {"parts": FUNCTION_CALL_PARTS}}]})

    assert reply.text == ""
    assert reply.tool_calls == (ToolCall("read_file", {"path": "app.py"}),)


def test_thought_signature_survives_the_round_trip():
    """Gemini returns HTTP 400 if a function call comes back without the
    signature it was sent with, so the original parts are replayed as is."""
    reply = parse_reply({"candidates": [{"content": {"parts": FUNCTION_CALL_PARTS}}]})
    history = [
        Message("user", "read app.py"),
        reply.as_message(),
        Message("tool", tool_results=(ToolResult("read_file", "print('hi')"),)),
    ]

    body = build_request_body(history)

    assert body["contents"][1] == {"role": "model", "parts": FUNCTION_CALL_PARTS}


def test_tool_results_are_sent_as_function_responses_in_a_user_turn():
    results = (
        ToolResult("read_file", "print('hi')", id="call-1"),
        ToolResult("read_file", "File does not exist.", is_error=True),
    )

    body = build_request_body([Message("tool", tool_results=results)])

    assert body["contents"] == [
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "read_file",
                        "response": {"result": "print('hi')"},
                        "id": "call-1",
                    }
                },
                {
                    "functionResponse": {
                        "name": "read_file",
                        "response": {"error": "File does not exist."},
                    }
                },
            ],
        }
    ]


def test_several_function_calls_in_one_reply_make_the_round_trip_together():
    # Asking for independent tools together saves requests. Gemini puts the
    # thought signature on the first call only; all calls go back as they
    # came, and all results go back in one user turn, in the same order.
    parts = [
        {
            "functionCall": {"name": "read_file", "args": {"path": "a.py"}},
            "thoughtSignature": "sig-1",
        },
        {"functionCall": {"name": "read_file", "args": {"path": "b.py"}}},
    ]
    reply = parse_reply({"candidates": [{"content": {"parts": parts}}]})
    assert [call.args["path"] for call in reply.tool_calls] == ["a.py", "b.py"]

    results = tuple(
        ToolResult("read_file", f"contents of {call.args['path']}") for call in reply.tool_calls
    )
    history = [
        Message("user", "read both"),
        reply.as_message(),
        Message("tool", tool_results=results),
    ]
    body = build_request_body(history)

    assert len(body["contents"]) == 3
    assert body["contents"][1] == {"role": "model", "parts": parts}
    sent = [part["functionResponse"]["response"]["result"] for part in body["contents"][2]["parts"]]
    assert sent == ["contents of a.py", "contents of b.py"]


def test_assistant_tool_calls_without_raw_parts_are_rebuilt():
    message = Message("assistant", "Let me look.", (ToolCall("read_file", {"path": "a"}),))

    body = build_request_body([message])

    assert body["contents"][0] == {
        "role": "model",
        "parts": [
            {"text": "Let me look."},
            {"functionCall": {"name": "read_file", "args": {"path": "a"}}},
        ],
    }


def test_empty_answer_reports_the_finish_reason():
    body = {"candidates": [{"content": {}, "finishReason": "MALFORMED_FUNCTION_CALL"}]}

    with pytest.raises(ModelError, match="MALFORMED_FUNCTION_CALL"):
        parse_reply(body)


def test_rate_limit_error_carries_the_servers_retry_delay():
    body = {
        "error": {
            "message": "Quota exceeded",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
            ],
        }
    }
    client = GeminiClient("key", "some-model", RecordingTransport(429, body))

    with pytest.raises(ModelError) as error_info:
        client.generate([Message("user", "hello")])

    assert error_info.value.retry_after == 34.0


def test_error_without_retry_info_has_no_retry_delay():
    client = GeminiClient("key", "m", RecordingTransport(503, {"error": {"message": "x"}}))

    with pytest.raises(ModelError) as error_info:
        client.generate([Message("user", "hello")])

    assert error_info.value.retry_after is None


def test_token_counts_are_read_from_the_reply():
    body = reply_body("hi")
    body["usageMetadata"] = {
        "promptTokenCount": 1200,
        "candidatesTokenCount": 30,
        "cachedContentTokenCount": 1000,
        "totalTokenCount": 1230,
    }

    assert parse_reply(body).usage == Usage(
        input_tokens=1200, output_tokens=30, cached_tokens=1000
    )


def test_thinking_tokens_count_as_output():
    body = reply_body("hi")
    body["usageMetadata"] = {
        "promptTokenCount": 50,
        "candidatesTokenCount": 10,
        "thoughtsTokenCount": 200,
    }

    assert parse_reply(body).usage == Usage(input_tokens=50, output_tokens=210)


def test_missing_token_counts_are_zero():
    body = reply_body("hi")
    body["usageMetadata"] = {"promptTokenCount": 50}

    assert parse_reply(body).usage == Usage(input_tokens=50)


def test_reply_without_usage_metadata_has_no_usage():
    assert parse_reply(reply_body("hi")).usage is None


def quota_error_body(retry_delay: str) -> dict:
    return {
        "error": {
            "message": "You exceeded your current quota",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
            ],
        }
    }


def test_a_used_up_daily_quota_is_explained_as_one():
    # Seen in a real run: the free tier's 20 requests a day ran out and
    # the server said to retry in 18 hours.
    transport = RecordingTransport(429, quota_error_body("64903.7s"))
    client = GeminiClient("key", "some-model", transport)

    with pytest.raises(ModelError) as error_info:
        client.generate([Message("user", "hello")])

    message = str(error_info.value)
    assert "quota for model 'some-model' is used up" in message
    assert "18 h 1 min" in message
    assert "GEMINI_MODEL" in message
    assert "Wait a minute" not in message
    assert error_info.value.retry_after == 64903.7


def test_a_short_rate_limit_says_how_long_to_wait():
    client = GeminiClient("key", "m", RecordingTransport(429, quota_error_body("34s")))

    with pytest.raises(ModelError, match="Rate limit reached. Try again in about 34 s"):
        client.generate([Message("user", "hello")])


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.4, "0 s"), (34, "34 s"), (60, "1 min"), (150, "2 min"), (3600, "1 h 0 min"), (64903.7, "18 h 1 min")],
)
def test_waits_are_written_in_words(seconds, expected):
    assert format_wait(seconds) == expected
