import pytest

from loopsmith.llm import Message, ModelError
from loopsmith.llm.gemini import GeminiClient, build_request_body, parse_reply


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
