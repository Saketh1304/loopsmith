"""Google Gemini, called over plain HTTP.

API references:
  https://ai.google.dev/gemini-api/docs/generate-content/text-generation
  https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures
"""

from urllib.parse import quote

from loopsmith.llm.base import (
    LONG_WAIT_SECONDS,
    Message,
    ModelError,
    ModelReply,
    ToolCall,
    ToolSpec,
    Usage,
)
from loopsmith.llm.http import Transport, send_json

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    """Implements ModelClient for the Gemini API."""

    def __init__(self, api_key: str, model: str, transport: Transport = send_json):
        self._api_key = api_key
        self._model = model
        self._transport = transport

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ModelReply:
        url = f"{BASE_URL}/models/{quote(self._model, safe='')}:generateContent"
        status, body = self._transport(
            "POST", url, self._headers(), build_request_body(messages, system, tools)
        )
        if status != 200:
            raise self._error(status, body)
        return parse_reply(body)

    def list_models(self) -> list[str]:
        """Return the names of models this key can use for text generation."""
        url = f"{BASE_URL}/models?pageSize=1000"
        status, body = self._transport("GET", url, self._headers(), None)
        if status != 200:
            raise self._error(status, body)
        names = []
        for model in body.get("models", []):
            if "generateContent" in model.get("supportedGenerationMethods", []):
                names.append(model.get("name", "").removeprefix("models/"))
        return sorted(names)

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    def _error(self, status: int, body: dict) -> ModelError:
        detail = body.get("error", {}).get("message", "no details given")
        delay = _retry_delay(body)
        if status == 404:
            hint = (
                f"Model '{self._model}' was not found. Run "
                "`loopsmith --list-models` to see the models your key can use, "
                "then set GEMINI_MODEL in your .env file."
            )
        elif status == 429:
            hint = self._rate_limit_hint(delay)
        elif status in (401, 403):
            hint = "The request was rejected. Check that your API key is correct."
        elif status == 400:
            hint = "The server rejected the request as invalid."
        elif status >= 500:
            hint = "The model server had a problem. Try again shortly."
        else:
            hint = "The request failed."
        retryable = status == 429 or status >= 500
        return ModelError(f"{hint} (HTTP {status}: {detail})", status, retryable, delay)

    def _rate_limit_hint(self, delay: float | None) -> str:
        if delay is None:
            return "Rate limit reached. Wait a minute and try again."
        if delay > LONG_WAIT_SECONDS:
            return (
                f"The request quota for model '{self._model}' is used up. It "
                f"resets in about {format_wait(delay)}. Wait until then, or set "
                "GEMINI_MODEL in your .env file to another model "
                "(`loopsmith --list-models` shows the choices)."
            )
        return f"Rate limit reached. Try again in about {format_wait(delay)}."


def format_wait(seconds: float) -> str:
    """Turn a number of seconds into words, such as "18 h 1 min"."""
    minutes = int(seconds // 60)
    if minutes >= 60:
        return f"{minutes // 60} h {minutes % 60} min"
    if minutes >= 1:
        return f"{minutes} min"
    return f"{int(seconds)} s"


def _retry_delay(body: dict) -> float | None:
    """Read the wait the server suggests, such as "retryDelay": "34s"."""
    details = body.get("error", {}).get("details", [])
    if not isinstance(details, list):
        return None
    for detail in details:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def build_request_body(
    messages: list[Message],
    system: str | None = None,
    tools: list[ToolSpec] | None = None,
) -> dict:
    """Translate our neutral messages into Gemini's JSON shape."""
    body: dict = {"contents": [_content_for(message) for message in messages]}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            }
        ]
    return body


def _content_for(message: Message) -> dict:
    if message.role == "user":
        return {"role": "user", "parts": [{"text": message.text}]}

    if message.role == "assistant":
        # Replay the model's own parts untouched when we have them. They
        # can carry a `thoughtSignature`, and Gemini rejects the next
        # request with HTTP 400 if a function call comes back without it.
        if message.raw is not None:
            return {"role": "model", "parts": message.raw}
        parts: list[dict] = []
        if message.text:
            parts.append({"text": message.text})
        for call in message.tool_calls:
            parts.append({"functionCall": _with_id({"name": call.name, "args": call.args}, call.id)})
        return {"role": "model", "parts": parts}

    # role "tool": Gemini expects tool results in a "user" turn.
    parts = []
    for result in message.tool_results:
        key = "error" if result.is_error else "result"
        response = {"name": result.name, "response": {key: result.content}}
        parts.append({"functionResponse": _with_id(response, result.id)})
    return {"role": "user", "parts": parts}


def _with_id(payload: dict, call_id: str | None) -> dict:
    if call_id is not None:
        payload["id"] = call_id
    return payload


def parse_reply(body: dict) -> ModelReply:
    """Pull the answer text and any tool calls out of Gemini's response."""
    candidates = body.get("candidates", [])
    if not candidates:
        feedback = body.get("promptFeedback", {})
        raise ModelError(f"The model returned no answer. Feedback: {feedback}")

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    if not parts:
        reason = candidate.get("finishReason", "unknown")
        raise ModelError(f"The model returned an empty answer (finish reason: {reason}).")

    # Skip "thought" parts: those are the model's reasoning, not its answer.
    text = "".join(
        part["text"] for part in parts if "text" in part and not part.get("thought")
    )
    tool_calls = tuple(
        ToolCall(
            name=part["functionCall"].get("name", ""),
            args=part["functionCall"].get("args") or {},
            id=part["functionCall"].get("id"),
        )
        for part in parts
        if "functionCall" in part
    )
    message = Message("assistant", text, tool_calls, raw=parts)
    return ModelReply(
        text=text, tool_calls=tool_calls, message=message, usage=parse_usage(body)
    )


def parse_usage(body: dict) -> Usage | None:
    """Read the token counts Gemini attaches to every reply.

    Gemini reports thinking tokens separately from the answer. Both are
    output the model produced and both are billed as output, so they are
    added together here.
    """
    metadata = body.get("usageMetadata")
    if not isinstance(metadata, dict):
        return None
    return Usage(
        input_tokens=_count(metadata, "promptTokenCount"),
        output_tokens=_count(metadata, "candidatesTokenCount")
        + _count(metadata, "thoughtsTokenCount"),
        cached_tokens=_count(metadata, "cachedContentTokenCount"),
    )


def _count(metadata: dict, key: str) -> int:
    """A missing count means zero: Gemini leaves out fields that are 0."""
    value = metadata.get(key)
    return value if isinstance(value, int) else 0
