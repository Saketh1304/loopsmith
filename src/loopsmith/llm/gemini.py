"""Google Gemini, called over plain HTTP.

API reference: https://ai.google.dev/gemini-api/docs/generate-content/text-generation
"""

from urllib.parse import quote

from loopsmith.llm.base import Message, ModelError, ModelReply
from loopsmith.llm.http import Transport, send_json

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Gemini calls the assistant's role "model".
_ROLE_NAMES = {"user": "user", "assistant": "model"}


class GeminiClient:
    """Implements ModelClient for the Gemini API."""

    def __init__(self, api_key: str, model: str, transport: Transport = send_json):
        self._api_key = api_key
        self._model = model
        self._transport = transport

    def generate(self, messages: list[Message], system: str | None = None) -> ModelReply:
        url = f"{BASE_URL}/models/{quote(self._model, safe='')}:generateContent"
        status, body = self._transport(
            "POST", url, self._headers(), build_request_body(messages, system)
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
        if status == 404:
            hint = (
                f"Model '{self._model}' was not found. Run "
                "`loopsmith --list-models` to see the models your key can use, "
                "then set GEMINI_MODEL in your .env file."
            )
        elif status == 429:
            hint = "Rate limit reached. Wait a minute and try again."
        elif status in (400, 401, 403):
            hint = "The request was rejected. Check that your API key is correct."
        elif status >= 500:
            hint = "The model server had a problem. Try again shortly."
        else:
            hint = "The request failed."
        retryable = status == 429 or status >= 500
        return ModelError(f"{hint} (HTTP {status}: {detail})", status, retryable)


def build_request_body(messages: list[Message], system: str | None) -> dict:
    """Translate our neutral messages into Gemini's JSON shape."""
    body: dict = {
        "contents": [
            {"role": _ROLE_NAMES[message.role], "parts": [{"text": message.text}]}
            for message in messages
        ]
    }
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    return body


def parse_reply(body: dict) -> ModelReply:
    """Pull the answer text out of Gemini's response JSON."""
    candidates = body.get("candidates", [])
    if not candidates:
        feedback = body.get("promptFeedback", {})
        raise ModelError(f"The model returned no answer. Feedback: {feedback}")
    parts = candidates[0].get("content", {}).get("parts", [])
    # Skip "thought" parts: those are the model's reasoning, not its answer.
    text = "".join(
        part["text"] for part in parts if "text" in part and not part.get("thought")
    )
    return ModelReply(text=text)
