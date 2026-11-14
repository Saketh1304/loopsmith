"""A stand-in model for tests: no network, no key, fully predictable."""

from loopsmith.llm.base import Message, ModelReply


class FakeClient:
    """Returns scripted replies in order and records what it was sent."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[tuple[list[Message], str | None]] = []

    def generate(self, messages: list[Message], system: str | None = None) -> ModelReply:
        self.calls.append((list(messages), system))
        if not self._replies:
            raise AssertionError("FakeClient ran out of scripted replies")
        return ModelReply(text=self._replies.pop(0))
