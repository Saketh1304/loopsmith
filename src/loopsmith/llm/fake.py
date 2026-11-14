"""A stand-in model for tests: no network, no key, fully predictable."""

from loopsmith.llm.base import Message, ModelReply, ToolCall, ToolSpec


class FakeClient:
    """Returns scripted replies in order and records what it was sent.

    Each scripted reply is either a string (a plain text answer), a
    ToolCall or list of ToolCalls (the model asking for tools), or a
    ready-made ModelReply.
    """

    def __init__(self, replies: list):
        self._replies = list(replies)
        self.calls: list[tuple[list[Message], str | None, list[ToolSpec] | None]] = []

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ModelReply:
        self.calls.append((list(messages), system, tools))
        if not self._replies:
            raise AssertionError("FakeClient ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, ModelReply):
            return reply
        if isinstance(reply, ToolCall):
            return ModelReply(text="", tool_calls=(reply,))
        if isinstance(reply, list):
            return ModelReply(text="", tool_calls=tuple(reply))
        return ModelReply(text=reply)
