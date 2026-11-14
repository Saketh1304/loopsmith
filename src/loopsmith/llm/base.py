"""Provider-neutral types.

The rest of the agent only uses what is defined here. It never imports a
specific provider, so swapping Gemini for another model means writing one
new class that satisfies `ModelClient`.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """Describes one tool to the model.

    `parameters` is a JSON Schema object saying which arguments the tool
    takes. The model reads the name, description and schema to decide
    when and how to call the tool.
    """

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    """The model asking for a tool to be run."""

    name: str
    args: dict
    id: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """What happened when a tool was run, sent back to the model."""

    name: str
    content: str
    is_error: bool = False
    id: str | None = None


@dataclass(frozen=True)
class Message:
    """One entry in the conversation history.

    - role "user": text from the person.
    - role "assistant": the model's turn, with text and/or tool calls.
    - role "tool": the results of the tool calls in the previous turn.

    `raw` holds the provider's own version of an assistant turn. Some
    providers attach data that must be sent back unchanged (Gemini's
    thought signatures), so the client stores the original here and
    replays it as it was received.
    """

    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    raw: Any = None


@dataclass(frozen=True)
class ModelReply:
    """What the model answered.

    `message` is the assistant turn to append to the history.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    message: Message | None = None

    def as_message(self) -> Message:
        if self.message is not None:
            return self.message
        return Message("assistant", self.text, self.tool_calls)


class ModelError(Exception):
    """A call to the model failed.

    `status` is the HTTP status code, or None for a network failure.
    `retryable` says whether trying again later could succeed.
    `retry_after` is how many seconds the server asked us to wait, if it said.
    """

    def __init__(
        self,
        message: str,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


class ModelClient(Protocol):
    """The interface every provider implements.

    A Protocol is Python's version of a Java interface, with one
    difference: a class satisfies it by having the right methods, without
    declaring `implements`.
    """

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ModelReply:
        """Send the conversation so far and return the model's reply."""
        ...
