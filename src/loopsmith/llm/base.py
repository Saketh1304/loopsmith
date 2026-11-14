"""Provider-neutral types.

The rest of the agent only uses what is defined here. It never imports a
specific provider, so swapping Gemini for another model means writing one
new class that satisfies `ModelClient`.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One turn in the conversation."""

    role: Role
    text: str


@dataclass(frozen=True)
class ModelReply:
    """What the model answered."""

    text: str


class ModelError(Exception):
    """A call to the model failed.

    `status` is the HTTP status code, or None for a network failure.
    `retryable` says whether trying again later could succeed.
    """

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ModelClient(Protocol):
    """The interface every provider implements.

    A Protocol is Python's version of a Java interface, with one
    difference: a class satisfies it by having the right methods, without
    declaring `implements`.
    """

    def generate(self, messages: list[Message], system: str | None = None) -> ModelReply:
        """Send the conversation so far and return the model's reply."""
        ...
