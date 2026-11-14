"""What a tool is.

A tool is a plain Python function plus a description the model can read.
"""

from dataclasses import dataclass
from typing import Callable

from loopsmith.llm.base import ToolSpec


class ToolError(Exception):
    """A tool could not do what was asked.

    The message is sent back to the model, so it should say what went
    wrong in a way that helps the model try something else.
    """


@dataclass(frozen=True)
class Tool:
    """One action the agent can take.

    `run` does the work. `preview`, when present, marks the tool as one
    that changes things: it returns a description of what is about to
    happen, which is shown to the user for approval before `run` is
    called. Read-only tools leave `preview` as None and run at once.
    """

    name: str
    description: str
    parameters: dict
    run: Callable[[dict], str]
    preview: Callable[[dict], str] | None = None

    @property
    def spec(self) -> ToolSpec:
        """The part of the tool the model sees."""
        return ToolSpec(self.name, self.description, self.parameters)

    @property
    def needs_approval(self) -> bool:
        return self.preview is not None


def require_string(args: dict, name: str, allow_empty: bool = False) -> str:
    """Read a required string argument, or raise a ToolError."""
    value = args.get(name)
    if not isinstance(value, str):
        raise ToolError(f"Missing required argument '{name}' (a string).")
    if not allow_empty and not value.strip():
        raise ToolError(f"Argument '{name}' must not be empty.")
    return value
