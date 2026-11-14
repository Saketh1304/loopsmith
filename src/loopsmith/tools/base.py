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
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], str]

    @property
    def spec(self) -> ToolSpec:
        """The part of the tool the model sees."""
        return ToolSpec(self.name, self.description, self.parameters)


def require_string(args: dict, name: str) -> str:
    """Read a required string argument, or raise a ToolError."""
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Missing required argument '{name}' (a non-empty string).")
    return value
