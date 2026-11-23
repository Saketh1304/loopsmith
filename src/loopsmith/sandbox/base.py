"""What an executor is: the thing that actually runs a command.

`run_command` decides what to run and how to report it; an executor
decides where it runs. Today that is this machine (`LocalExecutor`).
The next step adds a Docker container, and `run_command` will not need
to change for it.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    """What a finished command produced."""

    exit_code: int
    output: str  # stdout and stderr together, in the order they were printed


class CommandTimedOut(Exception):
    """The command was still running when its time ran out, and was stopped."""


class Executor(Protocol):
    """Runs shell commands somewhere. Like ModelClient, a Protocol: any
    class with these two members is an executor, without saying so."""

    @property
    def where(self) -> str:
        """Where commands run, in words for the model's system prompt, such
        as "on Windows through the system shell"."""
        ...

    def run(self, command: str, timeout_seconds: int) -> CommandResult:
        """Run one command and wait for it.

        A non-zero exit code is a normal result. Raises CommandTimedOut if
        the command takes longer than `timeout_seconds`.
        """
        ...
