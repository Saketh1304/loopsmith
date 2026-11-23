"""Where the agent's commands run: on this machine, or in a sandbox."""

from loopsmith.sandbox.base import CommandResult, CommandTimedOut, Executor
from loopsmith.sandbox.local import LocalExecutor

__all__ = ["CommandResult", "CommandTimedOut", "Executor", "LocalExecutor"]
