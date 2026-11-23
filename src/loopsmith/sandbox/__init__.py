"""Where the agent's commands run: on this machine, or in a sandbox."""

from loopsmith.sandbox.base import CommandResult, CommandTimedOut, Executor
from loopsmith.sandbox.docker import DEFAULT_IMAGE, DockerExecutor, SandboxError, current_user
from loopsmith.sandbox.local import LocalExecutor

__all__ = [
    "DEFAULT_IMAGE",
    "CommandResult",
    "CommandTimedOut",
    "DockerExecutor",
    "Executor",
    "LocalExecutor",
    "SandboxError",
    "current_user",
]
