"""Where the agent's commands run: on this machine, or in a sandbox."""

from loopsmith.sandbox.base import CommandResult, CommandTimedOut, Executor
from loopsmith.sandbox.docker import DockerExecutor, SandboxError, current_user
from loopsmith.sandbox.image import IMAGE_DESCRIPTION, ensure_image
from loopsmith.sandbox.local import LocalExecutor

__all__ = [
    "IMAGE_DESCRIPTION",
    "CommandResult",
    "CommandTimedOut",
    "DockerExecutor",
    "Executor",
    "LocalExecutor",
    "SandboxError",
    "current_user",
    "ensure_image",
]
