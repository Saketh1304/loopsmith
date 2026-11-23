"""Running commands on this machine, as the user. No sandbox at all.

This is how run_command has always worked; the code moved here from
tools/shell.py unchanged. A command can do anything the user's account
can do, which is why every command needs approval.
"""

import os
import platform
import subprocess
from pathlib import Path

from loopsmith.config import API_KEY_VAR
from loopsmith.sandbox.base import CommandResult, CommandTimedOut


class LocalExecutor:
    def __init__(self, root: Path):
        self._root = root

    @property
    def where(self) -> str:
        return f"on {platform.system()} through the system shell"

    def run(self, command: str, timeout_seconds: int) -> CommandResult:
        # The child process must not inherit the model API key.
        environment = {k: v for k, v in os.environ.items() if k != API_KEY_VAR}
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self._root,
                env=environment,
                stdin=subprocess.DEVNULL,  # a command that waits for input fails fast
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # one stream, in the order it was printed
                timeout=timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            # Known gap: only the shell is stopped. Processes it started can
            # keep running. The Docker executor fixes this by stopping
            # everything in the container.
            raise CommandTimedOut() from None
        return CommandResult(completed.returncode, completed.stdout or "")
