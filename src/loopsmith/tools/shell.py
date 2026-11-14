"""A tool that runs shell commands in the workspace.

This is the most powerful and the most dangerous tool. A command can do
anything the user's account can do, so it always needs approval, and it
runs with a time limit and without the API key in its environment.
"""

import os
import subprocess

from loopsmith.config import API_KEY_VAR
from loopsmith.tools.base import Tool, ToolError, require_string
from loopsmith.tools.files import Workspace

DEFAULT_TIMEOUT_SECONDS = 60
MAX_OUTPUT_CHARS = 20_000


def _shorten(output: str) -> str:
    """Keep the start and the end of long output.

    Errors and test summaries are usually at the end, and the command's
    first lines say what ran, so both ends are worth keeping.
    """
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(output) - MAX_OUTPUT_CHARS
    return f"{output[:half]}\n[... {omitted} characters left out ...]\n{output[-half:]}"


def run_command_tool(
    workspace: Workspace, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
) -> Tool:
    def preview(args: dict) -> str:
        return f"Run command: {require_string(args, 'command')}"

    def run(args: dict) -> str:
        command = require_string(args, "command")
        # The child process must not inherit the model API key.
        environment = {k: v for k, v in os.environ.items() if k != API_KEY_VAR}
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=workspace.root,
                env=environment,
                stdin=subprocess.DEVNULL,  # a command that waits for input fails fast
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # one stream, in the order it was printed
                timeout=timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"The command did not finish within {timeout_seconds} seconds "
                "and was stopped. Do not run servers or commands that wait for input."
            ) from None

        output = _shorten(completed.stdout or "").strip() or "(no output)"
        # A non-zero exit code is a normal result, not a tool error: the
        # model needs to read a failing test's output to fix the code.
        return f"exit code: {completed.returncode}\n{output}"

    return Tool(
        name="run_command",
        description=(
            "Run a shell command in the project folder and return its exit "
            f"code and output. Commands are stopped after {timeout_seconds} "
            "seconds and cannot read keyboard input. Use this to run tests."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command line to run.",
                }
            },
            "required": ["command"],
        },
        run=run,
        preview=preview,
    )
