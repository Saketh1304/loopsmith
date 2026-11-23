"""A tool that runs shell commands in the workspace.

This is the most powerful and the most dangerous tool, so it always needs
approval and runs with a time limit. Where the command actually runs is
up to an executor (see loopsmith.sandbox): on this machine by default.
This file only decides what the model is told about it.
"""

from loopsmith.sandbox import CommandTimedOut, Executor, LocalExecutor
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
    workspace: Workspace,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    executor: Executor | None = None,
) -> Tool:
    # The same injection-with-a-default as GeminiClient's transport: real
    # use gets this machine, a test or the sandbox passes its own.
    runner = executor if executor is not None else LocalExecutor(workspace.root)

    def preview(args: dict) -> str:
        return f"Run command: {require_string(args, 'command')}"

    def run(args: dict) -> str:
        command = require_string(args, "command")
        try:
            result = runner.run(command, timeout_seconds)
        except CommandTimedOut:
            raise ToolError(
                f"The command did not finish within {timeout_seconds} seconds "
                "and was stopped. Do not run servers or commands that wait for input."
            ) from None

        output = _shorten(result.output).strip() or "(no output)"
        # A non-zero exit code is a normal result, not a tool error: the
        # model needs to read a failing test's output to fix the code.
        return f"exit code: {result.exit_code}\n{output}"

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
