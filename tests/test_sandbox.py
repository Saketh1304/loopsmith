"""Tests for executors and for run_command's use of them.

run_command decides what the model is told; the executor decides where
the command runs. A fake executor checks the first part without running
anything. The real local executor uses the current Python interpreter,
so it behaves the same on Windows, macOS and Linux.
"""

import platform
import sys

import pytest

from loopsmith.cli import system_prompt
from loopsmith.sandbox import CommandResult, CommandTimedOut, LocalExecutor
from loopsmith.tools import ToolError, Workspace, default_tools
from loopsmith.tools.shell import MAX_OUTPUT_CHARS, run_command_tool


class FakeExecutor:
    """Records each command and answers with a scripted result."""

    where = "inside a test box"

    def __init__(self, result: CommandResult | None = None, times_out: bool = False):
        self._result = result or CommandResult(0, "")
        self._times_out = times_out
        self.commands: list[tuple[str, int]] = []

    def run(self, command: str, timeout_seconds: int) -> CommandResult:
        self.commands.append((command, timeout_seconds))
        if self._times_out:
            raise CommandTimedOut()
        return self._result


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path)


def run_tool(workspace, executor, command="make test", timeout_seconds=60) -> str:
    tool = run_command_tool(workspace, timeout_seconds=timeout_seconds, executor=executor)
    return tool.run({"command": command})


# --- run_command with an injected executor ---


def test_run_command_hands_the_command_and_time_limit_to_the_executor(workspace):
    executor = FakeExecutor()

    run_tool(workspace, executor, command="pytest -q", timeout_seconds=30)

    assert executor.commands == [("pytest -q", 30)]


def test_run_command_reports_what_the_executor_returned(workspace):
    executor = FakeExecutor(CommandResult(2, "1 failed\n"))

    assert run_tool(workspace, executor) == "exit code: 2\n1 failed"


def test_an_executor_time_out_becomes_a_tool_error(workspace):
    executor = FakeExecutor(times_out=True)

    with pytest.raises(ToolError, match="within 5 seconds"):
        run_tool(workspace, executor, timeout_seconds=5)


def test_output_from_any_executor_is_shortened(workspace):
    long_output = "START" + "x" * (MAX_OUTPUT_CHARS * 2) + "END"
    executor = FakeExecutor(CommandResult(0, long_output))

    result = run_tool(workspace, executor)

    assert "characters left out" in result
    assert result.endswith("END")


def test_empty_output_from_an_executor_is_reported(workspace):
    assert run_tool(workspace, FakeExecutor()) == "exit code: 0\n(no output)"


def test_default_tools_give_run_command_the_executor(workspace):
    executor = FakeExecutor(CommandResult(0, "from the box"))

    tools = {tool.name: tool for tool in default_tools(workspace, executor)}

    assert tools["run_command"].run({"command": "ls"}) == "exit code: 0\nfrom the box"
    assert executor.commands == [("ls", 60)]


def test_the_system_prompt_says_where_commands_run():
    assert "Commands run inside a test box." in system_prompt(FakeExecutor())


# --- The local executor: this machine, no sandbox ---


def python(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_the_local_executor_names_this_operating_system(workspace):
    where = LocalExecutor(workspace.root).where

    assert where == f"on {platform.system()} through the system shell"
    assert where in system_prompt()


def test_the_local_executor_returns_exit_code_and_output(workspace):
    result = LocalExecutor(workspace.root).run(python("print('hi'); exit(4)"), 30)

    assert result.exit_code == 4
    assert result.output.strip() == "hi"


def test_the_local_executor_raises_when_time_runs_out(workspace):
    with pytest.raises(CommandTimedOut):
        LocalExecutor(workspace.root).run(python("import time; time.sleep(3)"), 1)
