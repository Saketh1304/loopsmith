"""Tests for run_command. Commands use the current Python interpreter so
they behave the same on Windows, macOS and Linux."""

import sys

import pytest

from loopsmith.config import API_KEY_VAR
from loopsmith.tools import ToolError, Workspace
from loopsmith.tools.shell import MAX_OUTPUT_CHARS, run_command_tool


def python(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path)


def test_returns_exit_code_and_output(workspace):
    result = run_command_tool(workspace).run({"command": python("print('hello')")})

    assert result == "exit code: 0\nhello"


def test_failing_command_is_a_normal_result_with_its_output(workspace):
    code = "import sys; print('boom', file=sys.stderr); sys.exit(3)"

    result = run_command_tool(workspace).run({"command": python(code)})

    assert result == "exit code: 3\nboom"


def test_runs_inside_the_workspace_folder(workspace):
    (workspace.root / "marker.txt").write_text("x", encoding="utf-8")

    result = run_command_tool(workspace).run(
        {"command": python("import os; print(sorted(os.listdir('.')))")}
    )

    assert "marker.txt" in result


def test_api_key_is_not_passed_to_the_command(workspace, monkeypatch):
    monkeypatch.setenv(API_KEY_VAR, "super-secret")
    code = f"import os; print(os.environ.get('{API_KEY_VAR}'))"

    result = run_command_tool(workspace).run({"command": python(code)})

    assert "super-secret" not in result
    assert "None" in result


def test_command_waiting_for_input_gets_none(workspace):
    code = "import sys; print(repr(sys.stdin.read()))"

    result = run_command_tool(workspace).run({"command": python(code)})

    assert result == "exit code: 0\n''"


def test_slow_command_is_stopped(workspace):
    tool = run_command_tool(workspace, timeout_seconds=1)

    with pytest.raises(ToolError, match="within 1 seconds"):
        tool.run({"command": python("import time; time.sleep(3)")})


def test_long_output_keeps_the_start_and_the_end(workspace):
    code = f"print('START' + 'x' * {MAX_OUTPUT_CHARS * 2} + 'END')"

    result = run_command_tool(workspace).run({"command": python(code)})

    assert "START" in result
    assert result.endswith("END")
    assert "characters left out" in result
    assert len(result) < MAX_OUTPUT_CHARS + 200


def test_no_output_is_reported(workspace):
    assert run_command_tool(workspace).run({"command": python("pass")}) == (
        "exit code: 0\n(no output)"
    )


def test_preview_shows_the_command(workspace):
    assert run_command_tool(workspace).preview({"command": "pytest -q"}) == (
        "Run command: pytest -q"
    )


def test_command_is_required(workspace):
    with pytest.raises(ToolError, match="command"):
        run_command_tool(workspace).preview({})
