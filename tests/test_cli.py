import io

import pytest

from loopsmith.cli import ask_on_terminal, main, system_prompt
from loopsmith.config import API_KEY_VAR
from loopsmith.llm import Message, ModelError, ToolCall
from loopsmith.llm.fake import FakeClient


class FailingClient:
    def generate(self, messages, system=None, tools=None):
        raise ModelError("Rate limit reached.", status=429, retryable=True)


class ListingClient(FakeClient):
    def list_models(self):
        return ["model-a", "model-b"]


def test_sends_task_to_model_and_prints_reply(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(["Hello from the model"])

    exit_code = main(["say hello"], client=client)

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "Hello from the model"
    history, system, tools = client.calls[0]
    assert history == [Message("user", "say hello")]
    assert system == system_prompt()
    assert {tool.name for tool in tools} == {
        "list_files",
        "read_file",
        "edit_file",
        "write_file",
        "run_command",
    }


def test_agent_reads_a_real_file_from_the_current_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("the secret word is mango", encoding="utf-8")
    client = FakeClient([ToolCall("read_file", {"path": "notes.txt"}), "The word is mango."])

    exit_code = main(["what is the secret word?"], client=client)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "The word is mango."
    # The tool trace goes to stderr, keeping stdout clean.
    assert "read_file(path='notes.txt')" in captured.err
    # The model received the file's contents as the tool result.
    tool_message = client.calls[1][0][-1]
    assert tool_message.tool_results[0].content == "the secret word is mango"


def test_hitting_the_step_limit_exits_with_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([ToolCall("list_files", {}) for _ in range(5)])

    exit_code = main(["loop", "--max-steps", "2"], client=client)

    assert exit_code == 1
    assert "Stopped after 2 steps" in capsys.readouterr().out


def test_model_error_is_reported_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["say hello"], client=FailingClient())

    assert exit_code == 1
    assert "Rate limit reached" in capsys.readouterr().err


def test_fails_clearly_without_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    # Pretend no settings file exists anywhere on this machine.
    monkeypatch.setattr("loopsmith.config.find_env_file", lambda start=None: None)

    exit_code = main(["say hello"])

    assert exit_code == 1
    assert API_KEY_VAR in capsys.readouterr().err


def test_list_models_prints_names(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--list-models"], client=ListingClient([]))

    assert exit_code == 0
    assert capsys.readouterr().out.split() == ["model-a", "model-b"]


def test_no_arguments_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert "loopsmith" in capsys.readouterr().out


# --- Approval ---


class FakeTerminal(io.StringIO):
    """Stands in for the keyboard: a stdin that reports being a terminal."""

    def isatty(self):
        return True


WRITE_CALL = ToolCall("write_file", {"path": "new.txt", "content": "hello\n"})


def test_yes_flag_lets_changes_through(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([WRITE_CALL, "Created the file."])

    exit_code = main(["create new.txt", "--yes"], client=client)

    assert exit_code == 0
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_user_can_approve_a_change_on_the_terminal(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", FakeTerminal("y\n"))
    client = FakeClient([WRITE_CALL, "Created the file."])

    main(["create new.txt"], client=client)

    assert (tmp_path / "new.txt").exists()
    shown = capsys.readouterr().err
    assert "Create file new.txt" in shown
    assert "+hello" in shown
    assert "Allow? [y/N]" in shown


@pytest.mark.parametrize("answer", ["n\n", "\n", "maybe\n", ""])
def test_anything_but_yes_declines_the_change(monkeypatch, tmp_path, capsys, answer):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", FakeTerminal(answer))
    client = FakeClient([WRITE_CALL, "I did not create it."])

    main(["create new.txt"], client=client)

    assert not (tmp_path / "new.txt").exists()
    assert "declined" in client.calls[1][0][-1].tool_results[0].content


def test_without_a_terminal_changes_are_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))  # a pipe, not a terminal
    client = FakeClient([WRITE_CALL, "I could not create it."])

    main(["create new.txt"], client=client)

    assert not (tmp_path / "new.txt").exists()
    assert "--yes" in capsys.readouterr().err


def test_ask_on_terminal_accepts_yes_in_any_case(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", FakeTerminal("YES\n"))

    assert ask_on_terminal(WRITE_CALL, "Create file new.txt") is True
