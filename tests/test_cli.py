import pytest

from loopsmith.cli import SYSTEM_PROMPT, main
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
    assert system == SYSTEM_PROMPT
    assert {tool.name for tool in tools} == {"list_files", "read_file"}


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
