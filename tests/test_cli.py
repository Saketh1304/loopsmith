import pytest

from loopsmith.cli import SYSTEM_PROMPT, main
from loopsmith.config import API_KEY_VAR
from loopsmith.llm import Message, ModelError
from loopsmith.llm.fake import FakeClient


class FailingClient:
    def generate(self, messages, system=None):
        raise ModelError("Rate limit reached.", status=429, retryable=True)


class ListingClient(FakeClient):
    def list_models(self):
        return ["model-a", "model-b"]


def test_sends_task_to_model_and_prints_reply(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(["Hello from the model"])

    exit_code = main(["say hello"], client=client)

    assert exit_code == 0
    assert "Hello from the model" in capsys.readouterr().out
    assert client.calls == [([Message("user", "say hello")], SYSTEM_PROMPT)]


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
