import pytest

from loopsmith.cli import main
from loopsmith.config import API_KEY_VAR


def test_runs_when_key_is_set(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "abc123")

    exit_code = main(["fix the bug"])

    assert exit_code == 0
    assert "fix the bug" in capsys.readouterr().out


def test_fails_clearly_without_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_VAR, raising=False)

    exit_code = main(["fix the bug"])

    assert exit_code == 1
    assert API_KEY_VAR in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert "loopsmith" in capsys.readouterr().out
