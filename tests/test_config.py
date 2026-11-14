import pytest

from loopsmith.config import API_KEY_VAR, ConfigError, get_api_key, load_env_file


def test_env_file_values_are_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"# a comment\n\n{API_KEY_VAR}=abc123\n")

    load_env_file(env_file)

    assert get_api_key() == "abc123"


def test_quotes_and_spaces_are_stripped(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f'{API_KEY_VAR} = "abc123" \n')

    load_env_file(env_file)

    assert get_api_key() == "abc123"


def test_real_environment_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_VAR, "from-environment")
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_VAR}=from-file\n")

    load_env_file(env_file)

    assert get_api_key() == "from-environment"


def test_missing_file_is_ignored(tmp_path):
    load_env_file(tmp_path / "does-not-exist")


def test_missing_key_raises_helpful_error(monkeypatch):
    monkeypatch.delenv(API_KEY_VAR, raising=False)

    with pytest.raises(ConfigError, match=".env.example"):
        get_api_key()
