import pytest

from loopsmith.config import (
    API_KEY_VAR,
    HOME_ENV_FILE,
    ConfigError,
    find_env_file,
    get_api_key,
    load_env_file,
)


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


def test_model_name_defaults_when_unset_or_empty(monkeypatch):
    from loopsmith.config import DEFAULT_MODEL, MODEL_VAR, get_model_name

    monkeypatch.delenv(MODEL_VAR, raising=False)
    assert get_model_name() == DEFAULT_MODEL

    monkeypatch.setenv(MODEL_VAR, "  ")
    assert get_model_name() == DEFAULT_MODEL

    monkeypatch.setenv(MODEL_VAR, "custom-model")
    assert get_model_name() == "custom-model"


# --- Locating the settings file ---


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    return home


def test_env_file_is_found_in_the_starting_folder(tmp_path, fake_home):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("A=1\n")

    assert find_env_file(project) == project / ".env"


def test_env_file_is_found_in_a_parent_folder(tmp_path, fake_home):
    project = tmp_path / "project"
    deep = project / "examples" / "demo"
    deep.mkdir(parents=True)
    (project / ".env").write_text("A=1\n")

    assert find_env_file(deep) == project / ".env"


def test_home_file_is_the_fallback(tmp_path, fake_home):
    project = tmp_path / "project"
    project.mkdir()
    (fake_home / HOME_ENV_FILE).write_text("A=1\n")

    assert find_env_file(project) == fake_home / HOME_ENV_FILE


def test_project_file_wins_over_home_file(tmp_path, fake_home):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("A=1\n")
    (fake_home / HOME_ENV_FILE).write_text("A=2\n")

    assert find_env_file(project) == project / ".env"


def test_no_file_anywhere_returns_none(tmp_path, fake_home):
    project = tmp_path / "project"
    project.mkdir()

    assert find_env_file(project) is None
