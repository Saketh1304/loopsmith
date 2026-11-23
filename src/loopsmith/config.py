"""Configuration: reads settings such as the API key.

Settings come from environment variables. For convenience during
development, they can also be placed in a ".env" file, which is loaded
into the environment at startup and is never committed to git.
"""

import os
from pathlib import Path

API_KEY_VAR = "GEMINI_API_KEY"
MODEL_VAR = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-3.5-flash"

# The only settings loopsmith reads from a file. A project's own .env can
# hold anything (database passwords, other keys); none of it is loopsmith's
# business, so none of it is loaded (decision 57).
OWN_SETTINGS = (API_KEY_VAR, MODEL_VAR)


class ConfigError(Exception):
    """Raised when a required setting is missing."""


HOME_ENV_FILE = ".loopsmith.env"


def read_settings(path: Path) -> dict[str, str]:
    """loopsmith's own settings from a KEY=VALUE file. Other lines are
    ignored, and so is a file that cannot be read as text."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    settings = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in OWN_SETTINGS:
            settings[key] = value.strip().strip("\"'")
    return settings


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate the settings file: the first one that sets the API key.

    Looks for ".env" in the starting folder and then in each parent
    folder, so the agent works from a subfolder of a project. Falls back
    to ".loopsmith.env" in the user's home folder, which lets the agent
    run inside any project on the machine. A ".env" without the key is
    passed over: it belongs to the project, not to loopsmith.
    """
    folder = (start or Path.cwd()).resolve()
    candidates = [place / ".env" for place in (folder, *folder.parents)]
    candidates.append(Path.home() / HOME_ENV_FILE)
    for candidate in candidates:
        if candidate.is_file() and API_KEY_VAR in read_settings(candidate):
            return candidate
    return None


def load_env_file(path: Path | None = None) -> None:
    """Load loopsmith's settings from a file into the process environment.

    With no path given, the file is located by `find_env_file`. A
    variable that is already set in the real environment wins over the
    file, so a deployed setup can override local defaults.
    """
    if path is None:
        path = find_env_file()
    if path is None or not path.is_file():
        return
    for key, value in read_settings(path).items():
        if key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    """Return the model API key, or raise ConfigError with a fix."""
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise ConfigError(
            f"{API_KEY_VAR} is not set. Copy .env.example to .env and "
            "put your key in it, or put it in ~/.loopsmith.env to use "
            "loopsmith in any folder."
        )
    return key


def get_model_name() -> str:
    """Return the model to use: GEMINI_MODEL if set, else the default."""
    return os.environ.get(MODEL_VAR, "").strip() or DEFAULT_MODEL
