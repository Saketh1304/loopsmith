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


class ConfigError(Exception):
    """Raised when a required setting is missing."""


def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from a file into the process environment.

    A variable that is already set in the real environment wins over the
    file, so a deployed setup can override local defaults.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    """Return the model API key, or raise ConfigError with a fix."""
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise ConfigError(
            f"{API_KEY_VAR} is not set. Copy .env.example to .env and "
            "put your key in it."
        )
    return key


def get_model_name() -> str:
    """Return the model to use: GEMINI_MODEL if set, else the default."""
    return os.environ.get(MODEL_VAR, "").strip() or DEFAULT_MODEL
