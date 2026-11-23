"""Which files hold secrets.

One rule, used in two places: the file tools refuse to read or change
these files, and the sandbox hides them from commands. It lives in its
own module because the tools import the sandbox, so the sandbox cannot
import the tools.
"""

import os
from pathlib import Path

# Folders that belong to tools, not to the project. Secret files are not
# searched for inside them.
SKIPPED_FOLDERS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}


def is_secrets_file(path: Path) -> bool:
    """True for .env files, except the key-free .env.example template."""
    name = path.name.lower()
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def find_secret_files(root: Path) -> list[Path]:
    """Every secrets file under `root`, in a fixed order.

    os.walk visits each folder in turn, like Java's Files.walk. Changing
    the `subfolders` list in place tells it which folders to skip.
    """
    found = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [name for name in subfolders if name not in SKIPPED_FOLDERS]
        found += [Path(folder) / name for name in files if is_secrets_file(Path(name))]
    return sorted(found)
