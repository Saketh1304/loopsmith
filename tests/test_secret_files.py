"""Tests for the one rule about which files hold secrets."""

from pathlib import Path

import pytest

from loopsmith.secret_files import find_secret_files, is_secrets_file


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        (".env", True),
        (".ENV", True),
        (".env.local", True),
        (".env.production", True),
        (".env.example", False),  # the key-free template
        ("env.py", False),
        ("settings.env.txt", False),
    ],
)
def test_which_files_count_as_secrets(name, secret):
    assert is_secrets_file(Path(name)) is secret


def test_secret_files_are_found_in_every_folder_of_the_project(tmp_path):
    for relative in [
        ".env",
        ".env.example",
        "backend/.env.local",
        "backend/app.py",
        "node_modules/some-package/.env",
        ".git/.env",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    found = find_secret_files(tmp_path)

    # Not the template, and not inside folders that belong to tools.
    assert found == [tmp_path / ".env", tmp_path / "backend" / ".env.local"]


def test_a_project_without_secrets_has_none(tmp_path):
    (tmp_path / "app.py").write_text("x", encoding="utf-8")

    assert find_secret_files(tmp_path) == []
