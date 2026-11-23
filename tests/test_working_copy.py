"""Tests for working on a copy of the project (decisions 68 and 69).

Files are written with write_bytes, so their exact bytes are the same on
Windows and Linux (decision 29).
"""

import os
import tempfile
from pathlib import Path

import pytest

from loopsmith.sandbox.working_copy import (
    MAX_DIFF_LINES,
    MAX_DIFF_LINES_PER_FILE,
    ApplyReport,
    Changes,
    WorkingCopy,
)
from loopsmith.tools import Workspace


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for relative, content in {
        "stats.py": b"def mean(xs):\n    return sum(xs) / len(xs)\n",
        "notes.txt": b"untracked notes\n",
        "docs/guide.md": b"# Guide\n",
        "windows.txt": b"one\r\ntwo\r\n",
        ".env": b"GEMINI_API_KEY=real-looking\n",
        ".git/HEAD": b"ref: refs/heads/main\n",
        ".venv/lib/big.py": b"x = 1\n",
        "node_modules/pkg/index.js": b"module.exports = 1\n",
        "__pycache__/stats.cpython-312.pyc": b"\x00\x01",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.fixture
def copy(project):
    working_copy = WorkingCopy.create(project)
    yield working_copy
    working_copy.remove()


def snapshot(root):
    """Every file under root and its bytes, to show nothing changed."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


# --- Making the copy ---


def test_the_copy_has_the_project_files_but_not_secrets_or_tool_folders(copy, project):
    copied = snapshot(copy.root)

    assert sorted(copied) == ["docs/guide.md", "notes.txt", "stats.py", "windows.txt"]
    assert copied["windows.txt"] == b"one\r\ntwo\r\n"  # exact bytes, line endings kept
    assert copy.file_count == 4
    assert copy.root != project
    assert copy.project == project.resolve()


def test_the_copy_folder_has_one_spelling(project, tmp_path, monkeypatch):
    # Windows can name the temporary folder two ways (RUNNER~1 and
    # runneradmin on GitHub's machines), which once failed CI. A path with
    # "x/.." in it is another spelling of one folder, and works anywhere.
    real_mkdtemp = tempfile.mkdtemp
    (tmp_path / "x").mkdir()

    def roundabout_mkdtemp(prefix=""):
        made = Path(real_mkdtemp(prefix=prefix, dir=tmp_path))
        return str(tmp_path / "x" / ".." / made.name)

    monkeypatch.setattr(tempfile, "mkdtemp", roundabout_mkdtemp)
    working_copy = WorkingCopy.create(project)
    try:
        assert ".." not in working_copy.root.parts
        # The path loopsmith prints is the one the file tools work in.
        assert Workspace(working_copy.root).root == working_copy.root
    finally:
        working_copy.remove()


def test_symbolic_links_are_not_copied(project, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"private\n")
    try:
        os.symlink(outside, project / "link.txt")
    except OSError:
        pytest.skip("this system does not allow symbolic links")

    working_copy = WorkingCopy.create(project)
    try:
        assert not (working_copy.root / "link.txt").exists()
    finally:
        working_copy.remove()


def test_a_fresh_copy_has_no_changes(copy):
    assert copy.changes() == Changes()
    assert copy.changes().count == 0


# --- Finding what changed ---


def test_added_changed_and_deleted_files_are_found(copy):
    (copy.root / "stats.py").write_bytes(b"def mean(xs):\n    return sum(xs) / max(len(xs), 1)\n")
    (copy.root / "tests").mkdir()
    (copy.root / "tests" / "test_stats.py").write_bytes(b"def test_mean():\n    pass\n")
    (copy.root / "notes.txt").unlink()

    changes = copy.changes()

    assert changes.modified == ("stats.py",)
    assert changes.added == ("tests/test_stats.py",)
    assert changes.deleted == ("notes.txt",)
    assert changes.count == 3


def test_tool_folders_made_during_the_run_are_not_changes(copy):
    # Running Python and pytest in the copy leaves these behind.
    (copy.root / "__pycache__").mkdir()
    (copy.root / "__pycache__" / "stats.pyc").write_bytes(b"\x00")
    (copy.root / ".pytest_cache").mkdir()
    (copy.root / ".pytest_cache" / "README.md").write_bytes(b"cache\n")

    assert copy.changes() == Changes()


def test_a_secrets_file_made_in_the_copy_is_never_copied_back(copy, project):
    (copy.root / ".env.local").write_bytes(b"TOKEN=made-by-a-command\n")

    changes = copy.changes()

    assert changes.skipped == (".env.local",)
    assert changes.count == 0
    copy.apply(changes)
    assert not (project / ".env.local").exists()


# --- The review ---


def test_the_review_lists_the_changes_and_shows_their_diffs(copy):
    (copy.root / "stats.py").write_bytes(b"def mean(xs):\n    return sum(xs) / max(len(xs), 1)\n")
    (copy.root / "new.txt").write_bytes(b"hello\n")
    (copy.root / "notes.txt").unlink()

    review = copy.review(copy.changes())

    assert review.startswith("Changes made in the copy (1 changed, 1 added, 1 deleted):")
    assert "  changed  stats.py" in review
    assert "  added    new.txt" in review
    assert "  deleted  notes.txt" in review
    assert "--- a/stats.py" in review
    assert "-    return sum(xs) / len(xs)" in review
    assert "+    return sum(xs) / max(len(xs), 1)" in review
    assert "+hello" in review  # an added file shows all of its lines


def test_a_binary_file_is_named_but_not_shown(copy):
    (copy.root / "image.png").write_bytes(b"\x89PNG\x00\xff")

    assert "--- image.png: a binary file, not shown" in copy.review(copy.changes())


def test_long_diffs_are_cut_per_file_and_in_all(copy):
    for number in range(8):
        lines = "".join(f"line {i}\n" for i in range(400))
        (copy.root / f"big{number}.txt").write_bytes(lines.encode("utf-8"))

    review = copy.review(copy.changes())

    assert "more lines of this file]" in review
    assert "more diff lines. The full files are in" in review
    assert len(review.splitlines()) < MAX_DIFF_LINES + 50
    assert MAX_DIFF_LINES_PER_FILE < MAX_DIFF_LINES


# --- Applying ---


def test_applying_copies_the_changes_into_the_project(copy, project):
    (copy.root / "stats.py").write_bytes(b"new mean\n")
    (copy.root / "tests").mkdir()
    (copy.root / "tests" / "test_stats.py").write_bytes(b"new test\n")
    (copy.root / "notes.txt").unlink()

    report = copy.apply(copy.changes())

    assert report == ApplyReport(
        applied=("notes.txt", "stats.py", "tests/test_stats.py"), conflicts=()
    )
    assert (project / "stats.py").read_bytes() == b"new mean\n"
    assert (project / "tests" / "test_stats.py").read_bytes() == b"new test\n"
    assert not (project / "notes.txt").exists()
    # What the agent did not touch is still there, secrets included.
    assert (project / ".env").read_bytes() == b"GEMINI_API_KEY=real-looking\n"
    assert (project / "docs" / "guide.md").read_bytes() == b"# Guide\n"


def test_nothing_reaches_the_project_until_it_is_applied(copy, project):
    before = snapshot(project)
    for path in list(copy.root.rglob("*")):
        if path.is_file():
            path.unlink()  # the copy is wrecked...

    assert copy.changes().count == 4
    assert snapshot(project) == before  # ...and the project is untouched


def test_a_file_the_user_changed_during_the_run_is_a_conflict(copy, project):
    (copy.root / "stats.py").write_bytes(b"the agent's version\n")
    (project / "stats.py").write_bytes(b"the user's version\n")  # edited meanwhile

    report = copy.apply(copy.changes())

    assert report == ApplyReport(applied=(), conflicts=("stats.py",))
    assert (project / "stats.py").read_bytes() == b"the user's version\n"


def test_a_file_the_user_changed_is_not_deleted(copy, project):
    (copy.root / "notes.txt").unlink()
    (project / "notes.txt").write_bytes(b"the user added to these notes\n")

    report = copy.apply(copy.changes())

    assert report.conflicts == ("notes.txt",)
    assert (project / "notes.txt").exists()


def test_an_added_file_does_not_replace_one_the_user_created(copy, project):
    (copy.root / "todo.txt").write_bytes(b"the agent's list\n")
    (project / "todo.txt").write_bytes(b"the user's list\n")

    report = copy.apply(copy.changes())

    assert report.conflicts == ("todo.txt",)
    assert (project / "todo.txt").read_bytes() == b"the user's list\n"


def test_conflicts_do_not_stop_the_other_changes(copy, project):
    (copy.root / "stats.py").write_bytes(b"the agent's version\n")
    (project / "stats.py").write_bytes(b"the user's version\n")
    (copy.root / "docs" / "guide.md").write_bytes(b"# A better guide\n")

    report = copy.apply(copy.changes())

    assert report == ApplyReport(applied=("docs/guide.md",), conflicts=("stats.py",))
    assert (project / "docs" / "guide.md").read_bytes() == b"# A better guide\n"


def test_removing_the_copy_deletes_its_folder(project):
    working_copy = WorkingCopy.create(project)

    working_copy.remove()

    assert not working_copy.root.exists()
    assert (project / "stats.py").exists()
