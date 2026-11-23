"""Working on a copy of the project, then applying the result (decisions
68 and 69).

With --sandbox the agent never touches the project folder. loopsmith
copies the project into a temporary folder, and the file tools and the
container both work there. At the end, loopsmith shows what changed and
asks before copying anything back. A command that deletes everything
deletes only the copy.

When the copy is made, loopsmith records a fingerprint (a SHA-256 hash) of
every file. Before a change is copied back, the project's file must still
match its fingerprint. If the user changed that file during the run, their
version is kept and the file is reported as a conflict instead of being
overwritten. This is optimistic locking, like a @Version column in JPA:
assume nobody else changes the file, and check before writing.
"""

import difflib
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from loopsmith.secret_files import SKIPPED_FOLDERS, is_secrets_file

# The review shows each file's diff up to this many lines, and this many in
# all. The rest can be read in the copy itself.
MAX_DIFF_LINES_PER_FILE = 200
MAX_DIFF_LINES = 1_000


@dataclass(frozen=True)
class Changes:
    """What the agent changed in the copy, as paths relative to the project
    with "/" between folders on every system."""

    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    # Secrets files that appeared in the copy. They are never copied back.
    skipped: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)


@dataclass(frozen=True)
class ApplyReport:
    applied: tuple[str, ...] = ()
    # Files the user changed in the project during the run: their version
    # was kept, and the agent's is still in the copy.
    conflicts: tuple[str, ...] = ()


class WorkingCopy:
    def __init__(self, project: Path, root: Path, fingerprints: dict[str, str]):
        self.project = project  # the real project folder
        self.root = root  # the copy the agent works on
        self._fingerprints = fingerprints  # relative path -> SHA-256 when copied

    @property
    def file_count(self) -> int:
        return len(self._fingerprints)

    @classmethod
    def create(cls, project: Path) -> "WorkingCopy":
        """Copy the project into a new temporary folder.

        Left out: tool folders (.git, .venv, node_modules...), secrets files,
        and symbolic links, which can point outside the project. The
        project's .git is mounted into the container separately, read-only.
        """
        project = project.resolve()
        # resolve() gives the one full spelling of the folder. Windows can
        # hand out the temporary folder in a short form such as
        # C:\Users\RUNNER~1\..., while code that resolves the path sees
        # C:\Users\runneradmin\...: two spellings of one folder, which CI
        # caught (decision 70).
        root = Path(tempfile.mkdtemp(prefix="loopsmith-copy-")).resolve()
        fingerprints = {}
        for relative, source in _files(project):
            if is_secrets_file(source):
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)  # copy2 keeps the file's times and permissions
            fingerprints[relative] = _fingerprint(target)
        return cls(project, root, fingerprints)

    def changes(self) -> Changes:
        """Compare the copy with the fingerprints taken when it was made."""
        added, modified, skipped = [], [], []
        seen = set()
        for relative, path in _files(self.root):
            if is_secrets_file(path):
                skipped.append(relative)
                continue
            seen.add(relative)
            if relative not in self._fingerprints:
                added.append(relative)
            elif _fingerprint(path) != self._fingerprints[relative]:
                modified.append(relative)
        deleted = [relative for relative in self._fingerprints if relative not in seen]
        return Changes(
            added=tuple(sorted(added)),
            modified=tuple(sorted(modified)),
            deleted=tuple(sorted(deleted)),
            skipped=tuple(sorted(skipped)),
        )

    def review(self, changes: Changes) -> str:
        """A list of the changes, then a diff of every changed or added file."""
        lines = [f"Changes made in the copy ({_summary(changes)}):"]
        for label, group in (
            ("changed", changes.modified),
            ("added", changes.added),
            ("deleted", changes.deleted),
        ):
            lines += [f"  {label:<8} {relative}" for relative in group]
        lines += [f"  not copied back (a secrets file): {relative}" for relative in changes.skipped]

        diff = []
        for relative in changes.modified + changes.added:
            diff += self._diff(relative)
        if len(diff) > MAX_DIFF_LINES:
            omitted = len(diff) - MAX_DIFF_LINES
            diff = diff[:MAX_DIFF_LINES] + [
                f"[... {omitted:,} more diff lines. The full files are in {self.root}]"
            ]
        if diff:
            lines += ["", *diff]
        return "\n".join(lines)

    def apply(self, changes: Changes) -> ApplyReport:
        """Copy the changes back into the project, file by file.

        A changed or deleted file is only touched if the project's file
        still matches its fingerprint; an added file only if the project
        does not have one at that path yet. Anything else is a conflict,
        and the user's version stays.
        """
        applied, conflicts = [], []
        deleted = set(changes.deleted)
        for relative in changes.modified + changes.deleted:
            target = self.project / relative
            if _fingerprint_if_any(target) != self._fingerprints[relative]:
                conflicts.append(relative)
            elif relative in deleted:
                target.unlink()
                applied.append(relative)
            else:
                shutil.copy2(self.root / relative, target)
                applied.append(relative)
        for relative in changes.added:
            target = self.project / relative
            if target.exists():
                conflicts.append(relative)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / relative, target)
                applied.append(relative)
        return ApplyReport(applied=tuple(sorted(applied)), conflicts=tuple(sorted(conflicts)))

    def remove(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _diff(self, relative: str) -> list[str]:
        old = _text(self.project / relative) if relative in self._fingerprints else ""
        new = _text(self.root / relative)
        if old is None or new is None:
            return [f"--- {relative}: a binary file, not shown"]
        lines = list(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                f"a/{relative}",
                f"b/{relative}",
                lineterm="",
            )
        )
        if len(lines) > MAX_DIFF_LINES_PER_FILE:
            omitted = len(lines) - MAX_DIFF_LINES_PER_FILE
            lines = lines[:MAX_DIFF_LINES_PER_FILE] + [f"[... {omitted:,} more lines of this file]"]
        return lines


def _files(root: Path) -> Iterator[tuple[str, Path]]:
    """Every ordinary file under `root`, with its path relative to `root`.

    Skips tool folders and symbolic links. A function with `yield` is a
    generator: it hands out one item at a time, like a Java Iterator.
    """
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [name for name in subfolders if name not in SKIPPED_FOLDERS]
        for name in files:
            path = Path(folder) / name
            if name in SKIPPED_FOLDERS or path.is_symlink():
                continue
            yield path.relative_to(root).as_posix(), path


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint_if_any(path: Path) -> str | None:
    return _fingerprint(path) if path.is_file() else None


def _text(path: Path) -> str | None:
    """The file as text, "" if it is missing, None if it is not text."""
    if not path.is_file():
        return ""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def _summary(changes: Changes) -> str:
    parts = [
        f"{len(group)} {label}"
        for label, group in (
            ("changed", changes.modified),
            ("added", changes.added),
            ("deleted", changes.deleted),
        )
        if group
    ]
    return ", ".join(parts) or "no changes"
