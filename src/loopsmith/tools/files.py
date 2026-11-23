"""File tools, confined to one folder (the workspace)."""

import difflib
from pathlib import Path

from loopsmith.tools.base import Tool, ToolError, require_string

MAX_FILE_CHARS = 40_000
MAX_LIST_ENTRIES = 200
MAX_PREVIEW_LINES = 60
SKIPPED_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}


class Workspace:
    """The one folder the agent is allowed to touch.

    Every path the model supplies goes through `resolve`, which refuses
    anything outside the root. Without this check the model could ask
    for "../../some/private/file".
    """

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, relative_path: str) -> Path:
        # resolve() follows ".." and symlinks, so the check below is made
        # against the real final location.
        target = (self.root / relative_path).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(
                f"Path '{relative_path}' is outside the workspace. "
                "Use paths relative to the project folder."
            )
        return target

    def resolve_for_writing(self, relative_path: str) -> Path:
        target = self.resolve(relative_path)
        if is_secrets_file(target):
            raise ToolError(f"'{target.name}' holds secrets and cannot be changed.")
        if ".git" in target.relative_to(self.root).parts:
            raise ToolError("Files inside .git cannot be changed.")
        if target.is_dir():
            raise ToolError(f"'{self.display(target)}' is a folder, not a file.")
        return target

    def display(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


def is_secrets_file(path: Path) -> bool:
    """True for .env files, except the key-free .env.example template."""
    name = path.name.lower()
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


# --- Reading and writing text while keeping the file's line endings ---


def _read_keeping_line_endings(path: Path) -> tuple[str, bool]:
    """Return (text with "\\n" line endings, whether the file used "\\r\\n").

    Windows files often end lines with "\\r\\n". The model writes "\\n".
    Text is normalised to "\\n" for matching, and the original style is
    restored on save so an edit does not rewrite every line of the file.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw = handle.read()
    except UnicodeDecodeError:
        raise ToolError(f"'{path.name}' is not a text file.") from None
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def _write_with_line_endings(path: Path, text: str, use_crlf: bool) -> None:
    if use_crlf:
        text = text.replace("\n", "\r\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _diff(old: str, new: str, name: str) -> str:
    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm=""
        )
    )
    if len(lines) > MAX_PREVIEW_LINES:
        omitted = len(lines) - MAX_PREVIEW_LINES
        lines = lines[:MAX_PREVIEW_LINES] + [f"[... {omitted} more diff lines]"]
    return "\n".join(lines)


# --- Read-only tools ---


def _optional_line(args: dict, name: str) -> int | None:
    """Read an optional line number (counted from 1), or raise a ToolError."""
    value = args.get(name)
    if value is None:
        return None
    # JSON has only one kind of number, so a whole number can arrive as
    # 40.0. In Python, True and False are also ints, so bool is refused
    # by name.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"Argument '{name}' must be a whole number.")
    if value < 1:
        raise ToolError(f"Argument '{name}' must be 1 or more: lines are counted from 1.")
    return value


def _capped(text: str, first_line: int = 1) -> str:
    """Cut text that is too long to send, and say where to read on from."""
    if len(text) <= MAX_FILE_CHARS:
        return text
    shown = text[:MAX_FILE_CHARS]
    omitted = len(text) - MAX_FILE_CHARS
    # The line the cut fell in. Reading on from it repeats part of that
    # line, which is safer than skipping the rest of it.
    next_line = first_line + shown.count("\n")
    return (
        f"{shown}\n[... truncated, {omitted:,} more characters. "
        f"Read on with start_line={next_line}.]"
    )


def _line_range(text: str, start: int, end: int | None, name: str) -> str:
    """Lines `start` to `end` of the text, both included, under a header
    that says where they are, so the model knows how much it has not seen."""
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if start > total:
        raise ToolError(f"'{name}' has {total} lines, so start_line {start} is past its end.")
    if end is not None and end < start:
        raise ToolError(f"end_line ({end}) is before start_line ({start}).")
    end = total if end is None else min(end, total)
    header = f"[{name}: lines {start} to {end} of {total}]\n"
    return header + _capped("".join(lines[start - 1 : end]), first_line=start)


def read_file_tool(workspace: Workspace) -> Tool:
    def run(args: dict) -> str:
        path = workspace.resolve(require_string(args, "path"))
        start = _optional_line(args, "start_line")
        end = _optional_line(args, "end_line")
        if is_secrets_file(path):
            # File contents are sent to the model provider. Keys must not be.
            raise ToolError(f"'{path.name}' holds secrets and cannot be read.")
        if path.is_dir():
            raise ToolError(f"'{workspace.display(path)}' is a folder. Use list_files on it.")
        if not path.is_file():
            raise ToolError(f"File '{workspace.display(path)}' does not exist.")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"'{workspace.display(path)}' is not a text file.") from None
        if start is None and end is None:
            return _capped(text)
        # Reading part of a file keeps a large file from filling the
        # history, where it would be sent again on every later step.
        return _line_range(text, start or 1, end, workspace.display(path))

    return Tool(
        name="read_file",
        description=(
            "Read a text file in the project. Returns the whole file, or only "
            "the lines from start_line to end_line when they are given. Lines "
            "are counted from 1. For a long file, read just the part you need."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project folder.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read, counting from 1. "
                    "Leave out to start at the top.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read, included. Leave out to read to the end.",
                },
            },
            "required": ["path"],
        },
        run=run,
    )


def list_files_tool(workspace: Workspace) -> Tool:
    def run(args: dict) -> str:
        relative = args.get("path") or "."
        if not isinstance(relative, str):
            raise ToolError("Argument 'path' must be a string.")
        folder = workspace.resolve(relative)
        if not folder.is_dir():
            raise ToolError(f"Folder '{relative}' does not exist.")

        names = []
        for entry in sorted(folder.iterdir(), key=lambda e: e.name.lower()):
            if entry.name in SKIPPED_NAMES:
                continue
            names.append(entry.name + "/" if entry.is_dir() else entry.name)

        if not names:
            return "(empty folder)"
        if len(names) > MAX_LIST_ENTRIES:
            omitted = len(names) - MAX_LIST_ENTRIES
            names = names[:MAX_LIST_ENTRIES] + [f"[... {omitted} more entries]"]
        return "\n".join(names)

    return Tool(
        name="list_files",
        description=(
            "List the files and folders directly inside a project folder. "
            "Folder names end with '/'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Folder path relative to the project folder. "
                    "Leave out for the project root.",
                }
            },
        },
        run=run,
    )


# --- Tools that change files (these need approval) ---


def write_file_tool(workspace: Workspace) -> Tool:
    def plan(args: dict) -> tuple[Path, str, str, bool]:
        """Validate the arguments and work out (path, old, new, crlf)."""
        path = workspace.resolve_for_writing(require_string(args, "path"))
        new = require_string(args, "content", allow_empty=True)
        if path.is_file():
            old, use_crlf = _read_keeping_line_endings(path)
        else:
            old, use_crlf = "", False
        return path, old, new, use_crlf

    def preview(args: dict) -> str:
        path, old, new, _ = plan(args)
        name = workspace.display(path)
        action = "Overwrite" if path.is_file() else "Create"
        return f"{action} file {name}\n{_diff(old, new, name)}"

    def run(args: dict) -> str:
        path, _, new, use_crlf = plan(args)
        existed = path.is_file()
        _write_with_line_endings(path, new, use_crlf)
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {workspace.display(path)} ({len(new.splitlines())} lines)."

    return Tool(
        name="write_file",
        description=(
            "Create a new file, or replace the whole contents of an existing "
            "one. For a small change to an existing file, use edit_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project folder.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file.",
                },
            },
            "required": ["path", "content"],
        },
        run=run,
        preview=preview,
    )


def edit_file_tool(workspace: Workspace) -> Tool:
    def plan(args: dict) -> tuple[Path, str, str, bool]:
        path = workspace.resolve_for_writing(require_string(args, "path"))
        old_text = require_string(args, "old_text")
        new_text = require_string(args, "new_text", allow_empty=True)
        if not path.is_file():
            raise ToolError(
                f"File '{workspace.display(path)}' does not exist. "
                "Use write_file to create it."
            )
        old_text = old_text.replace("\r\n", "\n")
        new_text = new_text.replace("\r\n", "\n")
        content, use_crlf = _read_keeping_line_endings(path)

        count = content.count(old_text)
        if count == 0:
            raise ToolError(
                "old_text was not found in the file. It must match exactly, "
                "including spaces and line breaks. Read the file again and copy "
                "the text from it."
            )
        if count > 1:
            raise ToolError(
                f"old_text appears {count} times in the file. Include more "
                "surrounding lines so that it matches exactly one place."
            )
        return path, content, content.replace(old_text, new_text, 1), use_crlf

    def preview(args: dict) -> str:
        path, old, new, _ = plan(args)
        name = workspace.display(path)
        return f"Edit file {name}\n{_diff(old, new, name)}"

    def run(args: dict) -> str:
        path, _, new, use_crlf = plan(args)
        _write_with_line_endings(path, new, use_crlf)
        return f"Edited {workspace.display(path)}."

    return Tool(
        name="edit_file",
        description=(
            "Change part of an existing file by replacing one exact piece of "
            "text with another. old_text must appear exactly once in the file. "
            "Read the file first so that old_text matches it exactly."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project folder.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to replace. Must be unique in the file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to put in its place. Empty to delete.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        run=run,
        preview=preview,
    )
