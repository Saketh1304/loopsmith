"""Read-only file tools, confined to one folder (the workspace)."""

from pathlib import Path

from loopsmith.tools.base import Tool, ToolError, require_string

MAX_FILE_CHARS = 40_000
MAX_LIST_ENTRIES = 200
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

    def display(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


def is_secrets_file(path: Path) -> bool:
    """True for .env files, except the key-free .env.example template."""
    name = path.name.lower()
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def read_file_tool(workspace: Workspace) -> Tool:
    def run(args: dict) -> str:
        path = workspace.resolve(require_string(args, "path"))
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
        if len(text) > MAX_FILE_CHARS:
            omitted = len(text) - MAX_FILE_CHARS
            return f"{text[:MAX_FILE_CHARS]}\n[... truncated, {omitted} more characters]"
        return text

    return Tool(
        name="read_file",
        description="Read a text file in the project and return its contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project folder.",
                }
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


def default_tools(workspace: Workspace) -> list[Tool]:
    return [list_files_tool(workspace), read_file_tool(workspace)]
