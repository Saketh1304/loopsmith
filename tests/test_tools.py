import pytest

from loopsmith.tools import ToolError, Workspace, default_tools
from loopsmith.tools.files import (
    MAX_FILE_CHARS,
    edit_file_tool,
    list_files_tool,
    read_file_tool,
    write_file_tool,
)


@pytest.fixture
def workspace(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    # write_bytes, not write_text: on Windows, write_text turns "\n" into
    # "\r\n", and these tests check the exact bytes on disk.
    (project / "app.py").write_bytes(b"print('hi')\n")
    (project / "src").mkdir()
    (project / "src" / "util.py").write_bytes(b"x = 1\n")
    (project / ".git").mkdir()
    (project / ".env").write_text("GEMINI_API_KEY=secret\n", encoding="utf-8")
    (project / ".env.example").write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("private\n", encoding="utf-8")
    return Workspace(project)


def test_read_file_returns_contents(workspace):
    assert read_file_tool(workspace).run({"path": "app.py"}) == "print('hi')\n"


def test_read_file_in_subfolder(workspace):
    assert read_file_tool(workspace).run({"path": "src/util.py"}) == "x = 1\n"


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../outside.txt"])
def test_read_file_refuses_paths_outside_workspace(workspace, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file_tool(workspace).run({"path": path})


def test_read_file_refuses_absolute_path_outside_workspace(workspace, tmp_path):
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file_tool(workspace).run({"path": str(tmp_path / "outside.txt")})


def test_read_file_refuses_secrets_file(workspace):
    with pytest.raises(ToolError, match="secrets"):
        read_file_tool(workspace).run({"path": ".env"})


def test_read_file_allows_env_example(workspace):
    assert "GEMINI_API_KEY=" in read_file_tool(workspace).run({"path": ".env.example"})


def test_read_file_missing_file(workspace):
    with pytest.raises(ToolError, match="does not exist"):
        read_file_tool(workspace).run({"path": "nope.py"})


def test_read_file_on_folder_points_to_list_files(workspace):
    with pytest.raises(ToolError, match="list_files"):
        read_file_tool(workspace).run({"path": "src"})


def test_read_file_requires_path(workspace):
    with pytest.raises(ToolError, match="path"):
        read_file_tool(workspace).run({})


def test_read_file_rejects_binary(workspace):
    (workspace.root / "image.bin").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(ToolError, match="not a text file"):
        read_file_tool(workspace).run({"path": "image.bin"})


def test_read_file_truncates_large_files(workspace):
    (workspace.root / "big.txt").write_text("a" * (MAX_FILE_CHARS + 500), encoding="utf-8")

    result = read_file_tool(workspace).run({"path": "big.txt"})

    assert result.startswith("a" * MAX_FILE_CHARS)
    assert "500 more characters" in result


def test_list_files_shows_root_and_marks_folders(workspace):
    listing = list_files_tool(workspace).run({}).splitlines()

    assert listing == [".env", ".env.example", "app.py", "src/"]


def test_list_files_in_subfolder(workspace):
    assert list_files_tool(workspace).run({"path": "src"}) == "util.py"


def test_list_files_refuses_outside_workspace(workspace):
    with pytest.raises(ToolError, match="outside the workspace"):
        list_files_tool(workspace).run({"path": ".."})


def test_list_files_missing_folder(workspace):
    with pytest.raises(ToolError, match="does not exist"):
        list_files_tool(workspace).run({"path": "nope"})


def test_tool_specs_describe_the_tools(workspace):
    specs = {tool.name: tool.spec for tool in default_tools(workspace)}

    assert set(specs) == {"list_files", "read_file", "edit_file", "write_file", "run_command"}
    assert specs["read_file"].parameters["required"] == ["path"]


# --- Tools that change files ---


def read_bytes(workspace, name: str) -> bytes:
    return (workspace.root / name).read_bytes()


def test_changing_tools_need_approval_and_reading_tools_do_not(workspace):
    needs = {tool.name: tool.needs_approval for tool in default_tools(workspace)}

    assert needs == {
        "list_files": False,
        "read_file": False,
        "edit_file": True,
        "write_file": True,
        "run_command": True,
    }


def test_write_file_creates_a_file_and_its_folders(workspace):
    result = write_file_tool(workspace).run({"path": "docs/new/notes.md", "content": "a\nb\n"})

    assert read_bytes(workspace, "docs/new/notes.md") == b"a\nb\n"
    assert "Created docs/new/notes.md (2 lines)" in result


def test_write_file_overwrites_and_says_so(workspace):
    result = write_file_tool(workspace).run({"path": "app.py", "content": "print('bye')\n"})

    assert read_bytes(workspace, "app.py") == b"print('bye')\n"
    assert result.startswith("Overwrote app.py")


def test_write_file_preview_shows_a_diff_without_writing(workspace):
    preview = write_file_tool(workspace).preview({"path": "app.py", "content": "print('bye')\n"})

    assert preview.startswith("Overwrite file app.py")
    assert "-print('hi')" in preview
    assert "+print('bye')" in preview
    assert read_bytes(workspace, "app.py") == b"print('hi')\n"


def test_write_file_keeps_windows_line_endings_when_overwriting(workspace):
    (workspace.root / "win.py").write_bytes(b"a = 1\r\nb = 2\r\n")

    write_file_tool(workspace).run({"path": "win.py", "content": "a = 10\nb = 20\n"})

    assert read_bytes(workspace, "win.py") == b"a = 10\r\nb = 20\r\n"


def test_write_file_allows_empty_content(workspace):
    write_file_tool(workspace).run({"path": "empty.txt", "content": ""})

    assert read_bytes(workspace, "empty.txt") == b""


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("../outside.txt", "outside the workspace"),
        (".env", "secrets"),
        (".git/config", ".git"),
        ("src", "folder"),
    ],
)
def test_write_file_refuses_unsafe_targets(workspace, path, expected):
    tool = write_file_tool(workspace)

    with pytest.raises(ToolError, match=expected):
        tool.preview({"path": path, "content": "x"})
    with pytest.raises(ToolError, match=expected):
        tool.run({"path": path, "content": "x"})


def test_edit_file_replaces_the_one_matching_piece(workspace):
    (workspace.root / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")

    result = edit_file_tool(workspace).run(
        {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}
    )

    assert read_bytes(workspace, "calc.py") == b"def add(a, b):\n    return a + b\n"
    assert result == "Edited calc.py."


def test_edit_file_preview_shows_a_diff_without_writing(workspace):
    preview = edit_file_tool(workspace).preview(
        {"path": "app.py", "old_text": "hi", "new_text": "hello"}
    )

    assert preview.startswith("Edit file app.py")
    assert "-print('hi')" in preview
    assert "+print('hello')" in preview
    assert read_bytes(workspace, "app.py") == b"print('hi')\n"


def test_edit_file_keeps_windows_line_endings(workspace):
    (workspace.root / "win.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")

    # The model writes "\n"; the file uses "\r\n". The edit must still
    # match, and untouched lines must keep their endings.
    edit_file_tool(workspace).run(
        {"path": "win.py", "old_text": "a = 1\nb = 2", "new_text": "a = 10\nb = 20"}
    )

    assert read_bytes(workspace, "win.py") == b"a = 10\r\nb = 20\r\nc = 3\r\n"


def test_edit_file_can_delete_text(workspace):
    (workspace.root / "t.txt").write_bytes(b"keep remove keep\n")

    edit_file_tool(workspace).run({"path": "t.txt", "old_text": " remove", "new_text": ""})

    assert read_bytes(workspace, "t.txt") == b"keep keep\n"


def test_edit_file_fails_when_text_is_not_found(workspace):
    with pytest.raises(ToolError, match="not found"):
        edit_file_tool(workspace).run({"path": "app.py", "old_text": "nope", "new_text": "x"})


def test_edit_file_fails_when_text_is_ambiguous(workspace):
    (workspace.root / "dup.py").write_bytes(b"x = 1\nx = 1\n")

    with pytest.raises(ToolError, match="appears 2 times"):
        edit_file_tool(workspace).run({"path": "dup.py", "old_text": "x = 1", "new_text": "x = 2"})

    assert read_bytes(workspace, "dup.py") == b"x = 1\nx = 1\n"


def test_edit_file_on_missing_file_points_to_write_file(workspace):
    with pytest.raises(ToolError, match="write_file"):
        edit_file_tool(workspace).run({"path": "nope.py", "old_text": "a", "new_text": "b"})


def test_edit_file_refuses_secrets_file(workspace):
    with pytest.raises(ToolError, match="secrets"):
        edit_file_tool(workspace).run({"path": ".env", "old_text": "secret", "new_text": "x"})
