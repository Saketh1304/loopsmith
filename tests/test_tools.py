import pytest

from loopsmith.tools import ToolError, Workspace
from loopsmith.tools.files import (
    MAX_FILE_CHARS,
    default_tools,
    list_files_tool,
    read_file_tool,
)


@pytest.fixture
def workspace(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
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

    assert set(specs) == {"list_files", "read_file"}
    assert specs["read_file"].parameters["required"] == ["path"]
