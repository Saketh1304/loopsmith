"""Tools: the actions the agent can take."""

from loopsmith.tools.base import Tool, ToolError
from loopsmith.tools.files import (
    Workspace,
    edit_file_tool,
    list_files_tool,
    read_file_tool,
    write_file_tool,
)
from loopsmith.tools.shell import run_command_tool

__all__ = ["Tool", "ToolError", "Workspace", "default_tools", "research_tools"]


def default_tools(workspace: Workspace) -> list[Tool]:
    return [
        list_files_tool(workspace),
        read_file_tool(workspace),
        edit_file_tool(workspace),
        write_file_tool(workspace),
        run_command_tool(workspace),
    ]


def research_tools(workspace: Workspace) -> list[Tool]:
    """The tools that only look and never change anything."""
    return [list_files_tool(workspace), read_file_tool(workspace)]
