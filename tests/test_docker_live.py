"""Tests against real Docker. Skipped unless asked for, because they need
Docker running and download an image the first time:

    $env:LOOPSMITH_DOCKER_TESTS = "1"; pytest tests/test_docker_live.py

They check what the fake in test_docker.py cannot: that the commands
loopsmith builds really work with Docker.
"""

import os
import subprocess

import pytest

from loopsmith.config import API_KEY_VAR
from loopsmith.sandbox import CommandTimedOut, DockerExecutor, current_user

pytestmark = pytest.mark.skipif(
    os.environ.get("LOOPSMITH_DOCKER_TESTS") != "1",
    reason="real Docker tests; set LOOPSMITH_DOCKER_TESTS=1 to run them",
)


def container_exists(name: str) -> bool:
    listed = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", f"name=^{name}$"],
        capture_output=True,
        encoding="utf-8",
    )
    return bool(listed.stdout.strip())


@pytest.fixture
def box(tmp_path):
    with DockerExecutor(tmp_path, user=current_user()) as executor:
        yield executor


def test_commands_run_on_linux_in_the_project_folder(box, tmp_path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")

    result = box.run("uname -s; pwd; ls", 60)

    assert result.exit_code == 0
    assert result.output.split() == ["Linux", "/workspace", "marker.txt"]


def test_the_exit_code_comes_back(box):
    assert box.run("echo failing >&2; exit 3", 60).exit_code == 3


def test_files_made_in_the_container_appear_in_the_project_folder(box, tmp_path):
    box.run("echo hello > made.txt", 60)

    assert (tmp_path / "made.txt").read_text(encoding="utf-8") == "hello\n"


def test_the_api_key_is_not_inside_the_container(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_VAR, "super-secret")

    with DockerExecutor(tmp_path, user=current_user()) as box:
        result = box.run("env", 60)

    assert "super-secret" not in result.output
    assert API_KEY_VAR not in result.output


def test_a_timeout_removes_the_container_with_the_stuck_command(box):
    old = box.container

    with pytest.raises(CommandTimedOut):
        box.run("sleep 300", 2)

    assert not container_exists(old)
    assert box.run("echo still working", 60).output == "still working\n"


def test_stopping_removes_the_container(tmp_path):
    with DockerExecutor(tmp_path, user=current_user()) as box:
        name = box.container
        assert container_exists(name)

    assert not container_exists(name)
