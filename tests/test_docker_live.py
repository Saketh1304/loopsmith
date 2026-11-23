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
from loopsmith.sandbox.docker import cpu_limit

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


# --- The project's history and secrets ---


def test_secret_files_look_empty_and_cannot_be_changed(tmp_path):
    (tmp_path / ".env").write_text("FAKE_SECRET=not-a-real-key\n", encoding="utf-8")

    with DockerExecutor(tmp_path, user=current_user()) as box:
        read = box.run("cat .env", 60)
        write = box.run("echo stolen > .env", 60)

    assert (read.exit_code, read.output) == (0, "")
    assert write.exit_code != 0
    assert "Read-only file system" in write.output
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "FAKE_SECRET=not-a-real-key\n"


def test_the_git_history_cannot_be_changed(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    with DockerExecutor(tmp_path, user=current_user()) as box:
        removed = box.run("rm -rf .git", 60)
        read = box.run("cat .git/HEAD", 60)

    assert removed.exit_code != 0
    assert read.output == "ref: refs/heads/main\n"  # still there, still readable
    assert (tmp_path / ".git" / "HEAD").exists()


# --- Network, limits and hardening, checked from inside the container ---


def test_the_container_has_no_network(box):
    # Only the loopback interface exists, so there is no way out. Nothing
    # is sent anywhere to find this out.
    assert box.run("ls /sys/class/net", 60).output.split() == ["lo"]


def test_the_network_can_be_allowed(tmp_path):
    with DockerExecutor(tmp_path, user=current_user(), network=True) as box:
        interfaces = box.run("ls /sys/class/net", 60).output.split()

    assert "lo" in interfaces
    assert len(interfaces) > 1  # a real network interface as well


def test_only_the_project_and_tmp_can_be_written(box, tmp_path):
    blocked = box.run("touch /etc/planted", 60)
    allowed = box.run("touch /tmp/scratch /workspace/made && echo ok", 60)

    assert blocked.exit_code != 0
    assert "Read-only file system" in blocked.output
    assert allowed.output == "ok\n"
    assert (tmp_path / "made").exists()


def test_programs_in_tmp_can_run(box):
    script = "printf '#!/bin/sh\\necho ran\\n' > /tmp/s.sh && chmod +x /tmp/s.sh && /tmp/s.sh"

    assert box.run(script, 60).output == "ran\n"


def test_the_container_has_no_special_powers(box):
    # grep reads its own status: no capabilities at all, and no way to
    # gain new privileges.
    status = box.run("grep -E '^(CapEff|NoNewPrivs):' /proc/self/status", 60)

    assert status.output.split() == ["CapEff:", "0000000000000000", "NoNewPrivs:", "1"]


def read_limit(box, *paths: str) -> str:
    """The contents of the first of these files that exists.

    The kernel keeps a container's limits in its cgroup files. Newer
    systems use the cgroup v2 layout; others, including some Docker
    Desktop setups, use v1, where the same limit has another file name.
    """
    for path in paths:
        result = box.run(f"cat {path}", 60)
        if result.exit_code == 0:
            return result.output.strip()
    pytest.fail(f"none of these cgroup files exist: {paths}")


def test_the_limits_are_in_force(box):
    cgroup = "/sys/fs/cgroup"

    memory = read_limit(box, f"{cgroup}/memory.max", f"{cgroup}/memory/memory.limit_in_bytes")
    processes = read_limit(box, f"{cgroup}/pids.max", f"{cgroup}/pids/pids.max")
    cpu = read_limit(box, f"{cgroup}/cpu.max", f"{cgroup}/cpu/cpu.cfs_quota_us")

    assert memory == str(2 * 1024**3)
    assert processes == "512"
    # CPU time allowed per 100,000 microseconds: 200,000 means two CPUs.
    assert cpu.split()[0] == str(cpu_limit() * 100_000)


def test_running_out_of_memory_is_explained_and_the_box_survives(box):
    # Filling 2.5 GB goes past the 2 GB limit, so Linux kills the command.
    result = box.run("python -c \"b = b'x' * (2560 * 1024 ** 2)\"", 120)

    assert result.exit_code == 137
    assert "most likely for using more than the sandbox's 2 GB of memory" in result.output
    # Only the command was killed; the container still works.
    assert box.run("echo alive", 60).output == "alive\n"
