"""Tests against real Docker. Skipped unless asked for, because they need
Docker running, and the first run builds loopsmith's image, which needs
the network:

    $env:LOOPSMITH_DOCKER_TESTS = "1"; pytest tests/test_docker_live.py

They check what the fake in test_docker.py cannot: that the commands
loopsmith builds really work with Docker.
"""

import os
import shutil
import subprocess

import pytest

from loopsmith.cli import main
from loopsmith.config import API_KEY_VAR
from loopsmith.llm import ToolCall
from loopsmith.llm.fake import FakeClient
from loopsmith.sandbox import (
    IMAGE_DESCRIPTION,
    CommandTimedOut,
    DockerExecutor,
    WorkingCopy,
    current_user,
    ensure_image,
)
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


@pytest.fixture(scope="session")
def sandbox_image():
    """loopsmith's own image, built the first time (scope="session": once
    for the whole test run, then shared by every test)."""
    return ensure_image()


def open_box(root, image, **options) -> DockerExecutor:
    """A sandbox on loopsmith's image, as `loopsmith --sandbox` makes one."""
    return DockerExecutor(
        root, image=image, user=current_user(), image_description=IMAGE_DESCRIPTION, **options
    )


@pytest.fixture
def box(tmp_path, sandbox_image):
    with open_box(tmp_path, sandbox_image) as executor:
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


def test_the_api_key_is_not_inside_the_container(tmp_path, monkeypatch, sandbox_image):
    monkeypatch.setenv(API_KEY_VAR, "super-secret")

    with open_box(tmp_path, sandbox_image) as box:
        result = box.run("env", 60)

    assert "super-secret" not in result.output
    assert API_KEY_VAR not in result.output


def test_a_timeout_removes_the_container_with_the_stuck_command(box):
    old = box.container

    with pytest.raises(CommandTimedOut):
        box.run("sleep 300", 2)

    assert not container_exists(old)
    assert box.run("echo still working", 60).output == "still working\n"


def test_stopping_removes_the_container(tmp_path, sandbox_image):
    with open_box(tmp_path, sandbox_image) as box:
        name = box.container
        assert container_exists(name)

    assert not container_exists(name)


# --- loopsmith's own image (decision 66) ---


def test_the_image_has_python_pytest_and_git(box):
    result = box.run("python --version && pytest --version && git --version", 60)

    assert result.exit_code == 0
    assert "Python 3.12." in result.output
    assert "pytest 9.2.0" in result.output
    assert "git version" in result.output


def test_a_projects_tests_run_with_the_network_off(box, tmp_path):
    # The point of the image: "run the tests" works inside the sandbox.
    (tmp_path / "test_sample.py").write_bytes(
        b"def test_passes():\n    assert 1 + 1 == 2\n\n\n"
        b"def test_fails():\n    assert 1 + 1 == 3\n"
    )

    result = box.run("pytest -q -p no:cacheprovider", 120)

    assert result.exit_code == 1  # pytest's code for "some tests failed"
    assert "1 failed, 1 passed" in result.output


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git on this machine")
def test_git_reads_the_history_but_cannot_change_it(tmp_path, sandbox_image):
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), *identity, "commit", "-q", "--allow-empty", "-m", "first"],
        check=True,
    )

    with open_box(tmp_path, sandbox_image) as box:
        log = box.run("git log --format=%s", 60)
        status = box.run("git status --short", 60)
        commit = box.run(
            "git -c user.name=Agent -c user.email=agent@example.com "
            "commit --allow-empty -m second",
            60,
        )

    assert (log.exit_code, log.output) == (0, "first\n")
    assert status.exit_code == 0
    assert commit.exit_code != 0
    assert "Read-only file system" in commit.output


# --- Working on a copy (decision 68) ---


def test_a_command_can_only_damage_the_copy(tmp_path, sandbox_image):
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    (project / "untracked.txt").write_bytes(b"not in git, nowhere else\n")
    (project / "sub" / "code.py").write_bytes(b"x = 1\n")
    copy = WorkingCopy.create(project)
    try:
        with open_box(copy.root, sandbox_image, git_dir=project / ".git") as box:
            box.run("rm -rf /workspace/* && echo wiped", 60)

        # The project is untouched, untracked file included...
        assert (project / "untracked.txt").read_bytes() == b"not in git, nowhere else\n"
        assert (project / "sub" / "code.py").read_bytes() == b"x = 1\n"
        # ...and the damage shows up as changes to review, not applied ones.
        assert copy.changes().deleted == ("sub/code.py", "untracked.txt")
    finally:
        copy.remove()


def test_the_whole_flow_through_the_cli(tmp_path, monkeypatch, sandbox_image):
    # A scripted model asks for one command; the real container runs it on
    # the copy; the fix reaches the project only once the review is approved.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stats.py").write_bytes(b"buggy\n")
    fix = ToolCall("run_command", {"command": "printf 'fixed\\n' > stats.py"})
    reviews = []

    def approve_review(review: str) -> bool:
        reviews.append(review)
        assert (tmp_path / "stats.py").read_bytes() == b"buggy\n"  # not yet applied
        return True

    exit_code = main(
        ["fix stats.py", "--sandbox", "--yes"],
        client=FakeClient([fix, "Fixed."]),
        confirm_apply=approve_review,
    )

    assert exit_code == 0
    assert "+fixed" in reviews[0]
    assert (tmp_path / "stats.py").read_bytes() == b"fixed\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git on this machine")
def test_git_in_the_box_compares_the_copy_with_the_real_history(tmp_path, sandbox_image):
    project = tmp_path / "project"
    project.mkdir()
    (project / "x.txt").write_bytes(b"one\n")
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(project), "add", "x.txt"], check=True)
    subprocess.run(["git", "-C", str(project), *identity, "commit", "-q", "-m", "first"], check=True)
    copy = WorkingCopy.create(project)
    try:
        with open_box(copy.root, sandbox_image, git_dir=project / ".git") as box:
            status = box.run("echo two >> x.txt && git status --short", 60)

        assert status.exit_code == 0
        assert status.output.split() == ["M", "x.txt"]  # the copy, against the real history
        assert (project / "x.txt").read_bytes() == b"one\n"  # the project is unchanged
        assert copy.changes().modified == ("x.txt",)
    finally:
        copy.remove()


# --- The project's history and secrets ---


def test_secret_files_look_empty_and_cannot_be_changed(tmp_path, sandbox_image):
    (tmp_path / ".env").write_text("FAKE_SECRET=not-a-real-key\n", encoding="utf-8")

    with open_box(tmp_path, sandbox_image) as box:
        read = box.run("cat .env", 60)
        write = box.run("echo stolen > .env", 60)

    assert (read.exit_code, read.output) == (0, "")
    assert write.exit_code != 0
    assert "Read-only file system" in write.output
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "FAKE_SECRET=not-a-real-key\n"


def test_the_git_history_cannot_be_changed(tmp_path, sandbox_image):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    with open_box(tmp_path, sandbox_image) as box:
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


def test_the_network_can_be_allowed(tmp_path, sandbox_image):
    with open_box(tmp_path, sandbox_image, network=True) as box:
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
