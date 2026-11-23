"""Tests for the Docker executor and the --sandbox flag, without Docker.

A fake stands in for the `docker` program: it records every command
loopsmith would run and answers with scripted results. The real thing is
tested in test_docker_live.py, which only runs when asked to.
"""

import os
import subprocess

import pytest

import loopsmith.cli
from loopsmith.cli import main
from loopsmith.config import API_KEY_VAR
from loopsmith.llm import ModelError, ToolCall
from loopsmith.llm.fake import FakeClient
from loopsmith.sandbox import (
    DEFAULT_IMAGE,
    CommandResult,
    CommandTimedOut,
    DockerExecutor,
    SandboxError,
    current_user,
)
from loopsmith.sandbox.docker import LABEL
from loopsmith.tools import Workspace
from loopsmith.tools.shell import run_command_tool


class FakeDocker:
    """Stands in for the `docker` program. Called with the command's words
    and a time limit, as the real ProcessRunner is."""

    def __init__(
        self,
        start_result=(0, "3f9a1c\n"),
        exec_result=(0, ""),
        exec_times_out=False,
        start_times_out=False,
        missing=False,
    ):
        self.start_result = start_result
        self.exec_result = exec_result
        self.exec_times_out = exec_times_out
        self.start_times_out = start_times_out
        self.missing = missing
        self.calls: list[tuple[list[str], int | None]] = []

    def __call__(self, args, timeout_seconds):
        self.calls.append((args, timeout_seconds))
        if self.missing:
            raise FileNotFoundError("docker")
        subcommand = args[1]
        if subcommand == "run":
            if self.start_times_out:
                raise subprocess.TimeoutExpired(args, timeout_seconds)
            return self.start_result
        if subcommand == "exec":
            if self.exec_times_out:
                raise subprocess.TimeoutExpired(args, timeout_seconds)
            return self.exec_result
        return 0, ""

    def subcommands(self) -> list[str]:
        return [args[1] for args, _ in self.calls]

    def last(self, subcommand: str) -> list[str]:
        return [args for args, _ in self.calls if args[1] == subcommand][-1]


def executor(tmp_path, docker: FakeDocker, **options) -> DockerExecutor:
    return DockerExecutor(tmp_path, run_process=docker, **options)


# --- Starting the container ---


def test_start_runs_a_long_lived_container_with_the_project_mounted(tmp_path):
    docker = FakeDocker()

    box = executor(tmp_path, docker)
    box.start()

    args = docker.last("run")
    assert args[:2] == ["docker", "run"]
    for flag in ("--detach", "--init"):
        assert flag in args
    assert args[args.index("--label") + 1] == LABEL
    assert args[args.index("--volume") + 1] == f"{tmp_path}:/workspace"
    assert args[args.index("--workdir") + 1] == "/workspace"
    # The container just waits; commands arrive later with docker exec.
    assert args[-3:] == [DEFAULT_IMAGE, "sleep", "infinity"]
    assert box.container is not None
    assert box.container.startswith("loopsmith-")
    assert args[args.index("--name") + 1] == box.container


def test_the_image_can_be_chosen(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker, image="python:3.13-slim").start()

    assert docker.last("run")[-3] == "python:3.13-slim"


def test_the_container_runs_as_the_given_user(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker, user="1000:1000").start()

    args = docker.last("run")
    assert args[args.index("--user") + 1] == "1000:1000"


def test_without_a_user_docker_picks_its_own(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker).start()

    assert "--user" not in docker.last("run")


def test_current_user_is_uid_and_gid_where_the_system_has_them(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 100, raising=False)
    assert current_user() == "1000:100"

    monkeypatch.delattr(os, "getuid")
    assert current_user() is None  # as on Windows


def test_the_api_key_is_never_given_to_the_container(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_VAR, "super-secret")
    docker = FakeDocker()

    box = executor(tmp_path, docker)
    box.start()
    box.run("env", 30)

    everything_sent = " ".join(" ".join(args) for args, _ in docker.calls)
    assert "super-secret" not in everything_sent
    assert API_KEY_VAR not in everything_sent


# --- When Docker cannot start it ---


def test_missing_docker_is_a_clear_sandbox_error(tmp_path):
    with pytest.raises(SandboxError, match="Docker is not installed"):
        executor(tmp_path, FakeDocker(missing=True)).start()


def test_a_docker_that_is_not_running_is_a_clear_sandbox_error(tmp_path):
    docker = FakeDocker(start_result=(1, "Cannot connect to the Docker daemon.\n"))

    with pytest.raises(SandboxError) as error_info:
        executor(tmp_path, docker).start()

    message = str(error_info.value)
    assert "Check that Docker Desktop is running" in message
    assert "Cannot connect to the Docker daemon." in message


def test_a_start_that_takes_too_long_is_a_sandbox_error(tmp_path):
    with pytest.raises(SandboxError, match="did not start the container"):
        executor(tmp_path, FakeDocker(start_times_out=True)).start()


# --- Running commands ---


def test_a_command_runs_in_the_container_through_sh(tmp_path):
    docker = FakeDocker(exec_result=(1, "1 failed\n"))
    box = executor(tmp_path, docker)
    box.start()

    result = box.run("pytest -q", 30)

    assert docker.calls[-1] == (
        ["docker", "exec", "--workdir", "/workspace", box.container, "sh", "-c", "pytest -q"],
        30,
    )
    assert result == CommandResult(1, "1 failed\n")


def test_the_first_command_starts_the_container_if_needed(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker).run("ls", 30)

    assert docker.subcommands() == ["run", "exec"]


def test_a_timeout_replaces_the_container_so_nothing_keeps_running(tmp_path):
    docker = FakeDocker(exec_times_out=True)
    box = executor(tmp_path, docker)
    box.start()
    old = box.container

    with pytest.raises(CommandTimedOut):
        box.run("sleep 100", 5)

    # The old container, with the stuck command in it, is removed, and a
    # fresh one is ready for the next command.
    assert docker.subcommands() == ["run", "exec", "rm", "run"]
    assert docker.last("rm") == ["docker", "rm", "--force", old]
    assert box.container is not None
    assert box.container != old


def test_run_command_works_through_the_docker_executor(tmp_path):
    docker = FakeDocker(exec_result=(0, "hello\n"))
    tool = run_command_tool(Workspace(tmp_path), executor=executor(tmp_path, docker))

    assert tool.run({"command": "echo hello"}) == "exit code: 0\nhello"


def test_where_tells_the_model_it_is_a_linux_container(tmp_path):
    where = executor(tmp_path, FakeDocker()).where

    assert "Linux container" in where
    assert DEFAULT_IMAGE in where


# --- Stopping ---


def test_stop_removes_the_container_once(tmp_path):
    docker = FakeDocker()
    box = executor(tmp_path, docker)
    box.start()
    name = box.container

    box.stop()
    box.stop()

    assert docker.subcommands() == ["run", "rm"]
    assert docker.last("rm") == ["docker", "rm", "--force", name]
    assert box.container is None


def test_stop_never_raises(tmp_path):
    def docker(args, timeout_seconds):
        if args[1] == "rm":
            raise OSError("Docker went away")
        return 0, ""

    box = executor(tmp_path, docker)
    box.start()

    box.stop()  # no exception

    assert box.container is None


def test_a_with_block_starts_and_always_stops(tmp_path):
    docker = FakeDocker()

    with pytest.raises(RuntimeError):
        with executor(tmp_path, docker) as box:
            assert box.container is not None
            raise RuntimeError("something failed during the run")

    assert docker.subcommands() == ["run", "rm"]


# --- The --sandbox flag ---


class FakeSandbox:
    """Replaces DockerExecutor inside the CLI for these tests."""

    where = "inside a fake container"
    made: list["FakeSandbox"] = []
    fail_to_start = False

    def __init__(self, root, image, user=None):
        self.image = image
        self.started = False
        self.stopped = False
        self.commands: list[str] = []
        FakeSandbox.made.append(self)

    def start(self):
        if FakeSandbox.fail_to_start:
            raise SandboxError("Docker could not start the sandbox container.")
        self.started = True

    def stop(self):
        self.stopped = True

    def run(self, command, timeout_seconds):
        self.commands.append(command)
        return CommandResult(0, "ran in the box")


@pytest.fixture
def fake_sandbox(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loopsmith.cli, "DockerExecutor", FakeSandbox)
    monkeypatch.setattr(FakeSandbox, "made", [])
    monkeypatch.setattr(FakeSandbox, "fail_to_start", False)
    return FakeSandbox


def test_sandbox_flag_runs_commands_in_the_container(fake_sandbox, capsys):
    client = FakeClient([ToolCall("run_command", {"command": "pytest"}), "Done."])

    exit_code = main(["run the tests", "--sandbox", "--yes"], client=client)

    assert exit_code == 0
    [box] = fake_sandbox.made
    assert box.image == DEFAULT_IMAGE
    assert box.commands == ["pytest"]
    assert (box.started, box.stopped) == (True, True)
    # The model is told where its commands run, and saw the box's output.
    history, system, _ = client.calls[-1]
    assert "Commands run inside a fake container." in system
    assert history[-1].tool_results[0].content == "exit code: 0\nran in the box"
    assert "starting the sandbox container" in capsys.readouterr().err


def test_the_sandbox_image_can_be_chosen_on_the_command_line(fake_sandbox):
    main(["hi", "--sandbox", "--sandbox-image", "python:3.13-slim"], client=FakeClient(["Hi."]))

    assert fake_sandbox.made[0].image == "python:3.13-slim"


def test_the_container_is_stopped_even_when_the_run_fails(fake_sandbox):
    class FailingClient:
        def generate(self, messages, system=None, tools=None):
            raise ModelError("quota used up", status=429)

    exit_code = main(["hi", "--sandbox"], client=FailingClient())

    assert exit_code == 1
    assert fake_sandbox.made[0].stopped is True


def test_a_sandbox_that_cannot_start_stops_the_run_before_any_model_call(
    fake_sandbox, capsys
):
    fake_sandbox.fail_to_start = True
    client = FakeClient(["never used"])

    exit_code = main(["hi", "--sandbox"], client=client)

    assert exit_code == 1
    assert "Error: Docker could not start the sandbox container." in capsys.readouterr().err
    assert client.calls == []


def test_without_the_flag_no_container_is_started(fake_sandbox):
    main(["hi"], client=FakeClient(["Hi."]))

    assert fake_sandbox.made == []
