"""Tests for the Docker executor and the --sandbox flag, without Docker.

A fake stands in for the `docker` program: it records every command
loopsmith would run and answers with scripted results. The real thing is
tested in test_docker_live.py, which only runs when asked to.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import loopsmith.cli
from loopsmith.cli import main, system_prompt
from loopsmith.config import API_KEY_VAR
from loopsmith.llm import ModelError, ToolCall
from loopsmith.llm.fake import FakeClient
from loopsmith.sandbox import (
    IMAGE_DESCRIPTION,
    CommandResult,
    CommandTimedOut,
    DockerExecutor,
    SandboxError,
    current_user,
)
from loopsmith.sandbox.docker import (
    LABEL,
    MEMORY_LIMIT,
    PIDS_LIMIT,
    TMP_MOUNT,
    cpu_limit,
)
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


# Any image name will do for these tests: the fake runs nothing.
STAND_IN_IMAGE = "python:3.12-slim"


def executor(tmp_path, docker: FakeDocker, **options) -> DockerExecutor:
    options.setdefault("image", STAND_IN_IMAGE)
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
    assert args[-3:] == [STAND_IN_IMAGE, "sleep", "infinity"]
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


# --- Network, limits and hardening ---


def option(args: list[str], flag: str) -> str:
    """The value given after a flag, as in `--memory 2g`."""
    return args[args.index(flag) + 1]


def test_the_container_has_no_network_by_default(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker).start()

    assert option(docker.last("run"), "--network") == "none"


def test_the_network_can_be_allowed(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker, network=True).start()

    assert "--network" not in docker.last("run")


def test_memory_cpu_and_process_limits_are_set(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker).start()

    args = docker.last("run")
    assert option(args, "--memory") == MEMORY_LIMIT == "2g"
    # The same figure for memory-swap means no swap on top of the limit.
    assert option(args, "--memory-swap") == MEMORY_LIMIT
    assert option(args, "--cpus") == str(cpu_limit())
    assert option(args, "--pids-limit") == str(PIDS_LIMIT) == "512"


def test_the_cpu_limit_never_asks_for_more_cpus_than_the_machine_has(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert cpu_limit() == 2

    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert cpu_limit() == 1

    # cpu_count() can return None when Python cannot tell.
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert cpu_limit() == 1


def test_the_container_has_no_special_powers_and_read_only_system_files(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker).start()

    args = docker.last("run")
    assert option(args, "--cap-drop") == "ALL"
    assert option(args, "--security-opt") == "no-new-privileges"
    assert "--read-only" in args
    # /tmp stays writable, limited in size, and programs in it can run.
    assert option(args, "--tmpfs") == TMP_MOUNT
    assert TMP_MOUNT.startswith("/tmp:rw,exec,")
    assert "size=512m" in TMP_MOUNT


def test_where_tells_the_model_there_is_no_network_and_what_is_writable(tmp_path):
    where = executor(tmp_path, FakeDocker()).where

    assert "no network access" in where
    assert "at most 2 GB of memory" in where
    assert "only the project folder and /tmp can be written to" in where


def test_where_says_so_when_the_network_is_allowed(tmp_path):
    where = executor(tmp_path, FakeDocker(), network=True).where

    assert "It has network access" in where


def test_the_system_prompt_carries_the_sandbox_rules(tmp_path):
    prompt = system_prompt(executor(tmp_path, FakeDocker()))

    assert "Commands run inside a Linux container" in prompt
    assert "no network access" in prompt


def test_a_command_killed_with_exit_code_137_is_explained(tmp_path):
    docker = FakeDocker(exec_result=(137, "building...\n"))

    result = executor(tmp_path, docker).run("make", 30)

    assert result.exit_code == 137
    assert result.output == (
        "building...\n\n[The command was killed (exit code 137), most likely for "
        "using more than the sandbox's 2 GB of memory.]"
    )


def test_the_explanation_survives_when_long_output_is_shortened(tmp_path):
    docker = FakeDocker(exec_result=(137, "x" * 50_000))
    tool = run_command_tool(Workspace(tmp_path), executor=executor(tmp_path, docker))

    result = tool.run({"command": "make"})

    assert result.startswith("exit code: 137\n")
    assert result.endswith("of memory.]")


# --- The project's history and secrets ---


def mounts(args: list[str]) -> list[str]:
    """Every `--volume` value, as in `C:\\proj:/workspace`."""
    return [args[i + 1] for i, word in enumerate(args) if word == "--volume"]


def hidden_mounts(args: list[str]) -> list[tuple[Path, str]]:
    """(file on this machine, path in the container) for each hidden secret.
    Split from the right, because a Windows path has a colon of its own."""
    found = []
    for mount in mounts(args):
        source, target, mode = mount.rsplit(":", 2) if mount.endswith(":ro") else (None, None, None)
        if target is not None and not target.endswith("/.git"):
            found.append((Path(source), target))
    return found


def make_secrets(root: Path) -> None:
    (root / ".env").write_text("GEMINI_API_KEY=real-looking\n", encoding="utf-8")
    (root / "backend").mkdir()
    (root / "backend" / ".env.local").write_text("DB_PASSWORD=x\n", encoding="utf-8")
    (root / ".env.example").write_text("GEMINI_API_KEY=\n", encoding="utf-8")


def test_the_git_history_is_mounted_read_only(tmp_path):
    (tmp_path / ".git").mkdir()
    docker = FakeDocker()

    executor(tmp_path, docker).start()

    assert f"{tmp_path / '.git'}:/workspace/.git:ro" in mounts(docker.last("run"))


def test_the_history_can_come_from_another_folder(tmp_path):
    # A working copy has no .git of its own; the project's is shown instead.
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    copy_root = tmp_path / "copy"
    copy_root.mkdir()
    docker = FakeDocker()

    executor(copy_root, docker, git_dir=project / ".git").start()

    assert mounts(docker.last("run")) == [
        f"{copy_root}:/workspace",
        f"{project / '.git'}:/workspace/.git:ro",
    ]


def test_a_project_without_secrets_or_git_mounts_only_the_folder(tmp_path):
    docker = FakeDocker()

    executor(tmp_path, docker).start()

    assert mounts(docker.last("run")) == [f"{tmp_path}:/workspace"]


def test_secret_files_are_covered_by_an_empty_read_only_file(tmp_path):
    make_secrets(tmp_path)
    docker = FakeDocker()
    box = executor(tmp_path, docker)

    box.start()

    hidden = hidden_mounts(docker.last("run"))
    # Every secrets file, but not the key-free template.
    assert [target for _, target in hidden] == ["/workspace/.env", "/workspace/backend/.env.local"]
    [empty] = {source for source, _ in hidden}
    assert empty.exists()
    assert empty.stat().st_size == 0
    # The real files on this machine are untouched.
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "GEMINI_API_KEY=real-looking\n"

    box.stop()

    assert not empty.exists()


def test_the_hidden_file_path_has_one_spelling(tmp_path, monkeypatch):
    # A Windows short name such as RUNNER~1 may not work as a Docker mount
    # source. "x/.." stands in for such a second spelling of one folder.
    real_mkstemp = tempfile.mkstemp
    temp = tmp_path / "temp"
    (temp / "x").mkdir(parents=True)

    def roundabout_mkstemp(prefix=""):
        handle, name = real_mkstemp(prefix=prefix, dir=temp)
        return handle, str(temp / "x" / ".." / Path(name).name)

    monkeypatch.setattr(tempfile, "mkstemp", roundabout_mkstemp)
    project = tmp_path / "project"
    project.mkdir()
    make_secrets(project)
    docker = FakeDocker()
    box = executor(project, docker)

    box.start()

    [(source, _), *_] = hidden_mounts(docker.last("run"))
    assert ".." not in source.parts
    assert source == source.resolve()
    box.stop()


def test_a_failed_start_leaves_no_empty_file_behind(tmp_path):
    make_secrets(tmp_path)
    docker = FakeDocker(start_result=(1, "Cannot connect to the Docker daemon.\n"))

    with pytest.raises(SandboxError):
        executor(tmp_path, docker).start()

    [(empty, _), *_] = hidden_mounts(docker.last("run"))
    assert not empty.exists()


def test_a_replaced_container_hides_the_secrets_too(tmp_path):
    make_secrets(tmp_path)
    docker = FakeDocker(exec_times_out=True)
    box = executor(tmp_path, docker)
    box.start()
    first = docker.last("run")

    with pytest.raises(CommandTimedOut):
        box.run("sleep 100", 5)

    second = docker.last("run")
    assert second is not first
    assert [t for _, t in hidden_mounts(second)] == [t for _, t in hidden_mounts(first)]
    assert not hidden_mounts(first)[0][0].exists()  # the old empty file is gone
    assert hidden_mounts(second)[0][0].exists()
    box.stop()


def test_where_tells_the_model_about_git_and_env_files(tmp_path):
    (tmp_path / ".git").mkdir()

    where = executor(tmp_path, FakeDocker()).where

    assert "The .git folder is read-only, and .env files look empty" in where


def test_where_says_there_is_no_git_history_when_none_is_mounted(tmp_path):
    # As in a subfolder of a repository (examples/buggy_stats): the history
    # is further up, outside what the container sees. A live run spent a
    # step on `git diff` because the prompt said there was a .git folder.
    docker = FakeDocker()
    box = executor(tmp_path, docker)

    box.start()

    assert not any(m.endswith("/.git:ro") for m in mounts(docker.last("run")))
    assert "There is no git history, so git commands such as git diff fail" in box.where
    assert "The .git folder" not in box.where
    assert ".env files look empty" in box.where
    box.stop()


def test_where_follows_the_history_shown_not_the_folder(tmp_path):
    # A working copy has no .git of its own, but the project's is mounted.
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    copy_root = tmp_path / "copy"
    copy_root.mkdir()

    where = executor(copy_root, FakeDocker(), git_dir=project / ".git").where

    assert "The .git folder is read-only" in where
    assert "no git history" not in where


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
    assert STAND_IN_IMAGE in where


def test_where_says_what_is_in_the_image_when_told(tmp_path):
    box = executor(
        tmp_path, FakeDocker(), image="loopsmith-sandbox:abc", image_description=IMAGE_DESCRIPTION
    )

    assert "(Docker image loopsmith-sandbox:abc: Python 3.12 with pytest and git)" in box.where


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

    def __init__(
        self, root, image, user=None, network=False, image_description=None, git_dir=None
    ):
        self.root = root
        self.git_dir = git_dir
        self.image = image
        self.image_description = image_description
        self.network = network
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


LOOPSMITH_IMAGE = "loopsmith-sandbox:fake"


class FakeImageBuilder:
    """Replaces ensure_image inside the CLI: no Docker, no build."""

    def __init__(self):
        self.calls = 0
        self.needs_build = False  # when True, it reports a first-time build
        self.error: SandboxError | None = None

    def __call__(self, on_build=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.needs_build and on_build is not None:
            on_build(LOOPSMITH_IMAGE)
        return LOOPSMITH_IMAGE


@pytest.fixture
def fake_sandbox(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loopsmith.cli, "DockerExecutor", FakeSandbox)
    monkeypatch.setattr(FakeSandbox, "made", [])
    monkeypatch.setattr(FakeSandbox, "fail_to_start", False)
    FakeSandbox.images = FakeImageBuilder()
    monkeypatch.setattr(loopsmith.cli, "ensure_image", FakeSandbox.images)
    return FakeSandbox


def test_sandbox_flag_runs_commands_in_the_container(fake_sandbox, capsys):
    client = FakeClient([ToolCall("run_command", {"command": "pytest"}), "Done."])

    exit_code = main(["run the tests", "--sandbox", "--yes"], client=client)

    assert exit_code == 0
    [box] = fake_sandbox.made
    # loopsmith's own image, and the model is told what is in it.
    assert box.image == LOOPSMITH_IMAGE
    assert box.image_description == IMAGE_DESCRIPTION
    assert box.network is False
    assert box.commands == ["pytest"]
    assert (box.started, box.stopped) == (True, True)
    # The model is told where its commands run, and saw the box's output.
    history, system, _ = client.calls[-1]
    assert "Commands run inside a fake container." in system
    assert history[-1].tool_results[0].content == "exit code: 0\nran in the box"
    assert (
        f"starting the sandbox container ({LOOPSMITH_IMAGE}, no network)"
        in capsys.readouterr().err
    )


def test_the_first_run_builds_the_image_and_says_so(fake_sandbox, capsys):
    fake_sandbox.images.needs_build = True

    main(["hi", "--sandbox"], client=FakeClient(["Hi."]))

    err = capsys.readouterr().err
    assert f"building the sandbox image {LOOPSMITH_IMAGE}: first time only" in err
    # Built before the container starts, which is what uses it.
    assert err.index("building the sandbox image") < err.index("starting the sandbox container")


def test_an_image_that_cannot_be_built_stops_the_run_before_any_model_call(
    fake_sandbox, capsys
):
    fake_sandbox.images.error = SandboxError("Docker could not build the sandbox image.")
    client = FakeClient(["never used"])

    exit_code = main(["hi", "--sandbox"], client=client)

    assert exit_code == 1
    assert "Error: Docker could not build the sandbox image." in capsys.readouterr().err
    assert fake_sandbox.made == []
    assert client.calls == []


def test_the_sandbox_network_flag_reaches_the_container(fake_sandbox, capsys):
    main(["hi", "--sandbox", "--sandbox-network"], client=FakeClient(["Hi."]))

    assert fake_sandbox.made[0].network is True
    assert "with network)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags", [["--sandbox-network"], ["--sandbox-image", "python:3.13-slim"]]
)
def test_sandbox_options_without_the_sandbox_are_refused(fake_sandbox, capsys, flags):
    # Ignoring them would let someone believe commands are sandboxed.
    client = FakeClient(["never used"])

    with pytest.raises(SystemExit) as exit_info:
        main(["hi", *flags], client=client)

    assert exit_info.value.code == 2
    assert "only work with --sandbox" in capsys.readouterr().err
    assert fake_sandbox.made == []
    assert client.calls == []


def test_the_sandbox_image_can_be_chosen_on_the_command_line(fake_sandbox):
    main(["hi", "--sandbox", "--sandbox-image", "python:3.13-slim"], client=FakeClient(["Hi."]))

    box = fake_sandbox.made[0]
    assert box.image == "python:3.13-slim"
    # Someone else's image: nothing to build, and nothing to tell the model
    # about what is inside it.
    assert fake_sandbox.images.calls == 0
    assert box.image_description is None


# --- The agent works on a copy (decision 68) ---


class ReviewAnswer:
    """Stands in for the question at the end: records the review shown and
    gives a scripted answer."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.reviews: list[str] = []

    def __call__(self, review: str) -> bool:
        self.reviews.append(review)
        return self.answer


EDIT_STATS = ToolCall("write_file", {"path": "stats.py", "content": "fixed\n"})


def run_sandboxed(tmp_path, answer: bool, client=None):
    (tmp_path / "stats.py").write_bytes(b"buggy\n")
    confirm = ReviewAnswer(answer)
    exit_code = main(
        ["fix the bug", "--sandbox", "--yes"],
        client=client or FakeClient([EDIT_STATS, "Done."]),
        confirm_apply=confirm,
    )
    return exit_code, confirm


def test_the_agent_and_the_container_work_on_a_copy(fake_sandbox, tmp_path, capsys):
    exit_code, confirm = run_sandboxed(tmp_path, answer=False)

    assert exit_code == 0
    [box] = fake_sandbox.made
    # The container got the copy, and the project's own history.
    assert box.root != tmp_path
    assert box.git_dir == tmp_path / ".git"
    # The agent's edit is in the copy; the project is untouched.
    assert (tmp_path / "stats.py").read_bytes() == b"buggy\n"
    assert (box.root / "stats.py").read_bytes() == b"fixed\n"
    err = capsys.readouterr().err
    assert "working on a copy of the project (1 files)" in err
    assert f"not applied. The changes are kept in {box.root}" in err
    shutil.rmtree(box.root)


def test_the_review_shows_the_diff_before_asking(fake_sandbox, tmp_path):
    _, confirm = run_sandboxed(tmp_path, answer=False)

    [review] = confirm.reviews
    assert review.startswith("Changes made in the copy (1 changed):")
    assert "-buggy" in review
    assert "+fixed" in review
    shutil.rmtree(fake_sandbox.made[0].root)


def test_applying_at_the_end_changes_the_project(fake_sandbox, tmp_path, capsys):
    exit_code, _ = run_sandboxed(tmp_path, answer=True)

    assert exit_code == 0
    assert (tmp_path / "stats.py").read_bytes() == b"fixed\n"
    assert "applied 1 of 1 changes" in capsys.readouterr().err
    assert not fake_sandbox.made[0].root.exists()  # the copy is cleaned up


def test_no_changes_means_no_question(fake_sandbox, tmp_path, capsys):
    confirm = ReviewAnswer(True)

    main(["hi", "--sandbox"], client=FakeClient(["Hi."]), confirm_apply=confirm)

    assert confirm.reviews == []
    assert "no changes to apply" in capsys.readouterr().err
    assert not fake_sandbox.made[0].root.exists()


def test_a_run_that_stops_early_applies_nothing_and_keeps_the_changes(
    fake_sandbox, tmp_path, capsys
):
    class FailsAfterTheEdit(FakeClient):
        def generate(self, messages, system=None, tools=None):
            if self.calls:
                raise ModelError("quota used up", status=429)
            return super().generate(messages, system=system, tools=tools)

    exit_code, confirm = run_sandboxed(
        tmp_path, answer=True, client=FailsAfterTheEdit([EDIT_STATS])
    )

    assert exit_code == 1
    assert confirm.reviews == []  # nobody is asked about a half-finished run
    assert (tmp_path / "stats.py").read_bytes() == b"buggy\n"
    copy_root = fake_sandbox.made[0].root
    assert (copy_root / "stats.py").read_bytes() == b"fixed\n"  # nothing is lost
    err = capsys.readouterr().err
    assert f"the run stopped early, so nothing was applied. The changes so far are in {copy_root}" in err
    assert "Error: quota used up" in err
    shutil.rmtree(copy_root)


def test_without_the_sandbox_there_is_no_copy(fake_sandbox, tmp_path, capsys):
    (tmp_path / "stats.py").write_bytes(b"buggy\n")

    main(["fix the bug", "--yes"], client=FakeClient([EDIT_STATS, "Done."]))

    assert (tmp_path / "stats.py").read_bytes() == b"fixed\n"
    assert "working on a copy" not in capsys.readouterr().err


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
