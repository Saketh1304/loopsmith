"""Running commands inside a Docker container.

One container is started for the whole run, with the project folder
mounted at /workspace. Each command runs in it with `docker exec`. The
agent and the API key stay outside: the container is never given the
key, the project's secrets files (.env) look empty inside it, and its
git history (.git) is read-only, so commands can neither read the
secrets nor destroy the history.

Docker is driven through its command-line tool, `docker`, so loopsmith
still needs no Python packages. The function that runs `docker` is
injected, so tests can check the exact commands without Docker.
"""

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from loopsmith.sandbox.base import CommandResult, CommandTimedOut
from loopsmith.secret_files import find_secret_files

CONTAINER_WORKDIR = "/workspace"

DOCKER_MISSING_MESSAGE = (
    "Docker is not installed, or `docker` is not on the PATH. "
    "Install Docker Desktop, or run without --sandbox."
)

# Every loopsmith container carries this label. If loopsmith is killed
# before it can clean up, the leftovers can be found and removed with:
#   docker rm -f $(docker ps -aq --filter label=loopsmith.sandbox)
LABEL = "loopsmith.sandbox"

# Starting may include downloading the image the first time.
START_TIMEOUT_SECONDS = 300
STOP_TIMEOUT_SECONDS = 60

# --- Limits (decision 54) ---

MEMORY_LIMIT = "2g"
MEMORY_LIMIT_TEXT = "2 GB"
MAX_CPUS = 2
# Linux counts threads against this limit, not only processes, and one
# Java build can run a few hundred. 512 still stops a fork bomb at once.
PIDS_LIMIT = 512

# Exit code 137 is 128 + 9: the process was killed with signal 9
# (SIGKILL), which is what Linux's out-of-memory killer sends.
KILLED_EXIT_CODE = 137

# --- Hardening (decision 55) ---

# With the image's files read-only, /tmp is the one writable place besides
# the project. It lives in memory and counts against the memory limit.
# `exec` lets programs in it run: packages installed there contain
# compiled code that has to be loaded. mode=1777 lets any user write.
TMP_MOUNT = "/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777"


def cpu_limit() -> int:
    """MAX_CPUS, or fewer on a smaller machine: Docker refuses to start a
    container that asks for more CPUs than there are."""
    return min(MAX_CPUS, os.cpu_count() or 1)


# (the docker command as a list of words, a time limit or None) ->
# (exit code, output). Raises subprocess.TimeoutExpired when the time runs
# out, and FileNotFoundError when the `docker` program is not installed.
ProcessRunner = Callable[[list[str], int | None], tuple[int, str]]


def run_process(args: list[str], timeout_seconds: int | None) -> tuple[int, str]:
    completed = subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout or ""


def current_user() -> str | None:
    """The user to run as inside the container, as "uid:gid".

    On Linux and macOS this makes files the container creates in the
    project belong to the user, not to root. On Windows it is None:
    Docker Desktop handles file ownership itself.
    """
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None


class SandboxError(Exception):
    """The container could not be started: Docker is missing, not running,
    or refused. The run cannot go on safely without it."""


class DockerExecutor:
    def __init__(
        self,
        root: Path,
        image: str,
        run_process: ProcessRunner = run_process,
        user: str | None = None,
        network: bool = False,
        image_description: str | None = None,
        git_dir: Path | None = None,
    ):
        self._root = root
        # The git history to show inside, read-only. It is the project's own
        # .git even when `root` is a working copy that has none (decision 68),
        # so `git status` inside compares the copy with the real history.
        self._git_dir = git_dir if git_dir is not None else root / ".git"
        # Which image to run is the caller's choice; making sure it exists
        # is too (see image.py). This class only runs containers.
        self._image = image
        # What the image holds, in words for the model, such as "Python 3.12
        # with pytest and git". None for an image loopsmith knows nothing about.
        self._image_description = image_description
        self._run_process = run_process
        self._user = user
        # Off unless asked for (decision 53). The agent's own model calls
        # happen outside the container, so it never needs the network.
        self._network = network
        self.container: str | None = None  # the running container's name
        # An empty file on this machine, mounted over each secrets file
        # while a container runs (decision 56).
        self._empty_file: Path | None = None

    @property
    def where(self) -> str:
        """Also tells the model what the box allows, so it does not waste
        steps trying to download things or write outside the project."""
        network = "network access" if self._network else "no network access"
        image = f"Docker image {self._image}"
        if self._image_description:
            image += f": {self._image_description}"
        return (
            f"inside a Linux container ({image}) through sh. "
            f"It has {network}, at most {MEMORY_LIMIT_TEXT} of memory, and only "
            "the project folder and /tmp can be written to. The .git folder is "
            "read-only, and .env files look empty"
        )

    def _mounts(self) -> list[str]:
        """The project folder, with its history and its secrets protected.

        Docker mounts the deeper paths over the project folder, so inside
        the container they replace what the folder has at those paths.
        """
        args = ["--volume", f"{self._root}:{CONTAINER_WORKDIR}"]
        git = self._git_dir
        if git.exists():
            # Commands can still read the history (git log, git diff) but not
            # change or delete it, so the user can always get files back.
            args += ["--volume", f"{git}:{CONTAINER_WORKDIR}/.git:ro"]
        secrets = find_secret_files(self._root)
        self._remove_empty_file()  # never leave an old one behind
        if secrets:
            handle, name = tempfile.mkstemp(prefix="loopsmith-hidden-")
            os.close(handle)
            # The full spelling of the path, never a Windows short form such
            # as RUNNER~1, which Docker may not understand as a mount source
            # (decision 70).
            self._empty_file = Path(name).resolve()
            for path in secrets:
                inside = path.relative_to(self._root).as_posix()
                args += ["--volume", f"{self._empty_file}:{CONTAINER_WORKDIR}/{inside}:ro"]
        return args

    def _remove_empty_file(self) -> None:
        if self._empty_file is not None:
            self._empty_file.unlink(missing_ok=True)
            self._empty_file = None

    # --- Starting and stopping ---

    def start(self) -> None:
        """Start the container. Raises SandboxError if Docker cannot."""
        name = f"loopsmith-{uuid.uuid4().hex[:12]}"
        args = [
            "docker", "run",
            "--detach",              # start it and return at once
            "--init",                # a tiny init process that cleans up finished processes
            "--name", name,
            "--label", LABEL,
            *self._mounts(),
            "--workdir", CONTAINER_WORKDIR,
            "--env", "HOME=/tmp",    # a writable home folder for any user
            # Limits: memory (the same figure for memory-swap means no swap
            # on top), CPUs, and processes.
            "--memory", MEMORY_LIMIT,
            "--memory-swap", MEMORY_LIMIT,
            "--cpus", str(cpu_limit()),
            "--pids-limit", str(PIDS_LIMIT),
            # Hardening: none of root's special powers, no way to gain them
            # back, and the image's own files cannot be changed.
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", TMP_MOUNT,
        ]  # fmt: skip
        if not self._network:
            args += ["--network", "none"]  # only the container's own loopback
        if self._user is not None:
            args += ["--user", self._user]
        # Keep the container alive doing nothing; commands arrive by exec.
        args += [self._image, "sleep", "infinity"]

        try:
            code, output = self._run_process(args, START_TIMEOUT_SECONDS)
        except FileNotFoundError:
            error = SandboxError(DOCKER_MISSING_MESSAGE)
        except subprocess.TimeoutExpired:
            error = SandboxError(
                f"Docker did not start the container within {START_TIMEOUT_SECONDS} "
                "seconds. Check that Docker Desktop is running."
            )
        else:
            if code == 0:
                self.container = name
                return
            error = SandboxError(
                "Docker could not start the sandbox container. Check that Docker "
                f"Desktop is running. Docker said: {output.strip()}"
            )
        self._remove_empty_file()  # no container will use it
        raise error

    def stop(self) -> None:
        """Remove the container and everything still running in it.

        Safe to call twice, and never raises: it runs while cleaning up,
        often after something else has already gone wrong.
        """
        if self.container is not None:
            name, self.container = self.container, None
            try:
                self._run_process(["docker", "rm", "--force", name], STOP_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._remove_empty_file()

    # `with DockerExecutor(...) as executor:` starts it and always stops it,
    # like try-with-resources on a Java AutoCloseable.
    def __enter__(self) -> "DockerExecutor":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # --- Running commands ---

    def run(self, command: str, timeout_seconds: int) -> CommandResult:
        if self.container is None:
            self.start()
        args = ["docker", "exec", "--workdir", CONTAINER_WORKDIR, self.container]
        args += ["sh", "-c", command]
        try:
            code, output = self._run_process(args, timeout_seconds)
        except subprocess.TimeoutExpired:
            # Stopping `docker exec` here does not stop the command inside
            # the container: it would keep running. Replacing the container
            # does, along with anything the command started. The project's
            # files are safe, because they live in the mounted folder.
            self.stop()
            self.start()
            raise CommandTimedOut() from None
        if code == KILLED_EXIT_CODE:
            # Without this the model sees only "exit code: 137" and may run
            # the same command again. It is the likely cause, not a certain
            # one: a command can also be killed with signal 9 some other way.
            output += (
                f"\n[The command was killed (exit code {code}), most likely for using "
                f"more than the sandbox's {MEMORY_LIMIT_TEXT} of memory.]"
            )
        return CommandResult(code, output)
