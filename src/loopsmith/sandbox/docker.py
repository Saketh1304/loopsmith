"""Running commands inside a Docker container.

One container is started for the whole run, with the project folder
mounted at /workspace. Each command runs in it with `docker exec`. The
agent and the API key stay outside: the container is never given the
key, so no command can read it.

Docker is driven through its command-line tool, `docker`, so loopsmith
still needs no Python packages. The function that runs `docker` is
injected, so tests can check the exact commands without Docker.
"""

import os
import subprocess
import uuid
from pathlib import Path
from typing import Callable

from loopsmith.sandbox.base import CommandResult, CommandTimedOut

DEFAULT_IMAGE = "python:3.12-slim"
CONTAINER_WORKDIR = "/workspace"

# Every loopsmith container carries this label. If loopsmith is killed
# before it can clean up, the leftovers can be found and removed with:
#   docker rm -f $(docker ps -aq --filter label=loopsmith.sandbox)
LABEL = "loopsmith.sandbox"

# Starting may include downloading the image the first time.
START_TIMEOUT_SECONDS = 300
STOP_TIMEOUT_SECONDS = 60

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
        image: str = DEFAULT_IMAGE,
        run_process: ProcessRunner = run_process,
        user: str | None = None,
    ):
        self._root = root
        self._image = image
        self._run_process = run_process
        self._user = user
        self.container: str | None = None  # the running container's name

    @property
    def where(self) -> str:
        return f"inside a Linux container (Docker image {self._image}) through sh"

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
            "--volume", f"{self._root}:{CONTAINER_WORKDIR}",
            "--workdir", CONTAINER_WORKDIR,
            "--env", "HOME=/tmp",    # a writable home folder for any user
        ]  # fmt: skip
        if self._user is not None:
            args += ["--user", self._user]
        # Keep the container alive doing nothing; commands arrive by exec.
        args += [self._image, "sleep", "infinity"]

        try:
            code, output = self._run_process(args, START_TIMEOUT_SECONDS)
        except FileNotFoundError:
            raise SandboxError(
                "Docker is not installed, or `docker` is not on the PATH. "
                "Install Docker Desktop, or run without --sandbox."
            ) from None
        except subprocess.TimeoutExpired:
            raise SandboxError(
                f"Docker did not start the container within {START_TIMEOUT_SECONDS} "
                "seconds. Check that Docker Desktop is running."
            ) from None
        if code != 0:
            raise SandboxError(
                "Docker could not start the sandbox container. Check that Docker "
                f"Desktop is running. Docker said: {output.strip()}"
            )
        self.container = name

    def stop(self) -> None:
        """Remove the container and everything still running in it.

        Safe to call twice, and never raises: it runs while cleaning up,
        often after something else has already gone wrong.
        """
        if self.container is None:
            return
        name, self.container = self.container, None
        try:
            self._run_process(["docker", "rm", "--force", name], STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass

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
        return CommandResult(code, output)
