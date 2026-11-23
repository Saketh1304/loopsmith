"""loopsmith's own sandbox image, built from a Dockerfile that ships inside
the package, the first time it is needed (decisions 66 and 67).

`docker run` downloads a missing image from Docker Hub by itself, but this
image exists nowhere online: it has to be built on this machine. Building
needs the network once; every sandbox run after that starts offline.
"""

import hashlib
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from typing import Callable

from loopsmith.sandbox.docker import (
    DOCKER_MISSING_MESSAGE,
    LABEL,
    ProcessRunner,
    SandboxError,
    run_process,
)

DOCKERFILE = "python.Dockerfile"
IMAGE_NAME = "loopsmith-sandbox"

# Told to the model in the system prompt, so it knows what it can run.
IMAGE_DESCRIPTION = "Python 3.12 with pytest and git"

CHECK_TIMEOUT_SECONDS = 60
# The first build downloads the base image and the packages.
BUILD_TIMEOUT_SECONDS = 600

# When a build fails, the end of Docker's output is shown: that is where
# the error is.
ERROR_TAIL_CHARS = 1_500

# Called with the image's tag just before a build starts, so the user
# knows why this run is slow.
BuildObserver = Callable[[str], None]


def dockerfile_text() -> str:
    """The Dockerfile that ships inside the package.

    importlib.resources finds a file that belongs to a package wherever the
    package is installed, a bit like getResourceAsStream in Java.
    """
    return resources.files("loopsmith.sandbox").joinpath(DOCKERFILE).read_text(encoding="utf-8")


def image_tag(dockerfile: str) -> str:
    """The image's name plus a fingerprint of the Dockerfile.

    Any change to the Dockerfile gives a new tag, so the next run builds a
    fresh image instead of using a stale one. Line endings are evened out
    first: a Windows checkout has "\\r\\n" where a Linux one has "\\n", and
    both must give the same tag.
    """
    fingerprint = hashlib.sha256(_with_unix_line_endings(dockerfile).encode("utf-8"))
    return f"{IMAGE_NAME}:{fingerprint.hexdigest()[:12]}"


def ensure_image(
    run_process: ProcessRunner = run_process,
    on_build: BuildObserver | None = None,
) -> str:
    """Make sure loopsmith's image exists on this machine, building it if not.

    Returns the image's tag. Raises SandboxError if Docker cannot check or
    build it.
    """
    dockerfile = dockerfile_text()
    tag = image_tag(dockerfile)
    try:
        code, _ = run_process(["docker", "image", "inspect", tag], CHECK_TIMEOUT_SECONDS)
        if code == 0:
            return tag  # built on an earlier run
        if on_build is not None:
            on_build(tag)
        # A folder holding only the Dockerfile: Docker sends the whole folder
        # to its build, and nothing else is needed for this one.
        with tempfile.TemporaryDirectory(prefix="loopsmith-image-") as folder:
            (Path(folder) / "Dockerfile").write_bytes(
                _with_unix_line_endings(dockerfile).encode("utf-8")
            )
            code, output = run_process(
                ["docker", "build", "--tag", tag, "--label", LABEL, folder],
                BUILD_TIMEOUT_SECONDS,
            )
    except FileNotFoundError:
        raise SandboxError(DOCKER_MISSING_MESSAGE) from None
    except subprocess.TimeoutExpired:
        raise SandboxError(
            "Docker did not finish preparing the sandbox image in time. Check "
            "that Docker Desktop is running and that this machine has a network "
            "connection (needed for the first build only), then try again."
        ) from None
    if code != 0:
        raise SandboxError(
            "Docker could not build the sandbox image. The first build needs a "
            f"network connection. The end of Docker's output:\n{_tail(output)}"
        )
    return tag


def _with_unix_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n")


def _tail(output: str) -> str:
    output = output.strip()
    if len(output) <= ERROR_TAIL_CHARS:
        return output
    return "[...]\n" + output[-ERROR_TAIL_CHARS:]
