"""Tests for loopsmith's own sandbox image, without Docker.

The real build is tested in test_docker_live.py, which only runs when
asked to.
"""

import subprocess
import tomllib
from pathlib import Path

import pytest

from loopsmith.sandbox import SandboxError
from loopsmith.sandbox.docker import LABEL
from loopsmith.sandbox.image import (
    ERROR_TAIL_CHARS,
    IMAGE_NAME,
    dockerfile_text,
    ensure_image,
    image_tag,
)


class FakeDocker:
    """Answers `docker image inspect` and `docker build` as scripted, and
    keeps what the build was given."""

    def __init__(self, image_exists=False, build_result=(0, "built\n"), raises=None):
        self.image_exists = image_exists
        self.build_result = build_result
        self.raises = raises
        self.calls: list[list[str]] = []
        self.build_folder: Path | None = None
        self.built_dockerfile: bytes | None = None  # as Docker would have read it

    def __call__(self, args, timeout_seconds):
        self.calls.append(args)
        if self.raises is not None:
            raise self.raises
        if args[1:3] == ["image", "inspect"]:
            return (0, "[{...}]") if self.image_exists else (1, "Error: No such image")
        if args[1] == "build":
            self.build_folder = Path(args[-1])
            self.built_dockerfile = (self.build_folder / "Dockerfile").read_bytes()
            return self.build_result
        raise AssertionError(f"unexpected docker command: {args}")


# --- The Dockerfile and the tag ---


def test_the_dockerfile_ships_with_loopsmith():
    text = dockerfile_text()

    assert "FROM python:3.12-slim" in text
    assert "apt-get install -y --no-install-recommends git" in text
    assert "pip install --no-cache-dir pytest==9.2.0" in text


def test_the_dockerfile_is_declared_as_package_data():
    # Without this line in pyproject.toml, an installed (not editable) copy
    # of loopsmith would not include the Dockerfile.
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert "*.Dockerfile" in config["tool"]["setuptools"]["package-data"]["loopsmith.sandbox"]


def test_the_tag_is_the_image_name_and_a_fingerprint_of_the_dockerfile():
    tag = image_tag("FROM a\n")

    name, fingerprint = tag.split(":")
    assert name == IMAGE_NAME
    assert len(fingerprint) == 12
    assert all(char in "0123456789abcdef" for char in fingerprint)
    assert image_tag("FROM a\n") == tag  # the same file, the same tag
    assert image_tag("FROM b\n") != tag  # a changed file, a new tag


def test_line_endings_do_not_change_the_fingerprint():
    # A Windows checkout has \r\n where a Linux one has \n.
    assert image_tag("FROM a\r\nRUN b\r\n") == image_tag("FROM a\nRUN b\n")


# --- Making sure the image exists ---


def test_an_image_that_exists_is_not_built_again():
    docker = FakeDocker(image_exists=True)
    builds = []

    tag = ensure_image(docker, on_build=builds.append)

    assert tag == image_tag(dockerfile_text())
    assert docker.calls == [["docker", "image", "inspect", tag]]
    assert builds == []


def test_a_missing_image_is_built_from_the_shipped_dockerfile():
    docker = FakeDocker()
    builds = []

    tag = ensure_image(docker, on_build=builds.append)

    assert builds == [tag]  # the user is told before the slow part starts
    assert docker.calls[-1] == [
        "docker", "build", "--tag", tag, "--label", LABEL, str(docker.build_folder),
    ]  # fmt: skip
    # Exactly the shipped Dockerfile, with Unix line endings.
    assert docker.built_dockerfile == dockerfile_text().replace("\r\n", "\n").encode("utf-8")
    assert b"\r\n" not in docker.built_dockerfile
    # The build folder was temporary.
    assert not docker.build_folder.exists()


def test_a_failed_build_is_a_clear_error_with_the_end_of_dockers_output():
    output = "Step 1/3 ...\n" * 500 + "ERROR: No matching distribution found for pytest==9.2.0\n"
    docker = FakeDocker(build_result=(1, output))

    with pytest.raises(SandboxError) as error_info:
        ensure_image(docker)

    message = str(error_info.value)
    assert message.startswith("Docker could not build the sandbox image.")
    assert "The first build needs a network connection." in message
    assert message.endswith("No matching distribution found for pytest==9.2.0")
    assert "[...]" in message  # the long start was cut
    assert len(message) < ERROR_TAIL_CHARS + 300


def test_missing_docker_is_a_clear_error():
    with pytest.raises(SandboxError, match="Docker is not installed"):
        ensure_image(FakeDocker(raises=FileNotFoundError("docker")))


def test_a_build_that_takes_too_long_is_a_clear_error():
    timeout = subprocess.TimeoutExpired(["docker", "build"], 600)

    with pytest.raises(SandboxError, match="did not finish preparing the sandbox image"):
        ensure_image(FakeDocker(raises=timeout))
