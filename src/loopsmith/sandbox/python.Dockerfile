# The image `loopsmith --sandbox` runs commands in, unless --sandbox-image
# names another (decision 66). loopsmith builds it from this file the first
# time it is needed. The image's tag carries a fingerprint of this file, so
# after any change here the next run builds a fresh image by itself
# (decision 67).
#
# The sandbox has no network, so anything a command may need has to be in
# the image already.

FROM python:3.12-slim

# git, so commands can read the project's history (git log, git diff). The
# project's .git is mounted read-only, so they cannot change it.
# --no-install-recommends and the cleanup keep the image small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# pytest, so "run the tests" works offline. A fixed version means the same
# Dockerfile always gives the same tools.
RUN pip install --no-cache-dir pytest==9.2.0
