"""Set-up shared by every test. pytest loads this file by itself.

It keeps each test away from loopsmith's real settings (decision 65).
Two problems made it necessary, both found on 2026-11-24:

- load_env_file() writes into os.environ directly, which pytest does not
  undo, so a key or model set in one test stayed set for every test
  after it. Results could then depend on the order tests run in.
- A test that runs main() made load_env_file() search for a settings
  file, which could find and load the user's real API key from
  ~/.loopsmith.env. Tests must never touch a real key.
"""

import os

import pytest

from loopsmith import config


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    # autouse: this runs around every test without being asked for, like
    # @BeforeEach and @AfterEach in a JUnit base class.
    saved = os.environ.copy()
    for name in config.OWN_SETTINGS:
        os.environ.pop(name, None)  # every test starts with no key and no model

    # The search for a settings file finds nothing, so no real key can be
    # loaded. The tests of the search itself call find_env_file directly,
    # which this leaves alone.
    monkeypatch.setattr(config, "find_env_file", lambda start=None: None)

    yield  # the test runs here

    # Put the environment back exactly as it was, whatever the test changed,
    # through monkeypatch or straight into os.environ.
    os.environ.clear()
    os.environ.update(saved)
