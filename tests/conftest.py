"""Shared test setup."""

import pytest

from chatbrain import paths


@pytest.fixture(autouse=True)
def _no_real_data_folder(monkeypatch):
    # No test may reach the real data folder, whatever the shell sets: a test that needs
    # CHATBRAIN_DATA sets it to a folder under its own tmp_path.
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
