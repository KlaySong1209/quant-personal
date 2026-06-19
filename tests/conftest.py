"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def placeholder_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "experiments" / "exp_placeholder.yaml"

