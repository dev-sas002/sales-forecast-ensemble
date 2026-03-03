"""
Tests for src.config: path resolution and environment overrides.

config is a module whose values are computed at import from the environment,
so every test here reloads it under a controlled environment and restores it
afterwards. The previous version asserted the *default* DATA_DIR without
clearing DATA_DIR first, so the test passed on a bare checkout and failed
inside the container — where DATA_DIR is legitimately set — for a reason that
had nothing to do with the code being wrong.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src import config


@pytest.fixture
def fresh_config(monkeypatch: pytest.MonkeyPatch):
    """Reload src.config under a cleared environment, and restore it after."""

    def _load(**env: str):
        for var in ("PROJECT_ROOT", "DATA_DIR", "HASH_SALT", "MLFLOW_TRACKING_URI"):
            monkeypatch.delenv(var, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield _load
    # monkeypatch has undone the env by now; reload so the module state the
    # rest of the suite sees matches the real environment again.
    monkeypatch.undo()
    importlib.reload(config)


def test_root_is_the_directory_containing_src(fresh_config) -> None:
    mod = fresh_config()
    assert (mod.ROOT / "src" / "config.py").exists()


def test_data_dir_defaults_under_the_repo(fresh_config) -> None:
    mod = fresh_config()
    assert mod.DATA_DIR == mod.ROOT / "data"


def test_the_pipeline_never_writes_into_the_committed_examples(fresh_config) -> None:
    """
    Inputs and outputs must not share a directory.

    They used to: DATA_DIR defaulted to examples/sample_data/, so a pipeline
    run wrote derived parquet on top of committed files. That is how a stale
    feature table with a leaking column and a forecast file no code writes
    both ended up tracked — and then silently consumed by later runs.
    """
    mod = fresh_config()
    assert mod.EXAMPLES_DIR == mod.ROOT / "examples" / "sample_data"
    for written in (mod.DATA_DIR, mod.RAW_DIR, mod.LANDING_DIR, mod.FEATURE_DIR):
        assert mod.EXAMPLES_DIR not in written.parents
        assert written != mod.EXAMPLES_DIR


def test_data_dir_env_override(fresh_config) -> None:
    mod = fresh_config(DATA_DIR="/tmp/somewhere_else")
    assert Path("/tmp/somewhere_else") == mod.DATA_DIR
    # The derived paths must follow the override, not the default.
    assert Path("/tmp/somewhere_else/landing") == mod.LANDING_DIR
    assert Path("/tmp/somewhere_else/features") == mod.FEATURE_DIR


def test_project_root_env_override(fresh_config) -> None:
    mod = fresh_config(PROJECT_ROOT="/tmp/custom_root")
    assert Path("/tmp/custom_root") == mod.ROOT


def test_importing_config_creates_no_directories(fresh_config, tmp_path) -> None:
    # The regression that broke test collection: config used to mkdir at import,
    # so importing it against an unwritable path raised PermissionError.
    target = tmp_path / "not_created_on_import"
    mod = fresh_config(DATA_DIR=str(target))
    assert not target.exists()

    mod.ensure_dirs()
    assert (target / "landing").is_dir()
    assert (target / "features").is_dir()


def test_quantile_columns_follow_the_configured_quantiles() -> None:
    assert config.quantile_cols() == ["q0.1", "q0.5", "q0.9"]
    assert config.quantile_cols("lgb") == ["q0.1_lgb", "q0.5_lgb", "q0.9_lgb"]
    # Ascending order is relied on by the monotonicity checks downstream.
    assert list(config.QUANTILES) == sorted(config.QUANTILES)
