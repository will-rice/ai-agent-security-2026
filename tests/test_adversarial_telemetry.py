"""Tests for the adversarial search's wandb telemetry wrapper (pure, no wandb run)."""

import pytest

from jed_attack.adversarial import telemetry


def test_log_is_noop_without_active_run() -> None:
    """log() must not raise when no wandb run was ever initialized."""
    assert telemetry.log({"a": 1}) is None


def test_init_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """init() must not start a run when JED_WANDB=0."""
    monkeypatch.setenv("JED_WANDB", "0")
    telemetry.init("x", {})
    import wandb

    assert wandb.run is None
