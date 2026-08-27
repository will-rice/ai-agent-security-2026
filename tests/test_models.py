"""Model specs and download/serving helpers (no GPU/model load, no network)."""

from pathlib import Path

import pytest

from jed_attack.harness.models import (
    MODEL_SPECS,
    gguf_target_path,
    resolve_base_url,
    resolve_endpoints,
)


def test_model_specs_present() -> None:
    """Both target models are specified with the Kaggle-matching GGUF files."""
    assert MODEL_SPECS["gpt_oss"].repo == "unsloth/gpt-oss-20b-GGUF"
    assert MODEL_SPECS["gpt_oss"].filename == "gpt-oss-20b-Q4_K_M.gguf"
    assert MODEL_SPECS["gemma_4"].repo == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert MODEL_SPECS["gemma_4"].filename.endswith("Q4_K_M.gguf")


def test_gguf_target_path(tmp_path: Path) -> None:
    """The target path is models_dir / filename."""
    path = gguf_target_path("gpt_oss", tmp_path)
    assert path == tmp_path / "gpt-oss-20b-Q4_K_M.gguf"


def test_resolve_base_url_defaults_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base URL comes from the override, else the per-model env default."""
    monkeypatch.delenv("GPT_OSS_BASE_URL", raising=False)
    monkeypatch.delenv("GEMMA_BASE_URL", raising=False)
    assert resolve_base_url("gpt_oss") == "http://localhost:8080/v1"
    assert resolve_base_url("gemma_4") == "http://localhost:8081/v1"
    assert resolve_base_url("gpt_oss", "http://host:9/v1") == "http://host:9/v1"
    monkeypatch.setenv("GEMMA_BASE_URL", "http://env:1/v1")
    assert resolve_base_url("gemma_4") == "http://env:1/v1"


def test_resolve_endpoints_splits_env_else_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_endpoints reads *_BASE_URLS (comma list), else the single default."""
    monkeypatch.delenv("GPT_OSS_BASE_URLS", raising=False)
    assert resolve_endpoints("gpt_oss") == [resolve_base_url("gpt_oss")]
    monkeypatch.setenv(
        "GPT_OSS_BASE_URLS", "http://localhost:8080/v1, http://192.168.1.220:8080/v1"
    )
    assert resolve_endpoints("gpt_oss") == [
        "http://localhost:8080/v1",
        "http://192.168.1.220:8080/v1",
    ]
