"""Model specs and download-path logic (no GPU/model load)."""

from pathlib import Path

from jed_attack.harness.models import MODEL_SPECS, gguf_target_path


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
