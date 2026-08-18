#!/usr/bin/env python3
"""Build the JED Kaggle code-competition kernel folder from the current attack.

Every submission setting is FIXED for this competition and expressed as an
ALL-CAPS CONSTANT below — there are no CLI flags. That is deliberate: the two
failures that broke earlier submissions (a P100 accelerator and a slug that did
not match the title) were both wrong-flag mistakes, and constants make them
unrepresentable. To change what gets built, edit a constant; then run with no
arguments.

Emits a notebook that (1) writes attack.py to /kaggle/working/, (2) py_compiles
it, and (3) on the competition RERUN serves the real gateway (producing the
scored submission.csv), else writes a zero placeholder. Also writes
kernel-metadata.json wired to the competition.
"""

import json
import py_compile
import re
import tempfile
from pathlib import Path

# --- Fixed submission config (constants, not flags, so they can't be misset) ---
KAGGLE_USER = "willrice"
COMPETITION = "ai-agent-security-multi-step-tool-attacks"
TITLE = "JED per model router 2000"  # the kernel slug is DERIVED from this (see _slugify)
MACHINE_SHAPE = "NvidiaTeslaT4"  # this competition rejects P100 (400 FAILED_PRECONDITION)
ENABLE_GPU = True
ENABLE_INTERNET = False  # rerun env is offline; attack imports only aicomp_sdk + stdlib
ATTACK_PY = "run/submission_cuts/robust_lever_2000/attack.py"  # the artifact to embed (repo-relative; a cut path also works)
OUT_ROOT = "run/submission_kernel"  # kernel folders are written under here, one per slug
# -------------------------------------------------------------------------------

# build_kernel.py -> submit-kernel -> skills -> .claude -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _slugify(title: str) -> str:
    """Kaggle derives a kernel's slug from its title; mirror that so the metadata
    ``id`` matches the pushed slug (else the status poll targets the wrong slug)."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _repo_path(rel: str) -> Path:
    """Resolve a repo-relative constant against the repo root (cwd-independent)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def build() -> Path:
    """Render the notebook + kernel-metadata.json from the constants. Returns the folder."""
    slug = _slugify(TITLE)
    attack_src = _repo_path(ATTACK_PY).read_text(encoding="utf-8")

    # Fail loudly here rather than ship a kernel whose attack.py won't import.
    # TemporaryDirectory cleans the probe .py and its .pyc rather than leaking them.
    with tempfile.TemporaryDirectory() as probe_dir:
        probe = Path(probe_dir) / "attack_probe.py"
        probe.write_text(attack_src, encoding="utf-8")
        py_compile.compile(str(probe), doraise=True)

    out_dir = _repo_path(OUT_ROOT) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    setup = (
        "import glob, os, sys\n"
        "from pathlib import Path\n"
        "sys.argv = [sys.argv[0]]\n"
        "Path('/kaggle/working').mkdir(parents=True, exist_ok=True)\n"
        "for c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n"
        "    r = str(Path(c).parent)\n"
        "    if r not in sys.path: sys.path.insert(0, r)\n"
        "    break\n"
        "print('setup done | IS_RERUN:', bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')))"
    )
    write_attack = "%%writefile /kaggle/working/attack.py\n" + attack_src
    verify = (
        "import py_compile\n"
        "py_compile.compile('/kaggle/working/attack.py', doraise=True)\n"
        "src = open('/kaggle/working/attack.py').read()\n"
        "assert 'class AttackAlgorithm(AttackAlgorithmBase)' in src\n"
        "print('attack.py OK |', len(src), 'bytes')"
    )
    serve = (
        "import os, csv\n"
        "if bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')):\n"
        "    import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server as server\n"
        "    server.JEDAttackInferenceServer().serve()\n"
        "else:\n"
        "    with open('/kaggle/working/submission.csv', 'w', newline='') as f:\n"
        "        w = csv.writer(f); w.writerow(['Id', 'Score'])\n"
        "        for r in ['gpt_oss_public', 'gpt_oss_private', 'gemma_public', 'gemma_private']:\n"
        "            w.writerow([r, 0.0])\n"
        "    print('placeholder submission.csv written')"
    )

    def code(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(keepends=True)}

    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {TITLE}\n"]},
            code(setup), code(write_attack), code(verify), code(serve),
        ],
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    nb_name = f"{slug}.ipynb"
    (out_dir / nb_name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    metadata = {
        "id": f"{KAGGLE_USER}/{slug}",
        "title": TITLE,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": ENABLE_GPU,
        "machine_shape": MACHINE_SHAPE,
        "enable_internet": ENABLE_INTERNET,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    """Build the kernel folder from the constants and print where it went."""
    folder = build()
    print(f"built kernel folder: {folder}")
    print(f"  slug:    {_slugify(TITLE)}")
    print(f"  attack:  {_repo_path(ATTACK_PY)}")
    print(f"  machine: {MACHINE_SHAPE} | gpu={ENABLE_GPU} internet={ENABLE_INTERNET}")


if __name__ == "__main__":
    main()
