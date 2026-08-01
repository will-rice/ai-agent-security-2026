#!/usr/bin/env python3
"""Build a Kaggle code-competition kernel folder for the JED attack submission.

Emits a self-contained notebook that (1) writes the given attack.py to
/kaggle/working/, (2) py_compiles it as a guard, and (3) on the competition
RERUN serves the real gateway (producing the scored submission.csv), else writes
a zero placeholder so a normal notebook run still yields a valid output. Also
writes kernel-metadata.json wired to the competition data source.

This is the host-portable replacement for the old scratchpad-hardcoded builder.

Usage:
    python build_kernel.py ATTACK_PY OUT_DIR --user KAGGLE_USER --slug SLUG
        [--competition ai-agent-security-multi-step-tool-attacks]
        [--title "..."] [--enable-gpu/--no-gpu] [--enable-internet/--no-internet]
"""

import argparse
import json
import re
from pathlib import Path

COMPETITION = "ai-agent-security-multi-step-tool-attacks"
# This competition forbids P100; the eval runs on T4. Kaggle metadata selects the
# accelerator via ``machine_shape``.
DEFAULT_MACHINE_SHAPE = "NvidiaTeslaT4"


def _slugify(title: str) -> str:
    """Kaggle derives a kernel's slug from its title; mirror that so the metadata
    ``id`` matches the pushed slug (else the status poll targets the wrong slug)."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build(attack_py: Path, out_dir: Path, *, user: str, title: str,
          competition: str, gpu: bool, internet: bool, machine_shape: str) -> Path:
    """Write the notebook + kernel-metadata.json into out_dir. Returns the folder."""
    slug = _slugify(title)
    attack_src = attack_py.read_text(encoding="utf-8")
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
    # Cell 2 uses the %%writefile magic to drop attack.py verbatim.
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
            {"cell_type": "markdown", "metadata": {},
             "source": [f"# {title}\n"]},
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
        "id": f"{user}/{slug}",
        "title": title,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "machine_shape": machine_shape,
        "enable_internet": internet,
        "dataset_sources": [],
        "competition_sources": [competition],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("attack_py", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--user", required=True)
    p.add_argument("--title", required=True, help="Kernel title; the slug is derived from it")
    p.add_argument("--competition", default=COMPETITION)
    p.add_argument("--machine-shape", default=DEFAULT_MACHINE_SHAPE,
                   help=f"Accelerator (default {DEFAULT_MACHINE_SHAPE}; this competition forbids P100)")
    p.add_argument("--enable-gpu", dest="gpu", action="store_true", default=True)
    p.add_argument("--no-gpu", dest="gpu", action="store_false")
    p.add_argument("--enable-internet", dest="internet", action="store_true", default=False)
    p.add_argument("--no-internet", dest="internet", action="store_false")
    args = p.parse_args()
    folder = build(
        args.attack_py, args.out_dir, user=args.user, title=args.title,
        competition=args.competition, gpu=args.gpu, internet=args.internet,
        machine_shape=args.machine_shape,
    )
    print(f"built kernel folder: {folder}  (slug: {_slugify(args.title)})")


if __name__ == "__main__":
    main()
