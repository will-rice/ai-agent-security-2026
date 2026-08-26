#!/usr/bin/env python3
"""Build a Kaggle T4 *calibration* kernel.

Runs the REAL competition gateway on a T4 to measure per-template (gen_chars,
replay-seconds) on grading hardware, WITHOUT submitting (public guardrail only, no
leaderboard slot spent).

The embedded ``run/calib/attack.py`` probes each of its 15 single-post templates
``_PROBE_REPS`` times inside ``get_attack`` and prints a ``[fill_telemetry]`` JSON
with per-template ``median_latency_s`` and ``median_gen_chars`` — measured on the T4.
Pairing those (with the reasoning-inclusive gen_chars measured locally) fits the
T4 cost model ``T4_s ~ a*hops + b*gen_tokens``, replacing the green-seconds objective.

Push (does NOT submit -> no slot):
    kaggle kernels push -p run/submission_kernel/jed-attack-calib-probe
Then read the run log:
    kaggle kernels output willrice/jed-attack-calib-probe -p <dir>   # after COMPLETE
"""

import json
import os
import py_compile
import re
import tempfile
from pathlib import Path

# --- Fixed calibration config (constants, not flags) ---
KAGGLE_USER = "willrice"
COMPETITION = "ai-agent-security-multi-step-tool-attacks"
# Which model(s) to calibrate. gpt_oss ran already; MODELS="gemma" gets the other
# row's per-template T4 latency. Slug is derived from TITLE, so vary TITLE per run.
MODELS = os.getenv("JED_CALIB_MODELS", "gemma")
# Default name = calib probe; override for a full-artifact verifier run.
TITLE = os.getenv(
    "JED_CALIB_TITLE", f"JED attack calib probe {MODELS.replace(',', '-')}"
)
MACHINE_SHAPE = "NvidiaTeslaT4"
ENABLE_GPU = os.getenv("JED_CALIB_GPU", "1") != "0"  # CPU calib: JED_CALIB_GPU=0
# Default embeds the calib probe bank; point at the shipped artifact for a real score.
ATTACK_PY = os.getenv("JED_CALIB_ATTACK_PY", "run/calib/attack.py")
OUT_ROOT = "run/submission_kernel"
# Short budget = calib probe telemetry; 9000 = the real per-model LB budget.
BUDGET_S = float(os.getenv("JED_CALIB_BUDGET_S", "900"))
# PUBLIC repos, no HF token needed. env keys the model server reads for a local path.
GGUF = {
    "gpt_oss": (
        "GPT_OSS_MODEL_PATH",
        "unsloth/gpt-oss-20b-GGUF",
        "gpt-oss-20b-Q4_K_M.gguf",
    ),
    "gemma": (
        "GEMMA_MODEL_PATH",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    ),
}
# -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build() -> Path:
    """Render the calibration notebook + kernel-metadata.json. Returns the folder."""
    slug = _slugify(TITLE)
    attack_src = (REPO_ROOT / ATTACK_PY).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as probe_dir:
        probe = Path(probe_dir) / "attack_probe.py"
        probe.write_text(attack_src, encoding="utf-8")
        py_compile.compile(str(probe), doraise=True)

    setup = (
        "import glob, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "sys.argv = [sys.argv[0]]\n"
        "Path('/kaggle/working').mkdir(parents=True, exist_ok=True)\n"
        "for mod, pip in [('llama_cpp','llama-cpp-python'),"
        "('huggingface_hub','huggingface_hub')]:\n"
        "    try: __import__(mod)\n"
        "    except Exception:\n"
        "        print('installing', pip, flush=True)\n"
        "        subprocess.run([sys.executable,'-m','pip','install','-q',pip], "
        "check=False)\n"
        "for c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n"
        "    r = str(Path(c).parent)\n"
        "    if r not in sys.path: sys.path.insert(0, r)\n"
        "    print('mounted harness from', r, flush=True); break\n"
        "print('setup done', flush=True)"
    )
    write_attack = "%%writefile /kaggle/working/attack.py\n" + attack_src
    dl_specs = [GGUF[m] for m in MODELS.split(",")]
    download = (
        "import glob, os\n"
        "from pathlib import Path\n"
        f"os.environ['AICOMP_MODEL_NAMES'] = {MODELS!r}\n"
        f"SPECS = {dl_specs!r}\n"
        "# prefer a pre-mounted GGUF, else pull the PUBLIC unsloth repo (internet on)\n"
        "for env_key, repo, fname in SPECS:\n"
        "    hits = glob.glob(f'/kaggle/input/**/{fname}', recursive=True)\n"
        "    if hits:\n"
        "        os.environ[env_key] = hits[0]\n"
        "    else:\n"
        "        from huggingface_hub import hf_hub_download\n"
        "        os.environ[env_key] = hf_hub_download(repo_id=repo, filename=fname)\n"
        "    print(env_key, '=', os.environ[env_key], flush=True)\n"
        "FIXTURES = None\n"
        "for f in glob.glob('/kaggle/input/**/web_corpus.json', recursive=True):\n"
        "    d = Path(f).parent\n"
        "    if (d / 'mail_seed.json').exists(): FIXTURES = str(d); break\n"
        "print('FIXTURES =', FIXTURES, flush=True)"
    )
    run_gateway = (
        "import csv, json, time\n"
        "from pathlib import Path\n"
        f"BUDGET_S = {BUDGET_S}\n"
        "import kaggle_evaluation.jed_attack_134815.jed_attack_gateway as jag\n"
        "jag.DEFAULT_BUDGET_S = BUDGET_S\n"
        "jag.GATEWAY_RESPONSE_TIMEOUT_S = int(BUDGET_S + jag.ATTACK_ENV_OP_GRACE_S + "
        "jag.GATEWAY_RESPONSE_TIMEOUT_BUFFER_S)\n"
        "print(f'[calib] budget={BUDGET_S}s "
        'models={os.environ["AICOMP_MODEL_NAMES"]} '
        "(public guardrail only)', flush=True)\n"
        "from kaggle_evaluation.jed_attack_134815 import jed_attack_inference_server "
        "as ies\n"
        "t0 = time.time(); status = 'FAIL'\n"
        "try:\n"
        "    ies.JEDAttackInferenceServer().run(competition_data_folder=FIXTURES)\n"
        "    status = 'PASS'; print(f'[calib] === PASS === {time.time()-t0:.0f}s', "
        "flush=True)\n"
        "except Exception as e:\n"
        "    print(f'[calib] === FAIL === {type(e).__name__}: {str(e)[:400]}', "
        "flush=True)\n"
        "# the [fill_telemetry] line (per-template median_latency_s + "
        "median_gen_chars)\n"
        "# is printed to stderr by get_attack; the score lands in submission.csv\n"
        "if Path('submission.csv').exists():\n"
        "    print('[calib] submission.csv:\\n' + Path('submission.csv').read_text(), "
        "flush=True)\n"
        "# write a non-empty placeholder so the kernel produces an output file\n"
        "with open('/kaggle/working/submission.csv','a') as f: pass"
    )

    def code(src: str) -> dict:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }

    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {TITLE}\n"]},
            code(setup),
            code(write_attack),
            code(download),
            code(run_gateway),
        ],
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_dir = REPO_ROOT / OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
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
        **({"machine_shape": MACHINE_SHAPE} if ENABLE_GPU else {}),
        "enable_internet": True,  # needed to hf_hub_download the public GGUF
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def main() -> None:
    """Build the calibration kernel folder and print the push command."""
    folder = build()
    slug = _slugify(TITLE)
    print(f"built calibration kernel: {folder}")
    print(
        f"  attack: {REPO_ROOT / ATTACK_PY}  "
        f"(probe_reps=6, {MODELS} only, {BUDGET_S:.0f}s)"
    )
    print("\nPush (runs on T4, does NOT submit -> no slot):")
    print(f"  kaggle kernels push -p {folder}")
    print(f"  kaggle kernels status {KAGGLE_USER}/{slug}")


if __name__ == "__main__":
    main()
