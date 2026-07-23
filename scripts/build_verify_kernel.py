from pathlib import Path
sp = Path("/private/tmp/claude-501/-Users-will-Documents-projects-ai-agent-security-2026/800394c8-bbb7-4e8f-ac1c-421b4d98b024/scratchpad")
attack = sp.joinpath("composed_attack.py").read_text()
harness = r'''
# ======================================================================
# JED submission verifier — runs the REAL competition gateway locally on
# a Kaggle T4 to get the public score + per-candidate timing + pass/fail
# BEFORE spending a submission slot. Public guardrail only (the private
# guardrail wheel is not available to competitors locally).
#   Controls (Kaggle "Add-on" env or edit here):
#     JED_SMOKE=1  -> short budget, gpt_oss only (fast sanity)
#     JED_SMOKE=0  -> full 9000s budget, both models (true pre-submit gate)
# ======================================================================
import csv, glob, json, os, subprocess, sys, time
from pathlib import Path

SMOKE = os.getenv("JED_SMOKE", "1") == "1"
BUDGET_S = float(os.getenv("JED_BUDGET_S", "600" if SMOKE else "9000"))
MODELS = os.getenv("AICOMP_MODEL_NAMES", "gpt_oss" if SMOKE else "gpt_oss,gemma")

def _log(m): print(f"[jed-verify] {m}", flush=True)

# 0. ensure the GGUF backend + hub client (competition image has them; be defensive)
for mod, pipname in [("llama_cpp", "llama-cpp-python"), ("huggingface_hub", "huggingface_hub")]:
    try:
        __import__(mod)
    except Exception:
        _log(f"installing {pipname} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pipname], check=False)

# 1. mount the competition-provided aicomp_sdk + kaggle_evaluation on sys.path
for cand in glob.glob("/kaggle/input/**/kaggle_evaluation", recursive=True):
    root = os.path.dirname(cand)
    if root not in sys.path:
        sys.path.insert(0, root)
    _log(f"mounted harness from {root}")
    break

# 2. fixtures dir (web_corpus.json + mail_seed.json + file_seed/)
FIXTURES = None
for f in glob.glob("/kaggle/input/**/web_corpus.json", recursive=True):
    p = Path(f).parent
    if (p / "mail_seed.json").exists():
        FIXTURES = str(p); break
_log(f"fixtures = {FIXTURES}")

# 3. write our composed attack.py where the inference server expects it
Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
Path("/kaggle/working/attack.py").write_text(ATTACK_SOURCE)
_log(f"wrote attack.py ({len(ATTACK_SOURCE)} bytes)")

# 4. env: model selection + optional pre-downloaded GGUFs (else the model server
#    hf_hub_downloads them; needs notebook internet ON). Private guardrail left unset.
os.environ["AICOMP_MODEL_NAMES"] = MODELS
for env_key, fname in [("GPT_OSS_MODEL_PATH", "gpt-oss-20b-Q4_K_M.gguf"),
                       ("GEMMA_MODEL_PATH", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")]:
    if env_key not in os.environ:
        hits = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
        if hits:
            os.environ[env_key] = hits[0]; _log(f"{env_key} = {hits[0]}")

import kaggle_evaluation.jed_attack_134815.jed_attack_gateway as jag
jag.DEFAULT_BUDGET_S = BUDGET_S
jag.GATEWAY_RESPONSE_TIMEOUT_S = int(BUDGET_S + jag.ATTACK_ENV_OP_GRACE_S + jag.GATEWAY_RESPONSE_TIMEOUT_BUFFER_S)
_log(f"budget={BUDGET_S}s models={MODELS} (public guardrail only)")

# 5. run the real gateway (streams [ATTACK]/[REPLAY]/[GATEWAY] + per-candidate timing)
from kaggle_evaluation.jed_attack_134815 import jed_attack_inference_server as ies
t0 = time.time()
status = "FAIL"
try:
    ies.JEDAttackInferenceServer().run(competition_data_folder=FIXTURES)
    status = "PASS"; _log(f"=== PASS === ran clean in {time.time() - t0:.0f}s")
except Exception as e:
    et = getattr(e, "error_type", None); ed = str(getattr(e, "error_details", e))[:500]
    _log(f"=== FAIL === {type(e).__name__}: {et} {ed}")

# 6. report score + detail
if Path("submission.csv").exists():
    _log("submission.csv:\n" + Path("submission.csv").read_text())
    rows = list(csv.DictReader(open("submission.csv")))
    pub = [float(r["Score"]) for r in rows if r["Id"].endswith("_public")]
    if pub:
        _log(f"PUBLIC LB (mean over models) = {sum(pub) / len(pub):.3f}   [{status}]")
if Path("submission_details.json").exists():
    d = json.load(open("submission_details.json"))
    _log(f"guardrails scored = {d.get('guardrail_configs')}")
'''
Path("/private/tmp/claude-501/-Users-will-Documents-projects-ai-agent-security-2026/800394c8-bbb7-4e8f-ac1c-421b4d98b024/scratchpad/jed_verify_kernel.py").write_text(
    "ATTACK_SOURCE = r'''\n" + attack + "\n'''\n" + harness
)
print("built jed_verify_kernel.py")
