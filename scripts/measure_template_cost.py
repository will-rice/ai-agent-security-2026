#!/usr/bin/env python3
"""Measure each fill template's DETERMINISTIC generation cost on the bottleneck model.

The T4 replay cost per candidate is driven by how many characters the victim model
GENERATES to fire the tool call. That output is deterministic (greedy decode, fixed
seed), so it is a noise-free per-template throughput proxy — unlike wall-clock, which
the pipeline audit found is corrupted by concurrent GPU load. Fewer generated chars ->
faster replay -> more candidates fit the fixed T4 budget -> higher score. Ranking
templates by generated chars therefore optimizes T4 throughput entirely locally.

gpt_oss is the throughput bottleneck (GREEN_REPLAY_BUDGET_S is 5x gemma's), so measure
it. Run pinned to the idle GPU so it does not disturb the live optimizer on GPU 0:

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/measure_template_cost.py

A small n_ctx keeps the KV cache tiny so this gpt_oss fits alongside the optimizer's
resident gemma on the same card; the short prompts make the generated output identical
to a full-context run.
"""

import json
from pathlib import Path
from typing import cast

from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.evaluation.ops import resolve_fixtures_dir

from jed_attack.campaign import config, guardrails
from jed_attack.harness.models import gguf_agent_factory, gguf_target_path

MODEL = "gpt_oss"
N_CTX = 2048
SEED = 123
ATTACK_PATH = Path("run/build_next/attack.py")  # the live shipped template bank
OUT_PATH = Path("run/template_cost.json")
INDICES = (800001, 800002)  # distinct hosts; per-message output is deterministic


def _load_bank(attack_path: Path) -> dict:
    """Exec the rendered attack.py to recover its _TEMPLATES and _message renderer."""
    namespace: dict = {}
    code = compile(attack_path.read_text(encoding="utf-8"), "attack.py", "exec")
    exec(code, namespace)  # noqa: S102
    return namespace


def _gen_chars(trace: dict) -> int:
    return sum(
        len(m) for m in trace.get("assistant_messages", []) if m and m != "(no_tool)"
    )


def main() -> None:
    """Replay each template on the bottleneck model and rank by generated chars."""
    bank = _load_bank(ATTACK_PATH)
    templates = bank["_TEMPLATES"]
    render = bank["_message"]

    gguf = gguf_target_path(MODEL, config.MODELS_DIR)
    # main_gpu=0 -> the only CUDA_VISIBLE_DEVICES device (physical GPU 1); small n_ctx
    # keeps the KV cache tiny so it fits beside the optimizer's resident gemma there.
    factory = gguf_agent_factory(MODEL, gguf, n_ctx=N_CTX, main_gpu=0)
    guardrail = guardrails.GATE_GUARDRAILS[config.GATE_GUARDRAIL_NAME]
    hops = config.EVAL_HOPS
    fixtures = resolve_fixtures_dir()

    def replay(message: str) -> tuple[dict, list]:
        factory.reset()
        env = SandboxEnv(
            seed=SEED,
            fixtures_dir=fixtures,
            agent=factory(),
            guardrail=guardrail(),
            max_tool_hops=hops,
        )
        env.reset()
        env.interact(message, max_tool_hops=hops)
        trace = env.export_trace_dict()
        return trace, eval_predicates(trace)

    rows: list[dict[str, object]] = []
    for name, template in templates:
        chars: list[int] = []
        fires = 0
        for index in INDICES:
            trace, predicates = replay(render(template, index))
            chars.append(_gen_chars(trace))
            if predicates:
                fires += 1
        rows.append(
            {
                "template": name,
                "fire_rate": fires / len(INDICES),
                "gen_chars": round(sum(chars) / len(chars), 1),
            }
        )

    # Firing templates first, then cheapest generation cost (best throughput).
    rows.sort(
        key=lambda r: (
            cast("float", r["fire_rate"]) < 1.0,
            cast("float", r["gen_chars"]),
        )
    )
    print(f"{'template':26s} {'fire_rate':>9s} {'gen_chars':>9s}")
    for row in rows:
        print(
            f"{row['template']:24s} {row['fire_rate']:>8.2f} {row['gen_chars']:>8.0f}"
        )
    OUT_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
