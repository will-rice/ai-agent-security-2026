"""Behavioral tests for the campaign's operational shell launchers."""

import os
import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, source: str) -> None:
    """Write one executable test double."""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_optimizer_waits_for_inflight_score_before_kill(tmp_path: Path) -> None:
    """Shutdown targets only Python and lets an in-flight score exit gracefully."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "pkill.log"
    pgrep_state = tmp_path / "pgrep.state"
    _write_executable(
        fake_bin / "pkill",
        """#!/usr/bin/env bash
printf "%s\\n" "$*" >> "$COMMAND_LOG"
if [[ "$*" == *"-KILL"* ]]; then
  count=0
  if [ -f "$PGREP_STATE" ]; then
    read -r count < "$PGREP_STATE"
  fi
  if [ "$count" -le 16 ]; then
    printf "forced-while-active\\n" >> "$COMMAND_LOG"
  fi
fi
""",
    )
    _write_executable(
        fake_bin / "pgrep",
        """#!/usr/bin/env bash
count=0
if [ -f "$PGREP_STATE" ]; then
  read -r count < "$PGREP_STATE"
fi
count=$((count + 1))
printf "%s\\n" "$count" > "$PGREP_STATE"
if [ "$count" -le 16 ]; then
  exit 0
fi
exit 1
""",
    )
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "tmux", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(command_log),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PGREP_STATE": str(pgrep_state),
        }
    )
    subprocess.run(
        [_REPO / "src" / "jed_attack" / "scripts" / "run_optimizer.sh"],
        cwd=_REPO,
        env=env,
        check=True,
    )

    signals = command_log.read_text(encoding="utf-8").splitlines()
    term_signal = next(signal for signal in signals if "-TERM" in signal)
    process_pattern = term_signal.split("-f ", maxsplit=1)[1]
    assert re.search(
        process_pattern,
        "/repo/.venv/bin/python3 -m jed_attack.campaign.optimize_prompts",
    )
    assert not re.search(
        process_pattern,
        "uv run python -m jed_attack.campaign.optimize_prompts",
    )
    assert not re.search(
        process_pattern,
        "bash -lc 'uv run python -m jed_attack.campaign.optimize_prompts'",
    )
    assert "forced-while-active" not in signals


def test_judge_service_launcher_exports_selected_model(tmp_path: Path) -> None:
    """The judge service requests the same overridden model that vLLM serves."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_home = tmp_path / "home"
    vllm = fake_home / "vllm-venv" / "bin" / "vllm"
    vllm.parent.mkdir(parents=True)
    _write_executable(vllm, "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "tmux", "#!/usr/bin/env bash\nexit 0\n")
    capture = tmp_path / "model.txt"
    _write_executable(
        fake_bin / "uv",
        '#!/usr/bin/env bash\nprintf "%s" "${VLLM_MODEL-}" > "$CAPTURE_FILE"\n',
    )

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(fake_home),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TMPDIR": str(runtime_dir),
            "VLLM_MODEL": "example/override",
        }
    )
    subprocess.run(
        [_REPO / "src" / "jed_attack" / "scripts" / "serve_dylan_judges.sh"],
        cwd=_REPO,
        env=env,
        check=True,
    )

    launcher_env = env.copy()
    launcher_env.pop("VLLM_MODEL")
    launcher_env["CAPTURE_FILE"] = str(capture)
    subprocess.run(
        [runtime_dir / "jed_run_judgesvc.sh"],
        cwd=_REPO,
        env=launcher_env,
        check=True,
    )
    assert capture.read_text(encoding="utf-8") == "example/override"
