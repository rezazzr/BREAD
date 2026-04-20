"""Inspect the most recent TRAS / MCTS / APE run.

Loads ``logs/<most-recent>/metrics.jsonl``, counts phase occurrences,
and prints the best prompt found. Useful as a sanity check after running
a config.

Usage:
    poetry run python examples/inspect_run.py
    poetry run python examples/inspect_run.py path/to/run_dir
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(run_dir: Path | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    if run_dir is None:
        logs = sorted((repo / "logs").iterdir(), key=lambda p: p.stat().st_mtime)
        if not logs:
            print("No run directories under logs/. Run a config first.", file=sys.stderr)
            return 1
        run_dir = logs[-1]
    else:
        run_dir = Path(run_dir)

    print(f"Inspecting: {run_dir}\n")

    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        print(f"No metrics.jsonl in {run_dir}", file=sys.stderr)
        return 1

    phases = Counter()
    best_reward = float("-inf")
    last_entry = None
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if (phase := entry.get("phase")):
            phases[phase] += 1
        if "best_reward" in entry:
            best_reward = max(best_reward, entry["best_reward"])
        last_entry = entry

    print("Phase counts:")
    for phase, count in sorted(phases.items(), key=lambda kv: -kv[1]):
        print(f"  {phase:30} {count}")

    if best_reward != float("-inf"):
        print(f"\nBest eval reward observed: {best_reward:.4f}")

    data_json = run_dir / "data.json"
    if data_json.exists():
        data = json.loads(data_json.read_text())
        best_path = data.get("best_reward_path_selected_node") or []
        if best_path:
            best = best_path[0] if isinstance(best_path, list) else best_path
            prompt = best.get("prompt") if isinstance(best, dict) else None
            if prompt:
                print(f"\nSelected best prompt:\n{prompt}")

    report = run_dir / "tree_report.html"
    if report.exists():
        print(f"\nTree report: file://{report.resolve()}")
    return 0


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(arg))
