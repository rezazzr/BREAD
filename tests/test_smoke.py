"""End-to-end smoke tests — run the debug configs and verify metrics.

These tests exercise the full TRAS / APE pipeline against the in-process
``DebugModel`` (no API calls). They catch regressions that unit tests
would miss: config parsing, algorithm wiring, world-model / gradient-descent
interaction, logging structure.

Runtime: each test takes ~2 minutes (the debug model intentionally adds a
small latency per call so the live HTML report is visible). For faster
local iteration, run a single test: ``poetry run pytest tests/test_smoke.py::test_tras_debug``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "src" / "main.py"


def _run_config(config_path: Path, timeout_sec: int = 300) -> Path:
    """Run the pipeline with a given config, return its log directory."""
    env = {**os.environ, "NO_WANDB": "1"}
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "-c", str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    assert result.returncode == 0, (
        f"Run failed:\nSTDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )

    # Latest log directory — the pipeline creates one per run under ./logs/
    log_dirs = sorted((REPO_ROOT / "logs").iterdir(), key=lambda p: p.stat().st_mtime)
    assert log_dirs, "No log directory produced"
    return log_dirs[-1]


def _metrics_phases(log_dir: Path) -> set[str]:
    metrics = log_dir / "metrics.jsonl"
    assert metrics.exists(), f"metrics.jsonl missing in {log_dir}"
    phases = set()
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        phase = entry.get("phase")
        if phase:
            phases.add(phase)
    return phases


def _load_and_patch(src: Path, overrides: dict) -> Path:
    """Copy a config to a temp file with overrides applied."""
    data = yaml.safe_load(src.read_text())
    for k, v in overrides.items():
        keys = k.split(".")
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = v
    tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_text(yaml.safe_dump(data))
    return tmp


def test_mcts_debug_baseline():
    """The PromptAgent baseline (search_algo: mcts) runs end-to-end."""
    log_dir = _run_config(REPO_ROOT / "configs" / "debug.yaml")
    phases = _metrics_phases(log_dir)
    # Core MCTS phases that must appear
    assert {"init_eval", "forward", "gradient", "optimize"} <= phases


def test_tras_debug():
    """TRAS (search_algo: tras) runs end-to-end with the same debug config."""
    patched = _load_and_patch(
        REPO_ROOT / "configs" / "debug.yaml",
        {"search_algo": "tras"},
    )
    try:
        log_dir = _run_config(patched)
    finally:
        patched.unlink(missing_ok=True)
    phases = _metrics_phases(log_dir)
    assert {"init_eval", "forward", "gradient", "optimize"} <= phases
    # TRAS-specific: with gradient_sampling > 1, MCSA aggregation phase fires
    assert "gradient_summary" in phases, (
        f"MCSA aggregation phase missing (is gradient_sampling > 1?). "
        f"Phases seen: {phases}"
    )


def test_ape_debug_baseline():
    """The APE baseline runs end-to-end."""
    log_dir = _run_config(REPO_ROOT / "configs" / "debug_ape.yaml")
    phases = _metrics_phases(log_dir)
    assert "generation" in phases
    assert "evaluation" in phases
