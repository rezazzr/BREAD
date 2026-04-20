# TRAS: Stabilizing Black-Box Prompt Optimization with Textual Regularization and Signal Aggregation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Reference implementation of **TRAS** — a plug-and-play framework that stabilizes
automatic prompt optimization (APO) for black-box LLMs by augmenting the
standard *textual-gradient* update with a *textual-regularization* signal from
successful predictions and *Monte Carlo Signal Aggregation* (MCSA) over both
signals. TRAS also formalizes **Automatic Prompt Migration (APM)** — adapting
an expert prompt across model versions (e.g., GPT-3.5-turbo → GPT-4o) without
instruction loss.

> **Paper:** *Stabilizing Black-Box Prompt Optimization with Textual
> Regularization and Signal Aggregation*, Davari, Garg, Cai, Belilovsky,
> Canadian AI 2026.

Authors: MohammadReza Davari¹,²,³, Utkarsh Garg³, Weixin Cai³, Eugene Belilovsky¹,²
<br>¹ Concordia University · ² Mila – Quebec AI Institute · ³ Microsoft

---

## Quick start (≈ 1 minute, no API key)

```bash
git clone https://github.com/rezazzr/TRAS.git && cd TRAS
poetry install
NO_WANDB=1 poetry run python src/main.py -c configs/debug.yaml
```

The debug config uses an offline fake LLM (`DebugModel`) so it completes in
seconds without hitting any API. It exercises the full TRAS pipeline —
forward rollout, textual gradient, textual regularizer, MCSA aggregation,
update, MCTS search — and writes a log directory under `./logs/`.

To try the APE baseline in debug mode:

```bash
NO_WANDB=1 poetry run python src/main.py -c configs/debug_ape.yaml
```

## Running with real LLMs

1. Copy a template and fill in your OpenAI API key:
   ```bash
   cp configs/sample_config.yaml configs/my_run.local.yaml
   # edit configs/my_run.local.yaml — set api_key (or export OPENAI_API_KEY)
   ```
2. Run:
   ```bash
   poetry run python src/main.py -c configs/my_run.local.yaml
   ```

The `*.local.yaml` suffix is in `.gitignore` so you won't accidentally commit
a config that contains an API key. **Never commit a config with a key in it.**

## Reproducing the paper

Ready-to-use configs live under `configs/apo/` (Table 2 — standard APO on
GPT-3.5-turbo) and `configs/apm/` (Table 3 — APM from GPT-3.5-turbo to
GPT-4o). Each YAML leaves `api_key: null` by default; the runtime will fall
back to `OPENAI_API_KEY` from the environment.

```bash
export OPENAI_API_KEY=sk-...

# Standard APO — one task per line (Table 2)
poetry run python src/main.py -c configs/apo/causal_judgment.yaml
poetry run python src/main.py -c configs/apo/geometric_shapes.yaml
poetry run python src/main.py -c configs/apo/penguins.yaml
poetry run python src/main.py -c configs/apo/biosses.yaml
poetry run python src/main.py -c configs/apo/cb.yaml

# APM: GPT-3.5-turbo → GPT-4o (Table 3)
poetry run python src/main.py -c configs/apm/causal_judgment.yaml
poetry run python src/main.py -c configs/apm/geometric_shapes.yaml
poetry run python src/main.py -c configs/apm/penguins.yaml
poetry run python src/main.py -c configs/apm/biosses.yaml
poetry run python src/main.py -c configs/apm/cb.yaml
```

Each run produces a timestamped directory under `./logs/` with:
- `metrics.jsonl` — structured per-phase metrics (forward, gradient,
  regularizer, MCSA aggregation, optimize, per-batch eval, summaries)
- `tree_report.html` / `ape_report.html` — live HTML report you can open in
  a browser (add `open_report: true` to your config to auto-serve it)
- `data.json` — final search tree and selected best prompt

Paper accuracies are averaged over five seeds. To sweep, set different
`task_setting.seed` values.

## Paper ↔ code reference

A one-page mapping from paper terminology to code identifiers is available
at [docs/CONCEPTS.md](docs/CONCEPTS.md). Highlights:

| Paper                       | Code                                                                             |
| --------------------------- | -------------------------------------------------------------------------------- |
| **TRAS**                    | `search_algo: tras` (class `TRAS` — subclass of `MCTS`)                          |
| **PromptAgent baseline**    | `search_algo: mcts`                                                              |
| **APE baseline**            | `search_algo: ape`                                                               |
| **Textual gradient** (§3.2) | `GradientDescent.cal_gradient(..., gradient_type="negative")`                    |
| **Textual regularizer** (§3.3) | `GradientDescent.cal_gradient(..., gradient_type="positive")`                 |
| **MCSA** K samples (§3.4)   | `world_model_setting.mcsa_samples` (legacy alias: `gradient_sampling`)           |
| **Regularization warmup τ** (§3.3) | `world_model_setting.regularization_warmup_depth` (legacy: `positive_reinforcement_depth`) |

## Project layout

```
src/prompt_optim_agent/
├── agent.py                   # BaseAgent — orchestrator
├── search_algo/
│   ├── tras.py                # class TRAS(MCTS) — the paper's method
│   ├── mcts.py                # PromptAgent backbone (adapted from llm-reasoners)
│   └── ape.py                 # APE baseline
├── world_model/
│   ├── world_model.py         # shared forward / evaluation pipeline
│   ├── gradient_descent.py    # textual gradient + regularizer + MCSA
│   └── prompts/               # prompt templates (incl. ascend_* regularizer templates)
├── language_model/            # OpenAI, HuggingFace, debug backends
└── tasks/                     # BigBench-Hard + SuperGLUE task adapters

configs/                       # YAML experiment configs
├── debug.yaml                 # ~1-minute smoke test, no API
├── debug_ape.yaml             # APE variant of the debug smoke test
├── apo/                       # 5 configs for Table 2 (APO, GPT-3.5-turbo)
├── apm/                       # 5 configs for Table 3 (APM, GPT-3.5-turbo → GPT-4o)
└── sample_config.yaml         # annotated template

docs/
└── CONCEPTS.md                # paper ↔ code mapping reference

paper/                         # paper source (gitignored)
```

## Citation

If you use TRAS in your research, please cite the paper. A ready-to-use
BibTeX entry is shipped in [`CITATION.cff`](CITATION.cff); GitHub will also
render a "Cite this repository" button from that file.

## License

Apache 2.0 — see [LICENSE](LICENSE).
