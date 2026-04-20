# Paper ↔ Code concept reference

This page maps the terminology used in the paper to the identifiers you will
see in the codebase. Several names in the code predate the paper, so the
mapping below is the fastest way to find the implementation of any given
paper concept.

> Paper: Davari et al., *Stabilizing Black-Box Prompt Optimization with
> Textual Regularization and Signal Aggregation*, Canadian AI 2026.

## High-level method

| Paper                                       | Code                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| **TRAS** (Section 3)                        | `search_algo.TRAS` — thin subclass of `MCTS` with paper-aligned defaults  |
| **PromptAgent baseline**                    | `search_algo.MCTS` (no textual regularizer, no MCSA)                      |
| **APE baseline**                            | `search_algo.APE`                                                         |
| **Search backbone** (MCTS from PromptAgent) | `search_algo.mcts.MCTS` — inherited unchanged by `TRAS`                   |

Select the method in your YAML config:

```yaml
search_algo: tras   # TRAS (this paper)
search_algo: mcts   # PromptAgent baseline
search_algo: ape    # APE baseline
```

## Inner-loop signals (Section 3)

| Paper                             | Code                                                                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Textual gradient** $g_t$ (3.2)  | `GradientDescent.cal_gradient(..., gradient_type="negative")`; template: `gradient_prompt_template`                                   |
| **Textual regularizer** $r_t$ (3.3) | `GradientDescent.cal_gradient(..., gradient_type="positive")`; templates prefixed `ascend_*` (historical name for the regularizer)  |
| **MCSA aggregation** (3.4)        | Triggered inside `cal_gradient` when `nb_gradient_samples > 1`; aggregation prompt: `summarization_prompt_template`                  |
| **Update operator** (3.5)         | `GradientDescent.optimize` with `optimize_prompt_template` (gradient-only) / `ascend_optimize_prompt_template` (regularizer-only) / `mix_optimize_prompt_template` (both) |

## Configuration flags

| Paper symbol                                          | Preferred YAML key          | Legacy alias (still works)     | Default | Location                       |
| ----------------------------------------------------- | --------------------------- | ------------------------------ | ------- | ------------------------------ |
| MCSA sample count $K$ (Section 3.4)                   | `mcsa_samples`              | `gradient_sampling`            | `1`     | `world_model_setting`          |
| Regularization warmup $\tau_{\text{warmup}}$ (Section 3.3) | `regularization_warmup_depth` | `positive_reinforcement_depth` | `1`     | `world_model_setting`          |

The `GradientDescent` class reads the preferred key first and falls back to
the legacy alias. Existing configs keep working.

## APM (Automatic Prompt Migration, Section 3.6)

APM is a *regime* of TRAS, not a separate class. To run the APM experiments
(GPT-3.5-turbo → GPT-4o):

- Keep `search_algo: tras`.
- Initialize from an expert prompt optimized on the source LLM (rather than
  from a task-specific default).
- Set `world_model_setting.regularization_warmup_depth: 0` (paper Section 3.6:
  "activate textual regularization immediately").
- Run on the target LLM's endpoint.

Ready-to-use configs live under `configs/apm/`.

## Per-file quick index

| File                                                                                | What it implements                                                        |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `src/prompt_optim_agent/search_algo/tras.py`                                        | `TRAS` class (entry point for the paper's method)                         |
| `src/prompt_optim_agent/search_algo/mcts.py`                                        | MCTS backbone (adapted from PromptAgent / llm-reasoners)                  |
| `src/prompt_optim_agent/search_algo/ape.py`                                         | APE baseline                                                              |
| `src/prompt_optim_agent/world_model/world_model.py`                                 | Shared world model: forward rollout, evaluation, per-batch metrics        |
| `src/prompt_optim_agent/world_model/gradient_descent.py`                            | Textual gradient + textual regularization + MCSA                          |
| `src/prompt_optim_agent/world_model/prompts/gradient_descent_prompts.py`            | Prompt templates (including the `ascend_*` regularizer templates)         |
| `src/prompt_optim_agent/agent.py`                                                   | Orchestration (`BaseAgent`): wires task + model + world model + algorithm |
| `src/main.py`                                                                       | CLI entry point: loads YAML, validates, runs the agent                    |
