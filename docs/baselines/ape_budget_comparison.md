# APE vs TRAS Budget Comparison

This document explains how to derive comparable APE hyperparameters from TRAS settings so that both methods use a similar computational budget.

---

## The Problem

TRAS and APE have fundamentally different architectures:
- **TRAS**: iterative refinement with tree search — many small LLM calls (forward, gradient, summary, optimize) per node
- **APE**: one-shot generation + flat evaluation — few generation calls, then one eval pass per candidate

To compare them fairly, we need to equate their budgets on a common metric.

---

## Budget Unit: Candidate Prompts Evaluated

The **dominant shared cost** is the number of candidate prompts evaluated on the eval set using the base_model. This is the primary budget axis for fair comparison.

- **TRAS**: each child node created during MCTS expansion gets evaluated on the full eval set
- **APE**: each generated candidate gets evaluated on the full eval set

Both use the same `eval_instruction_with_loader()` function, so the cost per evaluation is identical.

---

## TRAS Budget Breakdown (per expansion)

One MCTS `_expand()` call runs `expand_width` times. Each expansion calls `world_model.step()` once:

| Step | LLM | Calls | Condition |
|------|-----|-------|-----------|
| Forward pass (run prompt on batch) | base_model | 1 | always |
| Negative gradient (error analysis) | optim_model | `gradient_sampling` | always (unless acc=1.0) |
| Negative gradient summary | optim_model | 1 | only if `gradient_sampling > 1` |
| Positive gradient (success analysis) | optim_model | `gradient_sampling` | only if `depth >= positive_reinforcement_depth` AND mixed accuracy |
| Positive gradient summary | optim_model | 1 | only if above AND `gradient_sampling > 1` |
| Optimize (generate new prompts) | optim_model | 1 | always |
| Eval each child on eval set | base_model | `num_new_prompts` | always |

### Total TRAS nodes evaluated (conservative estimate)

```
total_expansions = iteration_num × expand_width
total_nodes_evaluated = total_expansions × num_new_prompts
```

**Note:** This is a lower bound. During MCTS simulation, additional nodes may be expanded, increasing the total.

---

## APE Budget Breakdown

| Step | LLM | Calls |
|------|-----|-------|
| Generate candidates | optim_model | `num_subsamples` (each produces `num_prompts_per_subsample` candidates) |
| Eval each candidate on eval set | base_model | `total_candidates` (after dedup) |

### Total APE candidates evaluated

```
total_candidates = num_subsamples × num_prompts_per_subsample
```

---

## Matching Budgets

To match on the primary budget axis (eval calls):

```
APE total_candidates ≈ TRAS total_nodes_evaluated
                     = iteration_num × expand_width × num_new_prompts
```

### Example

TRAS config: `iteration_num=12`, `expand_width=3`, `num_new_prompts=1`
→ `total_nodes_evaluated = 12 × 3 × 1 = 36`

Recommended APE config: `num_subsamples=8`, `num_prompts_per_subsample=5`
→ `total_candidates = 40` (close to 36)

### Note on optim_model cost asymmetry

APE uses significantly fewer optim_model calls than TRAS. TRAS makes `O(expansion × gradient_sampling)` optim calls, while APE makes only `num_subsamples` calls. This is an inherent difference in the methods — APE trades optimization depth for breadth.

---

## Using the Budget Utility

```python
from prompt_optim_agent.search_algo.budget_utils import compute_ape_budget_from_tras

result = compute_ape_budget_from_tras(
    iteration_num=12,
    expand_width=3,
    num_new_prompts=1,
    gradient_sampling=2,
    positive_reinforcement_depth=2,
    depth_limit=8,
    train_batch_size=5,
    num_demos=5,
)

# result contains recommended APE hyperparameters:
# result["num_subsamples"], result["num_prompts_per_subsample"], etc.
```

Run directly:
```bash
python -m prompt_optim_agent.search_algo.budget_utils
```
