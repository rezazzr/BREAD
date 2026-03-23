# APE — Automatic Prompt Engineer

**Paper:** Zhou et al., "Large Language Models Are Human-Level Prompt Engineers" (ICLR 2023)
**ArXiv:** https://arxiv.org/abs/2211.01910
**Code:** https://github.com/keirp/automatic_prompt_engineer

---

## 1. Method Summary

APE treats prompt optimization as **natural language program synthesis**. Given input-output demonstrations from a task, it asks an LLM to infer what instruction could have produced those outputs, then scores and selects the best candidate.

### Algorithm (non-iterative version)

```
Require: D_train = {(Q, A)}_n: training examples, f: score function

1. candidates = []
2. for i = 1 to num_subsamples:
3.     demos = random_sample(D_train, num_demos)      # pick num_demos examples
4.     query = generation_template.fill(demos)          # build LLM query
5.     new_candidates = LLM.generate(query, n=num_prompts_per_subsample)
6.     candidates.extend(new_candidates)
7. candidates = deduplicate(candidates)
8. for each candidate in candidates:
9.     candidate.score = evaluate(candidate, D_eval)    # execution accuracy
10. return argmax(candidates, key=score)
```

**Key insight:** Each generation query uses a **different random subset** of demonstrations from the training set. This produces diverse candidate instructions because different subsets highlight different aspects of the task.

### What each hyperparameter controls

- `num_subsamples`: Number of **distinct LLM generation queries**, each with different demo examples
- `num_demos`: Number of input-output examples **per generation query** (sampled from the full training set)
- `num_prompts_per_subsample`: Number of candidate instructions the LLM generates **per query** (via temperature sampling)
- Total candidates = `num_subsamples × num_prompts_per_subsample`

---

## 2. Key Differences from TRAS

| Aspect | TRAS | APE |
|--------|------|-----|
| **Core approach** | Iterative refinement via textual gradients + MCTS tree search | One-shot generation from demos + scoring + selection |
| **Prompt generation** | Gradient descent: identify errors -> generate feedback -> propose fixes | Infer instruction from input-output demonstrations |
| **Search strategy** | MCTS with UCT, depth-limited tree exploration | Flat: generate pool -> rank -> select best |
| **Feedback signal** | Rich: error analysis, positive reinforcement, aggregated signals | None: just execution accuracy score |
| **Iteration** | Deep: each node refines parent's prompt using gradients | None (single generation pass) |
| **Complexity** | High: multiple LLM calls per node (forward + gradient + summary + optimize) | Low: one generation pass + one evaluation pass |
| **Init prompt needed?** | Yes, starts from a seed prompt and refines it | No, generates from scratch using demonstrations |

---

## 3. Prompts

### Forward Generation Template (used in instruction induction experiments)

```
I gave a friend an instruction and five inputs. The friend read the instruction
and wrote an output for every one of the inputs.
Here are the input-output pairs:

Input: [Q1]
Output: [A1]
Input: [Q2]
Output: [A2]
...

The instruction was <COMPLETE>
```

### Reverse Generation Template 1 (insert mode — used in BigBench experiments)

```
I instructed my friend to <INSERT>. The friend read the instruction and wrote
an output for every one of the inputs.
Here are the input-output pairs:

Input: [Q1]
Output: [A1]
Input: [Q2]
Output: [A2]
...
```

### Reverse Generation Template 2 (alternative insert mode)

```
Professor Smith was given the following instructions: <INSERT>
Here are the Professor's responses:

Q: [Q1]
A: [A1]
Q: [Q2]
A: [A2]
...
```

### Zero-shot Evaluation Template

```
Instruction: [INSTRUCTION]

Input: [Q_test]
Output: <COMPLETE>
```

**Note:** In our codebase, evaluation uses the existing `BaseTask.build_forward_prompts_completion()` mechanism, so we don't need a separate evaluation template.

---

## 4. Hyperparameters

| Parameter | Paper Default | Description |
|-----------|--------------|-------------|
| `num_subsamples` | 10 | Number of distinct generation queries (different demo subsets) |
| `num_demos` | 5 | Number of input-output examples per generation query |
| `num_prompts_per_subsample` | 5 | Candidates generated per query (total ~50) |
| `generation_mode` | "forward" | "forward" (completion) or "insert" (infilling) |
| Generation temperature | >0 (sampling) | For diversity in candidate generation |

---

## 5. Budget Comparison: TRAS vs APE

### TRAS LLM Calls per Node Expansion

One `world_model.step()` call (expanding a single node with one training batch) involves:

1. **Forward pass** (run current prompt on training batch): **1 base_model call**
2. **Negative gradient** (error analysis from wrong examples): **gradient_sampling × 1 optim_model call**
3. **Negative gradient summary** (if gradient_sampling > 1): **1 optim_model call**
4. **Positive gradient** (if depth >= positive_reinforcement_depth AND accuracy < 1.0 AND accuracy > 0.0): **gradient_sampling × 1 optim_model call**
5. **Positive gradient summary** (if gradient_sampling > 1 and positive gradient computed): **1 optim_model call**
6. **Optimize** (generate new prompts from gradients): **1 optim_model call**
7. **Eval** each child on eval set: **num_new_prompts × ceil(eval_size / eval_batch_size) base_model calls** (batched)

**Note:** Steps 3-5 depend on conditions (accuracy, depth, gradient_sampling). In the worst case (mixed accuracy, deep node, gradient_sampling > 1), all steps fire.

The MCTS `_expand()` repeats this `expand_width` times per node expansion (each with a different training batch). And the full search runs `iteration_num` iterations, each potentially expanding nodes during selection and simulation phases.

### Rough TRAS total budget

Per MCTS iteration (minimum — just one expansion):
- `expand_width` × (1 forward + up to `2×gradient_sampling + 2` optim calls + `num_new_prompts` eval passes)

Total nodes evaluated ≈ `iteration_num × expand_width × num_new_prompts` (lower bound; simulation may expand more nodes)

Total optim_model calls ≈ `iteration_num × expand_width × (2×gradient_sampling + 3)` (worst case with both gradients + summaries)

### APE total budget

1. **Generation:** `num_subsamples` optim_model calls (each producing `num_prompts_per_subsample` candidates)
2. **Evaluation:** `total_candidates × ceil(eval_size / eval_batch_size)` base_model calls (batched)

### Fair comparison

The dominant cost is **base_model evaluation calls**, since every candidate/node must be evaluated on the eval set. For a fair comparison, equate:

```
APE total_candidates ≈ TRAS total_nodes_evaluated
```

Where `total_nodes_evaluated` for TRAS = total number of child nodes created across all iterations.

See `src/prompt_optim_agent/search_algo/budget_utils.py` for a utility function that computes comparable APE hyperparameters given TRAS settings.

---

## 6. Code Sharing Analysis

### Fully reusable (no changes needed)
- `BaseTask` + all task implementations — APE evaluates the same way
- `BaseLanguageModel` + all model backends — APE uses the same LLM interface
- `MetricsTracker` — logging/wandb
- `Logger` — file logging
- Config loading infrastructure in `main.py`

### Reusable with extraction
- `WorldModel.eval_instruction_with_loader()` — APE needs the same eval logic
- `WorldModel.build_root()` pattern — APE can evaluate an initial prompt similarly
- `MCTSNode` — APE can reuse for result compatibility (prompt + reward)

### Not reusable (APE-specific)
- `GradientDescent` — APE doesn't use textual gradients
- `MCTS` — APE doesn't use tree search
- All gradient descent prompts — replaced by APE's generation prompts

---

## 7. Implementation Notes

### Files created
- `src/prompt_optim_agent/world_model/prompts/ape_prompts.py` — APE generation prompt templates
- `src/prompt_optim_agent/world_model/ape_world_model.py` — APEWorldModel (generation + evaluation)
- `src/prompt_optim_agent/search_algo/ape.py` — APE search algorithm
- `src/prompt_optim_agent/search_algo/ape_reporter.py` — APE-specific logging/reporting
- `src/prompt_optim_agent/search_algo/budget_utils.py` — TRAS↔APE budget comparison utility
- `configs/sample_ape_config.yaml` — sample APE config
- `docs/baselines/ape_budget_comparison.md` — budget comparison documentation

### Files modified
- `src/prompt_optim_agent/search_algo/__init__.py` — registered APE in SEARCH_ALGOS
- `src/prompt_optim_agent/world_model/__init__.py` — registered APEWorldModel in WORLD_MODELS
- `src/main.py` — added APE config validation, relaxed init_prompt requirement for APE
- `src/prompt_optim_agent/agent.py` — APE-aware console config panel

### Architecture decisions
- **Separate APEWorldModel** rather than extending existing WorldModel. The existing WorldModel has `GradientDescent` as a hard dependency in `__init__`. Adding conditionals would be fragile. APEWorldModel duplicates `eval_instruction_with_loader()` — acceptable tradeoff for clean separation.
- **MCTSNode reused** for APE candidates to maintain compatibility with `agent.run()` which checks `result_dict["best_reward_path_selected_node"]`.
- **Generation params in `world_model_setting`** because they're consumed by APEWorldModel's constructor. The search algo reads them from `self.world_model`.
- **No iterative APE** (Monte Carlo resampling) — the paper shows marginal improvement and we're comparing against TRAS which has a fundamentally different search strategy.

### How to run

```bash
python src/main.py -c configs/sample_ape_config.yaml
```

### How to view the live HTML report

APE generates a live HTML report (`ape_report.html`) that updates as candidates are generated and scored. To view it during or after a run:

```bash
# Start a local server in the log directory
cd logs/<your-run-directory>/
python -m http.server 8000

# Open in browser
# http://localhost:8000/ape_report.html
```

The report polls `ape_report_data.json` every 2 seconds and shows:
- Pipeline progress (Generate → Dedup → Evaluate → Select → Test)
- Summary stats (generated/unique counts, best scores, timing)
- Candidate ranking bar chart with scores and origin tracing
- Per-example eval/test breakdown for the best candidate
- Generation query details (demos used, candidates produced)

### How to compute comparable budget

```python
from prompt_optim_agent.search_algo.budget_utils import compute_ape_budget_from_tras
compute_ape_budget_from_tras(iteration_num=12, expand_width=3, num_new_prompts=1, gradient_sampling=2, positive_reinforcement_depth=2, depth_limit=8)
```
