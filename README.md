## 🍞 BREAD: Balanced Reinforcement & Aggregated Diversification

*A lightweight, plug-and-play framework for robust Automatic Prompt Optimization and migration-ready Continual Prompt Optimization.*

---

Large-language-model prompts shouldn’t go stale every time the underlying model changes. **BREAD** keeps them fresh.

1. **Balanced Reinforcement**
   BREAD extracts **both** *negative* signals (what went wrong) and *positive* signals (what went right) from every batch of predictions. The two signals are fed back to an LLM “meta-critic” that proposes targeted edits—retaining proven instructions while fixing the flaws.

2. **Expanded Aggregated Diversification**
   Instead of trusting a single critique, BREAD samples multiple, independent feedback threads, then **aggregates** them into a consensus summary. The result: richer guidance, less noise, and dramatically lower variance across seeds.

3. **Continual Prompt Optimization (CPO)**
   Drop-in utilities adapt an expert prompt tuned on, say, `gpt-3.5-turbo` to the shiny new `gpt-4o` (or any other API model) **without hose-pipe re-tuning**. Migration convergence is 6–8 % faster in our BBH benchmarks.

---

### 1 · Prompt-optimization results  *(model: `gpt-3.5-turbo`)*

| Dataset (init. acc.)                  | Method         | Acc. (± std.)   | *p*-value | Cohen’s *d* |
| ------------------------------------- | -------------- | --------------- | --------- | ----------- |
| **Causal Judgment**<br>(56.5 ± 3.67)  | Baseline       | 58.6 ± 3.98     | –         | –           |
|                                       | +FD \*         | 60.8 ± 2.38     | 0.020     | 1.687       |
|                                       | +PR \*         | 63.6 ± 2.62     | 0.040     | 1.336       |
|                                       | **BREAD** \*\* | **64.4 ± 2.16** | 0.008     | 2.162       |
| **Geometric Shapes**<br>(32.7 ± 2.04) | Baseline       | 52.1 ± 4.94     | –         | –           |
|                                       | +FD \*         | 57.8 ± 3.15     | 0.032     | 1.439       |
|                                       | +PR \*         | 61.6 ± 4.18     | 0.035     | 1.399       |
|                                       | **BREAD** \*\* | **63.3 ± 1.16** | 0.004     | 2.644       |
| **Penguins**<br>(60.5 ± 4.87)         | Baseline       | 65.1 ± 4.96     | –         | –           |
|                                       | +FD \*         | 66.1 ± 2.42     | 0.083     | 1.148       |
|                                       | +PR \*         | 66.9 ± 3.97     | 0.043     | 1.464       |
|                                       | **BREAD** \*\* | **68.6 ± 1.87** | 0.007     | 2.556       |
| **Biosses**<br>(25.2 ± 3.84)          | Baseline       | 62.5 ± 4.19     | –         | –           |
|                                       | +FD \*         | 67.0 ± 2.92     | 0.044     | 1.456       |
|                                       | +PR \*         | 68.2 ± 3.02     | 0.021     | 1.844       |
|                                       | **BREAD** \*\* | **70.4 ± 2.02** | 0.006     | 2.654       |
| **CB**<br>(68.5 ± 4.22)               | Baseline       | 81.7 ± 3.17     | –         | –           |
|                                       | +FD \*         | 84.2 ± 2.02     | 0.032     | 1.610       |
|                                       | +PR \*         | 84.2 ± 3.73     | 0.049     | 1.402       |
|                                       | **BREAD** \*\* | **85.7 ± 3.54** | 0.008     | 2.495       |

**\*** *p* < 0.05  **\*\*** *p* < 0.01 (vs. PromptAgent baseline)

---

### 2 · Prompt-migration results  *(`gpt-3.5-turbo` ➜ `gpt-4o`)*

| Dataset (DP / EP init.)                                  | Method         | Final Acc. (± std.) | *p*-value | Cohen’s *d* |
| -------------------------------------------------------- | -------------- | ------------------- | --------- | ----------- |
| **Causal Judgment**<br>DP 71.8 ± 1.92<br>EP 74.2 ± 3.46  | Baseline       | 73.8 ± 1.79         | –         | –           |
|                                                          | +FD \*         | 74.7 ± 1.44         | 0.053     | 1.214       |
|                                                          | +PR \*         | 75.8 ± 1.78         | 0.047     | 1.265       |
|                                                          | **BREAD** \*\* | **76.4 ± 1.59**     | 0.007     | 2.280       |
| **Geometric Shapes**<br>DP 54.8 ± 1.89<br>EP 58.2 ± 2.22 | Baseline       | 75.1 ± 5.80         | –         | –           |
|                                                          | +FD \*         | 79.4 ± 2.92         | 0.016     | 1.782       |
|                                                          | +PR \*         | 81.7 ± 3.07         | 0.021     | 1.641       |
|                                                          | **BREAD** \*\* | **84.5 ± 2.33**     | 0.008     | 2.175       |
| **Penguins**<br>DP 92.9 ± 1.85<br>EP 95.8 ± 1.72         | Baseline       | 92.3 ± 2.89         | –         | –           |
|                                                          | +FD \*         | 94.2 ± 1.33         | 0.083     | 1.148       |
|                                                          | +PR \*         | 96.7 ± 0.88         | 0.041     | 1.493       |
|                                                          | **BREAD** \*\* | **98.0 ± 0.73**     | 0.008     | 2.449       |
| **Biosses**<br>DP 69.9 ± 2.73<br>EP 72.2 ± 1.78          | Baseline       | 76.1 ± 1.97         | –         | –           |
|                                                          | +FD \*         | 78.4 ± 1.66         | 0.043     | 1.461       |
|                                                          | +PR \*         | 83.7 ± 3.86         | 0.003     | 3.186       |
|                                                          | **BREAD** \*\* | **88.3 ± 2.00**     | 0.0001    | 8.696       |
| **CB**<br>DP 79.3 ± 1.57<br>EP 80.3 ± 1.54               | Baseline       | 78.7 ± 2.13         | –         | –           |
|                                                          | +FD \*         | 82.7 ± 2.31         | 0.029     | 1.659       |
|                                                          | +PR \*         | 85.3 ± 4.49         | 0.014     | 2.047       |
|                                                          | **BREAD** \*\* | **87.5 ± 3.56**     | 0.006     | 2.713       |

**DP** = default prompt  **EP** = expert prompt (tuned on `gpt-3.5-turbo`)
Significance markers as above.


### Paper

Davari *et al.* (2025) **“Prompt Migration under Black-box Constraints: Rethinking Textual Gradient with Positive and Diversified Feedback.”**
<!-- If you use BREAD, please cite the paper: -->

---

Enjoy your prompts warm, crusty, and migration-ready—powered by **BREAD**.



## 🚀 Quick Start

```bash
# 1 · clone & enter the repo
git clone <your-fork-url> bread && cd bread

# 2 · set up an isolated env with Poetry
#    (→ installs exact versions from poetry.lock)
curl -sSL https://install.python-poetry.org | python3 -     # if you don’t have it
poetry install                                               # resolves & installs deps
poetry shell                                                 # drop into the venv

# 3 · create a run-config
cp configs/sample_config.yaml configs/my_run.yaml
# → open configs/my_run.yaml and fill in every <…> placeholder

# 4 · run the agent
python src/main.py -c configs/my_run.yaml
````

### What the command actually does

1. **Loads the YAML** you just edited.
2. **Spins up two language-model clients**
   – the *base* model answers questions, the *optimiser* model critiques & edits the prompt.
3. **Performs search** (MCTS by default) to mutate the prompt until validation accuracy stops improving.
4. **Logs everything** to `./logs/<timestamp>/…` and—if the `wandb` block is left intact—to Weights & Biases.
