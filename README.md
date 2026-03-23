## TRAS: Textual Regularization with Aggregated Signals

A lightweight framework for robust automatic prompt optimization under black-box constraints.

## 🚀 Quick Start

```bash
# 1 · clone & enter the repo
git clone <your-fork-url> tras && cd tras

# 2 · set up an isolated env with Poetry
#    (→ installs exact versions from poetry.lock)
curl -sSL https://install.python-poetry.org | python3 -     # if you don't have it
poetry install                                               # resolves & installs deps
poetry shell                                                 # drop into the venv

# 3 · create a run-config
cp configs/sample_config.yaml configs/my_run.yaml
# → open configs/my_run.yaml and fill in every <…> placeholder

# 4 · run the agent
python src/main.py -c configs/my_run.yaml
```

### What the command does

1. **Loads the YAML** you just edited
2. **Spins up two language-model clients** — the base model answers questions, the optimizer model critiques & edits the prompt
3. **Performs search** (MCTS by default) to mutate the prompt until validation accuracy stops improving
4. **Logs everything** to `./logs/<timestamp>/…` and—if the `wandb` block is left intact—to Weights & Biases
