import os
import time

from yaml_config_override import add_arguments

from prompt_optim_agent import BaseAgent
from prompt_optim_agent.language_model import LANGUAGE_MODELS
from prompt_optim_agent.tracking import MetricsTracker


def _check(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _check_type(cfg: dict, key: str, expected_type, label: str):
    _check(isinstance(cfg[key], expected_type), f"{label} must be {expected_type.__name__}")


def _check_choice(cfg: dict, key: str, choices: list, label: str):
    _check(cfg[key] in choices, f"{label} must be one of {choices}")


def _validate_model_setting(cfg: dict, label: str):
    """Validate a base_model_setting or optim_model_setting block."""
    valid_types = list(LANGUAGE_MODELS.keys())
    _check_choice(cfg, "model_type", valid_types, f"{label}.model_type")
    _check(cfg["model_name"] is not None, f"{label}.model_name must be specified")
    _check_type(cfg, "temperature", float, f"{label}.temperature")
    if cfg["model_type"] == "openai":
        _check(cfg.get("api_key") is not None, f"{label}.api_key is required for OpenAI models")


def validate_config(cfg):
    # Basic settings
    _check(cfg["task_name"] is not None, "task_name must be specified")
    _check_choice(cfg, "search_algo", ["mcts", "beam_search"], "search_algo")
    _check_type(cfg, "print_log", bool, "print_log")
    _check(cfg["log_dir"] is not None, "log_dir must be specified")
    _check(cfg["init_prompt"] is not None, "init_prompt must be specified")

    # Task setting
    task = cfg["task_setting"]
    _check(isinstance(task["train_size"], (int, type(None))), "task_setting.train_size must be int or None")
    _check_type(task, "eval_size", int, "task_setting.eval_size")
    _check_type(task, "test_size", int, "task_setting.test_size")
    _check_type(task, "seed", int, "task_setting.seed")
    _check_type(task, "post_instruction", bool, "task_setting.post_instruction")

    # Model settings
    _validate_model_setting(cfg["base_model_setting"], "base_model_setting")
    _validate_model_setting(cfg["optim_model_setting"], "optim_model_setting")

    # Search setting
    search = cfg["search_setting"]
    for key in ("iteration_num", "expand_width", "depth_limit", "min_depth", "beam_width"):
        _check_type(search, key, int, f"search_setting.{key}")
    _check_type(search, "w_exp", float, "search_setting.w_exp")

    # World model setting
    wm = cfg["world_model_setting"]
    _check_type(wm, "train_shuffle", bool, "world_model_setting.train_shuffle")
    _check_type(wm, "num_new_prompts", int, "world_model_setting.num_new_prompts")
    _check_type(wm, "train_batch_size", int, "world_model_setting.train_batch_size")


def main(args):
    wandb_config = args.pop("wandb", None)
    if os.environ.get("NO_WANDB"):
        wandb_config = None
    if wandb_config is not None:
        wandb_config["name"] = wandb_config.get("name", "") + "-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    tracker = MetricsTracker(wandb_config=wandb_config)

    agent = BaseAgent(tracker=tracker, **args)

    # Now that agent has created the log_dir, wire it into the tracker
    tracker.set_log_dir(agent.log_dir)
    tracker.set_config(args)

    agent.run()
    tracker.finish()


if __name__ == "__main__":
    args = add_arguments()
    validate_config(args)
    main(args)
