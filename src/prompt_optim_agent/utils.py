import logging
import os
from datetime import datetime
from glob import glob

# Suppress noisy third-party loggers
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.CRITICAL)


def parse_model_args(kwargs):
    base_args = {}
    optim_args = {}
    for key, value in kwargs.items():
        if key.startswith("base_"):
            base_args[key.removeprefix("base_")] = value
        elif key.startswith("optim_"):
            optim_args[key.removeprefix("optim_")] = value

    if base_args["api_key"] is not None and optim_args["api_key"] is None:
        optim_args["api_key"] = base_args["api_key"]

    return base_args, optim_args


def get_pacific_time():
    return datetime.now()


def create_logger(
    logging_dir: str, name: str, log_mode: str = "train"
) -> logging.Logger:
    """Create a file-only logger. Console output is handled by BreadConsole."""
    os.makedirs(logging_dir, exist_ok=True)

    name = f"{name}-{log_mode}"
    log_path = os.path.join(logging_dir, name)
    num = len(glob(f"{log_path}*"))
    log_path = f"{log_path}-{num:03d}.log"

    logger = logging.getLogger(f"bread.{name}.{num}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger
