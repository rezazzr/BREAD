import logging
import os
from datetime import datetime
from glob import glob

# Suppress noisy third-party loggers
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.CRITICAL)


def parse_model_args(kwargs):
    base_args = dict()
    optim_args = dict()
    for k in kwargs:
        if k.startswith("base_"):
            base_args[k.replace("base_", "")] = kwargs[k]
        elif k.startswith("optim_"):
            optim_args[k.replace("optim_", "")] = kwargs[k]
    if (base_args["api_key"] is not None) and (optim_args["api_key"] is None):
        optim_args["api_key"] = base_args["api_key"]

    return base_args, optim_args


def get_pacific_time():
    return datetime.now()


def create_logger(
    logging_dir: str, name: str, log_mode: str = "train"
) -> logging.Logger:
    """
    Create a logger that writes to a log file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    if log_mode == "train":
        name += "-train"
    else:
        name += "-test"
    logging_dir = os.path.join(logging_dir, name)
    num = len(glob(logging_dir + "*"))

    logging_dir += "-" + f"{num:03d}" + ".log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}")],
    )
    logger = logging.getLogger("prompt optimization agent")
    return logger
