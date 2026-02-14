import os
import time
from datetime import timedelta

from tasks import get_task

from .language_model import get_language_model
from .search_algo import get_search_algo
from .tracking import MetricsTracker
from .utils import create_logger, get_pacific_time
from .world_model import get_world_model


class BaseAgent:
    def __init__(
        self,
        task_name: str,
        search_algo: str,
        print_log: bool,
        log_dir: str,
        init_prompt: str,
        task_setting: dict,
        base_model_setting: dict,
        optim_model_setting: dict,
        search_setting: dict,
        world_model_setting: dict,
        tracker: MetricsTracker = None,
    ) -> None:
        """
        BaseAgent: set up task, logger, search algorithm, world model

        :param task_name: the names of .py files in the tasks folder
        :param search_algo: "mcts" or "beam_search"
        :param init_prompt: the initial prompt to start optimization from
        :param task_setting: task configuration (train_size, eval_size, etc.)
        :param base_model_setting: config for the model that answers questions
        :param optim_model_setting: config for the optimizer model that critiques & rewrites prompts
        :param search_setting: MCTS/beam search parameters
        :param world_model_setting: gradient descent engine parameters
        :param tracker: experiment tracking instance (wandb wrapper)
        """
        self.task_name = task_name
        self.search_algo_name = search_algo
        self.print_log = print_log
        self.log_dir = log_dir
        self.init_prompt = init_prompt
        self.tracker = tracker or MetricsTracker()

        self.task_setting = task_setting
        self.base_model_setting = base_model_setting
        self.optim_model_setting = optim_model_setting
        self.search_setting = search_setting
        self.world_model_setting = world_model_setting

        self.task = get_task(task_name)(**task_setting)

        if task_setting["data_dir"] is not None and task_name == "bigbench":
            task_name = (
                task_name + "_" + task_setting["data_dir"].split("/")[-1].split(".")[-2]
            )

        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}-{task_name}-algo_{self.search_algo_name}'

        self.log_dir = os.path.join(log_dir, exp_name)
        self.logger = create_logger(self.log_dir, f"{exp_name}", log_mode="train")
        self.logger.info(exp_name)
        self.log_vars()

        self.base_model = get_language_model(base_model_setting["model_type"])(
            **base_model_setting
        )

        self.optim_model = get_language_model(optim_model_setting["model_type"])(
            **optim_model_setting
        )

        self.world_model = get_world_model(self.search_algo_name)(
            task=self.task,
            logger=self.logger,
            base_model=self.base_model,
            optim_model=self.optim_model,
            tracker=self.tracker,
            **world_model_setting,
        )

        self.search_algo = get_search_algo(self.search_algo_name)(
            task=self.task,
            world_model=self.world_model,
            logger=self.logger,
            log_dir=self.log_dir,
            tracker=self.tracker,
            **self.search_setting,
        )

    def run(self):
        """Start searching from initial prompt."""
        self.logger.info(f"init_prompt: {self.init_prompt}")
        start_time = time.time()

        states, result_dict = self.search_algo.search(init_state=self.init_prompt)
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        self.tracker.log({"execution_time": exe_time})
        self.logger.info(f"\nDone! Execution time: {exe_time}")
        return states, result_dict

    def log_vars(self):
        """Log arguments."""
        ignored_print_vars = ["logger", "tracker"]
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars:
                continue
            var_value = vars_dict[var_name]
            self.logger.info(f"{var_name} : {var_value}")
