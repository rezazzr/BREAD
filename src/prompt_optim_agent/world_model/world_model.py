from typing import Generic, List

import wandb
from tqdm import tqdm

from src.prompt_optim_agent.language_model.base_model import BaseLanguageModel
from src.tasks.base_task import BaseTask

from ..search_algo.base_algo import Action, State
from ..search_algo.mcts import MCTSNode
from .gradient_descent import *


class WorldModel(Generic[State, Action]):
    def __init__(
        self,
        task: BaseTask,
        logger,
        base_model: BaseLanguageModel,
        optim_model,
        num_new_prompts=1,
        train_shuffle=True,
        train_batch_size: int = 5,
        test_batch_size: int = 1,
        eval_batch_size: int = 1,
        print_log: bool = True,
        **kwargs,
    ) -> None:
        """
        WorldModel is responsible for:
            State transition (generate new prompt based on the given node and batch data);
            Calculating reward for the given nodes;
            Calculating test metric on the test dataset.
        """

        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.num_new_prompts = num_new_prompts
        self.positive_reinforcement_depth = kwargs.get(
            "positive_reinforcement_depth", 0
        )

        self.train_shuffle = train_shuffle
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_dataloader = self.task.get_dataloader(
            "train", batch_size=train_batch_size, shuffle=train_shuffle
        )
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)

        self.test_dataloader = self.task.get_dataloader(
            "test", batch_size=test_batch_size, shuffle=False
        )
        self.eval_dataloader = self.task.get_dataloader(
            "eval", batch_size=eval_batch_size, shuffle=False
        )

        gradient_sampling = kwargs["gradient_sampling"]
        assert (
            gradient_sampling >= 1 and gradient_sampling % 1 == 0
        ), "gradient_sampling must be an integer greater than or equal to 1."

        self.gradient_descent = GradientDescent(
            task=self.task,
            logger=self.logger,
            base_model=base_model,
            optim_model=optim_model,
            num_new_prompts=num_new_prompts,
            print_log=print_log,
            positive_reinforcement_depth=self.positive_reinforcement_depth,
            gradient_sampling=gradient_sampling,
        )

        self.log_vars()

    def log_vars(self):
        """
        Log world_model arguments.
        """
        self.logger.info("----------------- World Model --------------------------")
        ignored_print_vars = [
            "task",
            "logger",
            "train_dataloader",
            "train_data_iterator",
            "test_dataloader",
            "eval_dataloader",
            "gradient_descent",
        ]
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars:
                continue
            var_value = vars_dict[var_name]
            self.logger.info(f"{var_name} : {var_value}")

    def _infinite_data_loader(self, data_loader):
        """
        Yield batches from dataloader.
        """
        while True:
            for batch in data_loader:
                yield batch

    def get_train_batch(self):
        return next(self.train_data_iterator)

    def _get_trajectory_prompts(self, node: MCTSNode):
        """
        Collect the trajectory of prompts from the root node to the given node.
        """
        trajectory_prompts = []
        temp_node = node
        trajectory_prompts.append(temp_node.prompt)
        while temp_node.parent is not None:
            temp_node = temp_node.parent
            trajectory_prompts.append(temp_node.prompt)
        return trajectory_prompts[::-1]

    def build_root(self, init_prompt):
        """
        Build root MCTSNode using the initial prompt
        """
        node = MCTSNode(prompt=init_prompt, action=None, parent=None)
        node.reward = self._reward_type_helper(
            self.evaluate_prompt(prompt=node.prompt)["metric"]
        )
        return node

    def step(self, node: MCTSNode, batch):
        """
        Optimization step:
            Generate new nodes based on the given node and batch of data.
        """
        new_nodes, gradient_descent_output = self._gradient_descent_step(
            node=node, batch=batch
        )
        return new_nodes, gradient_descent_output

    def _gradient_descent_step(
        self, node: MCTSNode, batch
    ) -> Tuple[List[MCTSNode], dict]:
        trajectory_prompts = self._get_trajectory_prompts(node=node)
        helper_data = dict(trajectory_prompts=trajectory_prompts)

        gradient_descent_output = self.gradient_descent(
            batch, node.prompt, helper_data, node.depth
        )
        wandb.log(
            {
                "node_id": node.id,
                "depth": node.depth,
                "batch_acc": gradient_descent_output["acc"],
            }
        )

        new_nodes = []
        for prompt in gradient_descent_output["optimized_prompts"]:
            child_node = MCTSNode(
                prompt=prompt,
                action=gradient_descent_output["optimized_prompts"],
                parent=node,
            )
            new_nodes.append(child_node)

        return new_nodes, gradient_descent_output

    def evaluate_child_node(self, node: MCTSNode):
        """
        Evaluate the given node on eval_dataloader to calculate the reward.
        """
        evaludate_output = self.evaluate_prompt(prompt=node.prompt)
        node.reward = self._reward_type_helper(evaludate_output["metric"])

    def evaluate_prompt(self, prompt):
        """
        Evaluate prompt on eval_dataloader to calculate the reward.
        """
        self.logger.info(f"prompt: {prompt}")
        metric, eval_output = self.eval_instruction_with_loader(
            task=self.task,
            eval_prompt=prompt,
            dataloader=self.eval_dataloader,
        )

        correct = eval_output["correct"]
        correct_np = np.array(correct)
        acc = correct_np[correct_np > -1].mean()
        evaludate_output = dict(metric=metric, correct=correct, acc=acc)
        return evaludate_output

    def test_prompt(self, prompt):
        """
        Test prompt on test_dataloader.
        """
        metric, eval_output = self.eval_instruction_with_loader(
            task=self.task,
            eval_prompt=prompt,
            dataloader=self.test_dataloader,
            use_test_metrics=True,
        )
        return metric, eval_output

    def eval_instruction_with_loader(
        self,
        task,
        eval_prompt,
        dataloader,
        record_outputs: bool = True,
        use_test_metrics: bool = False,
    ):
        """
        Evaluate eval_prompt on the given dataloader.
        Output:
            metric: task specific evaluation metric, e.g. Accuracy
            eval_output: the input question and predictions for each example in the dataloader
        """
        build_forward_prompts_func = task.build_forward_prompts_completion
        batch_forward_func = self.base_model.batch_forward_func

        all_questions = []
        all_labels = []
        all_preds = []
        all_prompts = []
        all_responses = []
        eval_output = {}

        pbar = tqdm(dataloader, leave=False)
        for batch in pbar:
            batch_prompts = build_forward_prompts_func(batch["question"], eval_prompt)
            responses, loging_dict = batch_forward_func(batch_prompts)
            wandb.log(
                {f"{key}_base_model": value for key, value in loging_dict.items()}
            )
            preds = task.batch_clean_responses(responses)
            # check to see if this particular task comes with answers
            batch_answers = batch.get("answer", None)
            labels = (
                task.clean_labels(batch_answers) if batch_answers is not None else None
            )
            all_preds.extend(preds)
            if labels is not None:
                all_labels.extend(labels)
            all_questions.extend(batch["question"])
            if record_outputs:
                all_prompts.extend(batch_prompts)
                all_responses.extend(responses)

        if record_outputs:
            eval_output["model_inputs"] = all_prompts
            eval_output["model_responses"] = all_responses
            eval_output["preds"] = all_preds
            eval_output["labels"] = all_labels
        eval_output["correct"] = task.cal_correct(
            preds=all_preds,
            questions=all_questions,
            labels=all_labels,
            prompt=eval_prompt,
            use_test_metrics=use_test_metrics,
        )
        metric = task.cal_metric_from_cal_correct_output(eval_output["correct"])
        return metric, eval_output

    def _reward_type_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
            return metric
