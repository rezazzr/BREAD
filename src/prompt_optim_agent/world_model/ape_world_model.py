import random
from typing import List

import numpy as np
from tqdm import tqdm

from prompt_optim_agent.console import get_console
from prompt_optim_agent.language_model.base_model import BaseLanguageModel
from prompt_optim_agent.tracking import MetricsTracker
from tasks.base_task import BaseTask

from .prompts.ape_prompts import (
    demo_template,
    demo_template_qa,
    forward_generation_template,
    insert_generation_template,
    insert_generation_template_v2,
)


GENERATION_TEMPLATES = {
    "forward": forward_generation_template,
    "insert": insert_generation_template,
    "insert_v2": insert_generation_template_v2,
}


class APEWorldModel:
    """
    World model for APE (Automatic Prompt Engineer).

    Handles:
    - Generating candidate instructions from training demonstrations
    - Evaluating candidate prompts on eval/test sets
    """

    def __init__(
        self,
        task: BaseTask,
        logger,
        base_model: BaseLanguageModel,
        optim_model: BaseLanguageModel,
        num_subsamples: int = 10,
        num_demos: int = 5,
        num_prompts_per_subsample: int = 5,
        generation_mode: str = "forward",
        eval_batch_size: int = 1,
        test_batch_size: int = 1,
        print_log: bool = True,
        tracker: MetricsTracker = None,
        **kwargs,
    ):
        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.tracker = tracker or MetricsTracker()

        self.num_subsamples = num_subsamples
        self.num_demos = num_demos
        self.num_prompts_per_subsample = num_prompts_per_subsample
        self.generation_mode = generation_mode

        if generation_mode not in GENERATION_TEMPLATES:
            raise ValueError(
                f"generation_mode must be one of {list(GENERATION_TEMPLATES.keys())}, "
                f"got '{generation_mode}'"
            )
        self.generation_template = GENERATION_TEMPLATES[generation_mode]
        self.demo_fmt = (
            demo_template_qa if generation_mode == "insert_v2" else demo_template
        )

        self.eval_dataloader = self.task.get_dataloader(
            "eval", batch_size=eval_batch_size, shuffle=False
        )
        self.test_dataloader = self.task.get_dataloader(
            "test", batch_size=test_batch_size, shuffle=False
        )

        self.train_data = self.task.dataset["train"]

        self.log_vars()

    def log_vars(self):
        self.logger.info("----------------- APE World Model --------------------------")
        self.logger.info(f"num_subsamples: {self.num_subsamples}")
        self.logger.info(f"num_demos: {self.num_demos}")
        self.logger.info(f"num_prompts_per_subsample: {self.num_prompts_per_subsample}")
        self.logger.info(f"generation_mode: {self.generation_mode}")
        self.logger.info(
            f"total candidates (before dedup): "
            f"{self.num_subsamples * self.num_prompts_per_subsample}"
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _sample_demos(self) -> str:
        """Sample num_demos random examples from training data and format them."""
        sampled = random.sample(
            self.train_data, min(self.num_demos, len(self.train_data))
        )
        formatted = [
            self.demo_fmt.format(input=ex["question"], output=ex["answer"])
            for ex in sampled
        ]
        return "\n\n".join(formatted)

    def _build_generation_query(self, demos_str: str) -> str:
        """Build a generation query from formatted demos."""
        if self.generation_mode == "forward":
            return self.generation_template.format(demos=demos_str)
        else:
            # Insert modes — the LLM fills in the blank
            return self.generation_template.format(
                instruction_placeholder="[INSERT]", demos=demos_str
            )

    def generate_candidates(self) -> tuple[list[dict], list[dict]]:
        """
        Generate candidate instructions by querying the optim_model.

        Returns:
            candidates: list of dicts with "prompt" and "origin" keys
            generation_logs: list of dicts with query details per subsample
        """
        generation_logs = []
        all_candidates = []

        for i in range(self.num_subsamples):
            demos_str = self._sample_demos()
            query = self._build_generation_query(demos_str)

            self.logger.info(f"--- Generation query {i+1}/{self.num_subsamples} ---")
            self.logger.info(f"Query:\n{query}")

            # Generate multiple candidates from the same query
            candidates_from_query = []
            for j in range(self.num_prompts_per_subsample):
                response, logging_dict = self.optim_model.generate(query)
                self.tracker.log({
                    "phase": "generation",
                    "query_idx": i,
                    "candidate_idx_in_query": j,
                    **{f"{key}_optim_model": value for key, value in logging_dict.items()},
                })
                candidate = response.strip()
                if candidate:
                    candidates_from_query.append(candidate)
                    all_candidates.append({
                        "prompt": candidate,
                        "origin": {"query_idx": i, "candidate_idx_in_query": j},
                    })

            generation_logs.append(
                {
                    "query_idx": i,
                    "query": query,
                    "demos": demos_str,
                    "candidates": candidates_from_query,
                }
            )

            self.logger.info(
                f"Generated {len(candidates_from_query)} candidates from query {i+1}"
            )

        return all_candidates, generation_logs

    @staticmethod
    def deduplicate(candidates: list[dict]) -> list[dict]:
        """Remove duplicate candidates while preserving order. Keeps first occurrence's origin."""
        seen = set()
        unique = []
        for c in candidates:
            if c["prompt"] not in seen:
                seen.add(c["prompt"])
                unique.append(c)
        return unique

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_prompt(self, prompt: str, candidate_idx: int = -1, origin: dict = None) -> dict:
        """Evaluate a prompt on the eval set."""
        ctx = {"phase": "evaluation", "eval_idx": candidate_idx}
        if origin:
            ctx.update(origin)
        metric, eval_output = self.eval_instruction_with_loader(
            task=self.task,
            eval_prompt=prompt,
            dataloader=self.eval_dataloader,
            tracker_context=ctx,
        )
        correct_np = np.array(eval_output["correct"])
        acc = correct_np[correct_np > -1].mean()

        get_console().eval_result(prompt, metric)
        return {"metric": metric, "correct": eval_output["correct"], "acc": acc}

    def test_prompt(self, prompt: str, origin: dict = None):
        """Evaluate a prompt on the test set."""
        ctx = {"phase": "test"}
        if origin:
            ctx.update(origin)
        metric, eval_output = self.eval_instruction_with_loader(
            task=self.task,
            eval_prompt=prompt,
            dataloader=self.test_dataloader,
            use_test_metrics=True,
            tracker_context=ctx,
        )
        return metric, eval_output

    def eval_instruction_with_loader(
        self,
        task,
        eval_prompt,
        dataloader,
        record_outputs: bool = True,
        use_test_metrics: bool = False,
        tracker_context: dict = None,
    ):
        """
        Evaluate eval_prompt on the given dataloader.
        Identical to WorldModel.eval_instruction_with_loader.
        """
        all_questions = []
        all_labels = []
        all_preds = []
        all_prompts = []
        all_responses = []
        eval_output = {}
        tracker_context = tracker_context or {}

        for batch in tqdm(dataloader, leave=False):
            batch_prompts = task.build_forward_prompts_completion(
                batch["question"], eval_prompt
            )
            responses, logging_dict = self.base_model.batch_forward_func(batch_prompts)
            self.tracker.log({
                **tracker_context,
                **{f"{key}_base_model": value for key, value in logging_dict.items()},
            })
            preds = task.batch_clean_responses(responses)
            batch_answers = batch.get("answer", None)
            labels = (
                task.clean_labels(batch_answers)
                if batch_answers is not None
                else None
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
        return metric[0] if isinstance(metric, tuple) else metric
