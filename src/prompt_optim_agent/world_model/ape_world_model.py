import random

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
from .world_model import WorldModel


GENERATION_TEMPLATES = {
    "forward": forward_generation_template,
    "insert": insert_generation_template,
    "insert_v2": insert_generation_template_v2,
}


class APEWorldModel(WorldModel):
    """
    World model for APE (Automatic Prompt Engineer).

    Handles:
    - Generating candidate instructions from training demonstrations
    - Evaluating candidate prompts on eval/test sets

    Inherits eval_instruction_with_loader, evaluate_prompt, test_prompt, and
    _reward_type_helper from WorldModel. APE has no gradient_descent step and
    no train_dataloader, so __init__ skips super().__init__() and sets up only
    the minimal state the inherited eval methods need.
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
        if generation_mode not in GENERATION_TEMPLATES:
            raise ValueError(
                f"generation_mode must be one of {list(GENERATION_TEMPLATES.keys())}, "
                f"got '{generation_mode}'"
            )

        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.tracker = tracker or MetricsTracker()

        self.num_subsamples = num_subsamples
        self.num_demos = num_demos
        self.num_prompts_per_subsample = num_prompts_per_subsample
        self.generation_mode = generation_mode
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

            generation_logs.append({
                "query_idx": i,
                "query": query,
                "demos": demos_str,
                "candidates": candidates_from_query,
            })

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

    def evaluate_prompt(self, prompt, candidate_idx: int = -1, origin: dict = None):
        ctx = {"phase": "evaluation", "eval_idx": candidate_idx}
        if origin:
            ctx.update(origin)
        return super().evaluate_prompt(prompt, tracker_context=ctx)

    def test_prompt(self, prompt, origin: dict = None):
        return super().test_prompt(prompt, tracker_context=origin)
