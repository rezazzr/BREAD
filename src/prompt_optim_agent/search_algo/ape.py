"""APE (Automatic Prompt Engineer) search algorithm.

Implements the non-iterative APE method:
  1. Generate candidate instructions from training demonstrations
  2. Deduplicate
  3. Score each candidate on the eval set
  4. Select the best
"""

import logging
import os
import time
from typing import Optional

from ..console import get_console
from ..tracking import MetricsTracker
from ..world_model.ape_world_model import APEWorldModel
from .ape_reporter import APEReporter
from .base_algo import SearchAlgo
from .mcts_tree_node import MCTSNode


class APE(SearchAlgo):

    def __init__(
        self,
        task,
        world_model: APEWorldModel,
        # logging
        logger: Optional[logging.Logger] = None,
        log_dir: str = None,
        tracker: MetricsTracker = None,
        **kwargs,
    ) -> None:
        """
        APE search algorithm.

        Generation parameters (num_subsamples, num_demos, num_prompts_per_subsample,
        generation_mode) are configured on the APEWorldModel via world_model_setting.

        :param task: the specific task
        :param world_model: APEWorldModel for candidate generation and evaluation
        :param logger: logger
        :param log_dir: logger directory
        :param tracker: experiment metrics tracker
        """
        self.task = task
        self.world_model = world_model
        self.tracker = tracker or MetricsTracker()

        self.log_dir = log_dir or os.getcwd()
        self.logger = logger

        self.nodes: list[MCTSNode] = []

        self.reporter = APEReporter(
            logger=self.logger,
            tracker=self.tracker,
            world_model=self.world_model,
            task=self.task,
            log_dir=self.log_dir,
        )
        self.reporter.log_vars({
            "num_subsamples": self.world_model.num_subsamples,
            "num_demos": self.world_model.num_demos,
            "num_prompts_per_subsample": self.world_model.num_prompts_per_subsample,
            "generation_mode": self.world_model.generation_mode,
        })

    def search(self, init_state: str = None):
        """
        Run the APE algorithm.

        Args:
            init_state: ignored for APE (APE generates from scratch).
                        Accepted for interface compatibility with SearchAlgo.

        Returns:
            ([], result_dict) — empty trace list (no iterations), result dict
        """
        console = get_console()
        MCTSNode.reset_id()

        # ---- Phase 1: Generate candidates ----
        console.phase("GENERATE", "generating candidate instructions...")
        self.logger.info("=" * 60)
        self.logger.info("Phase 1: Generating candidate instructions")
        self.logger.info("=" * 60)

        html_report = self.reporter.html_report

        gen_start = time.time()
        all_candidates, generation_logs = self.world_model.generate_candidates()
        gen_time = time.time() - gen_start

        for log_entry in generation_logs:
            self.reporter.log_generation_query(
                log_entry["query_idx"] + 1,
                self.world_model.num_subsamples,
                log_entry["query"],
                log_entry["candidates"],
            )
            html_report.update_generation(log_entry)

        html_report.update_timing({"generation_time": round(gen_time, 3)})

        # ---- Phase 2: Deduplicate ----
        unique_candidates = APEWorldModel.deduplicate(all_candidates)
        self.reporter.log_dedup_stats(len(all_candidates), len(unique_candidates))
        html_report.update_dedup(len(all_candidates), len(unique_candidates))
        console.status(
            f"Generated {len(all_candidates)} candidates, "
            f"{len(unique_candidates)} unique after deduplication"
        )

        if not unique_candidates:
            self.logger.warning("No candidates generated. Returning empty results.")
            return [], {}

        # ---- Phase 3: Score each candidate ----
        console.phase("EVALUATE", f"scoring {len(unique_candidates)} candidates...")
        self.logger.info("=" * 60)
        self.logger.info(f"Phase 2: Evaluating {len(unique_candidates)} candidates")
        self.logger.info("=" * 60)

        eval_start = time.time()
        scored_candidates = []
        for i, cand_entry in enumerate(unique_candidates):
            prompt = cand_entry["prompt"]
            origin = cand_entry["origin"]

            self.logger.info(
                f"\n--- Evaluating candidate {i + 1}/{len(unique_candidates)} "
                f"(from query {origin['query_idx']}, idx {origin['candidate_idx_in_query']}) ---"
            )
            console.status(
                f"Evaluating candidate {i + 1}/{len(unique_candidates)}"
            )

            eval_result = self.world_model.evaluate_prompt(
                prompt, candidate_idx=i, origin=origin
            )
            score = self.world_model._reward_type_helper(eval_result["metric"])

            # Create MCTSNode for compatibility
            node = MCTSNode(prompt=prompt, action=None, parent=None)
            node.reward = score
            self.nodes.append(node)

            num_correct = sum(1 for c in eval_result["correct"] if c == 1)
            num_total = len(eval_result["correct"])

            scored_candidates.append({
                "prompt": prompt,
                "score": score,
                "correct": eval_result["correct"],
                "acc": eval_result["acc"],
                "origin": origin,
                "node": node,
                "eval_output": eval_result["eval_output"],
            })

            self.tracker.log({
                "phase": "evaluation_summary",
                "eval_idx": i,
                **origin,
                "candidate_score": score,
                "num_correct": num_correct,
                "num_total": num_total,
            })

            # Live HTML update
            html_report.update_candidate_score({
                "prompt": prompt,
                "eval_score": score,
                "origin": origin,
                "num_correct": num_correct,
                "num_total": num_total,
                "correct": eval_result["correct"],
            })

        eval_time = time.time() - eval_start
        html_report.update_timing({"evaluation_time": round(eval_time, 3)})

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        self.reporter.log_candidate_scores(scored_candidates)

        best_entry = scored_candidates[0]
        best_node = best_entry["node"]

        self.logger.info(f"\nBest candidate: score={best_entry['score']:.4f}")
        self.logger.info(f"Prompt: {best_entry['prompt']}")

        timing = {
            "generation_time": round(gen_time, 3),
            "evaluation_time": round(eval_time, 3),
        }

        result_dict = self.reporter.prepare_output(
            scored_candidates=scored_candidates,
            best_node=best_node,
            best_origin=best_entry["origin"],
            best_eval_output=best_entry["eval_output"],
            all_nodes=self.nodes,
            generation_logs=generation_logs,
            timing=timing,
        )
        self.reporter.save_to_json(result_dict)

        return [], result_dict
