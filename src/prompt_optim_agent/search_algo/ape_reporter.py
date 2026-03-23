"""APE reporting and output formatting.

Handles logging, evaluation, metric tracking, and JSON serialization
of APE search results.
"""

import json
import logging
import os

import numpy as np

from ..ape_html_report import ApeHtmlReport
from ..console import get_console
from ..tracking import MetricsTracker
from .mcts_tree_node import MCTSNode


class APEReporter:
    def __init__(
        self,
        logger: logging.Logger,
        tracker: MetricsTracker,
        world_model,
        task,
        log_dir: str,
    ):
        self.logger = logger
        self.tracker = tracker
        self.world_model = world_model
        self.task = task
        self.log_dir = log_dir
        self.html_report = ApeHtmlReport(log_dir)

    def log_vars(self, vars_dict: dict):
        self.logger.info("-------------------- APE -----------------------")
        ignored_vars = {"task", "log_dir", "logger", "root", "nodes"}
        for var_name, var_value in vars_dict.items():
            if var_name not in ignored_vars:
                self.logger.info(f"{var_name} : {var_value}")
        self.logger.info("-------------------------------------------")

    def log_generation_query(self, query_idx: int, total: int, query: str, candidates: list[str]):
        self.logger.info(f"\n--- Generation query {query_idx}/{total} ---")
        self.logger.info(f"Query:\n{query}")
        self.logger.info(f"Candidates ({len(candidates)}):")
        for i, c in enumerate(candidates):
            self.logger.info(f"  [{i}] {c[:200]}{'...' if len(c) > 200 else ''}")

    def log_dedup_stats(self, total: int, unique: int):
        self.logger.info(
            f"\nDeduplication: {total} generated -> {unique} unique "
            f"({total - unique} duplicates removed)"
        )
        self.tracker.log({"phase": "deduplication", "num_generated": total, "num_unique": unique})

    def log_candidate_scores(self, scored_candidates: list[dict]):
        """Log ranked candidate table with per-example correctness."""
        self.logger.info("\n--------- Candidate Rankings ---------")
        for rank, entry in enumerate(scored_candidates):
            correct = entry.get("correct", [])
            correct_str = ""
            if correct:
                correct_str = (
                    f" | correct: {sum(1 for c in correct if c == 1)}/{len(correct)}"
                )
            self.logger.info(
                f"  #{rank + 1} | score: {entry['score']:.4f}{correct_str} | "
                f"{entry['prompt'][:150]}{'...' if len(entry['prompt']) > 150 else ''}"
            )
        self.logger.info("--------------------------------------")

    def log_best_candidate(self, prompt: str, eval_score: float, test_score=None):
        self.logger.info(f"\n========== Best Candidate ==========")
        self.logger.info(f"Eval score: {eval_score:.4f}")
        if test_score is not None:
            self.logger.info(f"Test score: {test_score}")
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"====================================")

    def _log_eval_details(self, label: str, prompt: str, eval_output: dict):
        """Log per-example eval details for the best candidate."""
        correct = eval_output.get("correct", [])
        preds = eval_output.get("preds", [])
        labels = eval_output.get("labels", [])
        questions = eval_output.get("model_inputs", [])

        self.logger.info(f"\n--- {label} per-example details ---")
        for i in range(len(correct)):
            marker = "✓" if correct[i] == 1 else "✗"
            pred_str = preds[i] if i < len(preds) else "?"
            label_str = labels[i] if i < len(labels) else "?"
            self.logger.info(
                f"  [{i}] {marker}  pred={pred_str}  label={label_str}"
            )

        correct_np = np.array(correct)
        valid = correct_np[correct_np > -1]
        self.logger.info(
            f"  Total: {int(valid.sum())}/{len(valid)} correct "
            f"({valid.mean():.4f})"
        )

    def prepare_output(
        self,
        scored_candidates: list[dict],
        best_node: MCTSNode,
        best_origin: dict,
        all_nodes: list[MCTSNode],
        generation_logs: list[dict],
        timing: dict = None,
    ) -> dict:
        """Prepare output dict, evaluate best on test set, track metrics."""
        console = get_console()
        timing = timing or {}

        best_ctx = {
            "phase": "best_candidate_eval",
            **best_origin,
        }

        # Detailed eval of best candidate (re-eval to get full output for logging)
        self.logger.info("\n--- Best candidate detailed evaluation ---")
        eval_metric, eval_output = self.world_model.eval_instruction_with_loader(
            task=self.task,
            eval_prompt=best_node.prompt,
            dataloader=self.world_model.eval_dataloader,
            tracker_context=best_ctx,
        )
        self._log_eval_details("Eval set", best_node.prompt, eval_output)

        # Test evaluation
        test_output = None
        if len(self.world_model.test_dataloader) != 0:
            console.phase("TEST", "evaluating best candidate on test set...")
            test_metric, test_output = self.world_model.test_prompt(
                best_node.prompt, origin=best_origin
            )
            best_node.test_metric = test_metric
            self._log_eval_details("Test set", best_node.prompt, test_output)
            self.log_best_candidate(
                best_node.prompt, best_node.reward, best_node.test_metric
            )
        else:
            self.log_best_candidate(best_node.prompt, best_node.reward)

        # Track summary metrics
        self.tracker.set_summary("test_accuracy", best_node.test_metric)
        self.tracker.set_summary("best_eval_score", best_node.reward)
        self.tracker.set_summary("num_candidates", len(scored_candidates))
        for key, value in timing.items():
            self.tracker.set_summary(key, value)

        # Track candidates table (with per-example correctness)
        columns = ["rank", "prompt", "eval_score", "num_correct", "num_total"]
        rows = []
        for i, entry in enumerate(scored_candidates):
            correct = entry.get("correct", [])
            num_correct = sum(1 for c in correct if c == 1) if correct else 0
            num_total = len(correct) if correct else 0
            rows.append([i + 1, entry["prompt"], entry["score"], num_correct, num_total])
        self.tracker.log_table("candidates", columns=columns, data=rows)

        # Track generation queries table
        gen_columns = ["query_idx", "demos", "num_candidates"]
        gen_rows = [
            [log["query_idx"], log["demos"], len(log["candidates"])]
            for log in generation_logs
        ]
        self.tracker.log_table("generation_queries", columns=gen_columns, data=gen_rows)

        # Console output
        console.selected_node_detail(
            node_id=best_node.id,
            depth=0,
            reward=best_node.reward,
            test_metric=getattr(best_node, "test_metric", None),
            prompt=best_node.prompt,
        )

        output = {
            "all_candidates": scored_candidates,
            "all_nodes": all_nodes,
            "best_candidate": best_node,
            "generation_logs": generation_logs,
            "eval_output": eval_output,
            "test_output": test_output,
            "timing": timing,
            # Compatibility with agent.run() which checks this key
            "best_reward_path_selected_node": [best_node],
        }
        return output

    def save_to_json(self, output: dict):
        """Save APE output to JSON with full analysis data."""
        # Build per-candidate details
        candidates_data = []
        for entry in output["all_candidates"]:
            c = {
                "prompt": entry["prompt"],
                "eval_score": entry["score"],
                "origin": entry.get("origin", {}),
            }
            if "correct" in entry:
                c["correct"] = entry["correct"]
                c["num_correct"] = sum(1 for x in entry["correct"] if x == 1)
                c["num_total"] = len(entry["correct"])
            candidates_data.append(c)

        # Build best candidate details
        best = output["best_candidate"]
        best_data = {
            "prompt": best.prompt,
            "eval_score": best.reward,
            "test_metric": best.test_metric,
        }

        # Eval output for best candidate
        eval_out = output.get("eval_output", {})
        best_eval_details = {}
        if eval_out:
            best_eval_details = {
                "preds": eval_out.get("preds", []),
                "labels": eval_out.get("labels", []),
                "correct": eval_out.get("correct", []),
            }

        # Test output for best candidate
        test_out = output.get("test_output", {})
        best_test_details = {}
        if test_out:
            best_test_details = {
                "preds": test_out.get("preds", []),
                "labels": test_out.get("labels", []),
                "correct": test_out.get("correct", []),
            }

        # Generation queries (structured)
        gen_queries = []
        for log in output.get("generation_logs", []):
            gen_queries.append({
                "query_idx": log["query_idx"],
                "demos": log["demos"],
                "query": log["query"],
                "candidates": log["candidates"],
            })

        data = {
            "candidates": candidates_data,
            "best_candidate": best_data,
            "best_candidate_eval_details": best_eval_details,
            "best_candidate_test_details": best_test_details,
            "generation_queries": gen_queries,
            "timing": output.get("timing", {}),
            "num_generated": sum(len(q["candidates"]) for q in gen_queries),
            "num_unique": len(candidates_data),
            "all_nodes": [n.to_dict() for n in output["all_nodes"]],
        }
        with open(os.path.join(self.log_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=4)

        # Finalize HTML report (marks as complete, writes final data)
        self.html_report.finalize(data)
