"""MCTS reporting and output formatting.

Handles logging, evaluation, metric tracking, and JSON serialization
of MCTS search results. Separated from the core MCTS algorithm to
keep the search logic focused.
"""

import json
import logging
import os

import numpy as np

from ..console import get_console
from ..tracking import MetricsTracker
from .base_algo import OptimNode
from .mcts_tree_node import MCTSNode


class MCTSReporter:
    def __init__(
        self,
        logger: logging.Logger,
        tracker: MetricsTracker,
        world_model,
        task,
        uct_fn,
        log_dir: str,
    ):
        """
        :param logger: the logger instance
        :param tracker: experiment metrics tracker
        :param world_model: world model (used for test_prompt)
        :param task: the task (used for test_metrics_definition)
        :param uct_fn: callable that computes UCT for a node
        :param log_dir: directory for JSON output
        """
        self.logger = logger
        self.tracker = tracker
        self.world_model = world_model
        self.task = task
        self.uct_fn = uct_fn
        self.log_dir = log_dir

    def eval_and_log_node(
        self, node: MCTSNode, evaluate=False, log_metric=False, eval_type="test"
    ):
        console = get_console()
        parent_info = (
            f"parent: {node.parent.id}" if node.parent is not None else "parent: N/A"
        )
        self.logger.info(
            f"node {node.id}:    {parent_info} | depth: {node.depth} | visited: {node.visited} "
            f"| expand_times: {node.expand_times}  | terminal: {node.is_terminal} | children: {len(node.children)}"
        )
        self.logger.info(
            f"   reward: {node.reward:.4f} | Q: {node.Q:.4f} | uct: {self.uct_fn(node):.4f} "
            f"| cum_rewards: {node.cum_rewards}"
        )
        self.logger.info(f"   prompt: {node.prompt}")

        if evaluate:
            if eval_type == "test":
                test_metric, eval_output = self.world_model.test_prompt(node.prompt)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported.")
            node.test_metric = test_metric
        if log_metric:
            if isinstance(node.test_metric, tuple):
                self.logger.info(f"   {eval_type} metric: {node.test_metric}")
            else:
                self.logger.info(f"   {eval_type} metric: {node.test_metric:.4f}")
            console.node_info(
                node_id=node.id,
                depth=node.depth,
                reward=node.reward,
                q=node.Q,
                uct=self.uct_fn(node),
                test_metric=node.test_metric,
            )
        self.logger.info("---------------------")
        if evaluate:
            return eval_output["correct"]
        return None

    def log_vars(self, vars_dict: dict):
        self.logger.info("-------------------- MCTS -----------------------")
        ignored_vars = {"task", "log_dir", "logger", "trace_in_each_iter", "root", "nodes"}
        for var_name, var_value in vars_dict.items():
            if var_name not in ignored_vars:
                self.logger.info(f"{var_name} : {var_value}")
        self.logger.info("-------------------------------------------")

    def log_nodes(self, nodes, evaluate=False, log_metric=False, eval_type="test"):
        for node in nodes:
            self.eval_and_log_node(
                node, evaluate=evaluate, log_metric=log_metric, eval_type=eval_type
            )
        self.logger.info("\n")

    def log_paths(self, paths, evaluate=False, log_metric=False, eval_type="test"):
        for i, path in enumerate(paths):
            self.logger.info(f"\n----------------  path {i} ------------------")
            for node in path:
                self.eval_and_log_node(
                    node, evaluate=evaluate, log_metric=log_metric, eval_type=eval_type
                )

    def prepare_output(
        self,
        trace_in_each_iter: list[list[MCTSNode]],
        nodes: list[MCTSNode],
        optim_nodes: list[OptimNode],
        k: int,
    ) -> dict:
        """Prepare output: evaluate test nodes, log paths, and track metrics."""
        console = get_console()

        self.logger.info(
            "\n---------------------  all iteration paths ------------------------"
        )
        self.log_paths(trace_in_each_iter)
        self.logger.info("\n---------------------  all nodes ------------------------")
        self.log_nodes(nodes)

        paths_nodes, paths_qs, paths_rewards = self._gather_path_stats(
            trace_in_each_iter, nodes, console
        )

        best_q_path = paths_nodes[
            np.argmax([np.mean(row) for row in paths_qs])
        ]
        best_reward_path = paths_nodes[
            np.argmax([np.mean(row) for row in paths_rewards])
        ]
        top_k_reward_nodes = sorted(
            nodes, key=lambda n: n.reward, reverse=True
        )[:k]

        if len(self.world_model.test_dataloader) != 0:
            console.phase("TEST", "evaluating best nodes on test set...")
            self._evaluate_test_nodes(
                nodes, best_q_path, best_reward_path, top_k_reward_nodes
            )

        selected_node = max(
            best_reward_path, key=lambda n: self._sort_helper(n.reward)
        )
        last_node_of_best_reward_path = best_reward_path[-1]

        self._track_metrics(
            trace_in_each_iter, nodes, optim_nodes,
            best_q_path, best_reward_path, selected_node,
            last_node_of_best_reward_path,
        )

        # Console: detailed selected node summary
        console.selected_node_detail(
            node_id=selected_node.id,
            depth=selected_node.depth,
            reward=selected_node.reward,
            test_metric=getattr(selected_node, "test_metric", None),
            prompt=selected_node.prompt,
        )
        if last_node_of_best_reward_path.id != selected_node.id:
            console.test_result(
                "last node of best path test",
                last_node_of_best_reward_path.test_metric,
            )

        self._generate_plots(console, nodes, paths_nodes, selected_node.id)

        return dict(
            all_paths=paths_nodes,
            all_nodes=nodes,
            best_q_path=best_q_path,
            best_reward_path=best_reward_path,
            top_k_reward_nodes=top_k_reward_nodes,
            best_reward_path_last_node=[last_node_of_best_reward_path],
            best_reward_path_selected_node=[selected_node],
        )

    def _gather_path_stats(self, trace_in_each_iter, nodes, console):
        """Resolve traced paths against final node state and log per-path statistics."""
        paths_nodes = []
        paths_qs = []
        paths_rewards = []

        for i, path in enumerate(trace_in_each_iter):
            path_nodes = []
            path_ids = []
            path_qs = []
            path_rewards = []

            for node_p in path:
                node = nodes[node_p.id]
                node.uct = self.uct_fn(node)
                path_ids.append(node.id)
                path_nodes.append(node)
                path_qs.append(node.Q)
                path_rewards.append(node.reward)

            paths_nodes.append(path_nodes)
            paths_qs.append(path_qs)
            paths_rewards.append(path_rewards)

            path_ucts = [n.uct for n in path_nodes]
            self.logger.info(f"path {i}: {path_ids} ")
            self.logger.info(
                f"mean values:   path_uct: {np.mean(path_ucts):.4f} | "
                f"path_q: {np.mean(path_qs):.4f} | path_reward: {np.mean(path_rewards):.4f}"
            )
            self.logger.info(f"path_ucts:  {path_ucts}")
            self.logger.info(f"paths_qs :  {paths_qs}")
            self.logger.info(f"path_reward : {path_rewards}")
            self.logger.info("---------------------------")

            console.path_table(i, path_ids, float(np.mean(path_rewards)), float(np.mean(path_qs)))

        return paths_nodes, paths_qs, paths_rewards

    def _generate_plots(self, console, nodes, paths_nodes, selected_node_id: int = -1):
        """Build plot data dicts and send to the console for final visualization."""
        node_dicts = [
            {
                "id": n.id,
                "parent": n.parent.id if n.parent is not None else -1,
                "depth": n.depth,
                "reward": n.reward,
                "q": n.Q,
                "uct": getattr(n, "uct", 0),
                "test_metric": getattr(n, "test_metric", None),
            }
            for n in nodes
        ]
        path_dicts = [
            [{"id": n.id, "depth": n.depth, "reward": n.reward} for n in path]
            for path in paths_nodes
        ]
        console.generate_final_plots(node_dicts, path_dicts, self.log_dir, selected_node_id=selected_node_id)

    def _evaluate_test_nodes(
        self, nodes, best_q_path, best_reward_path, top_k_reward_nodes
    ):
        """Evaluate selected nodes on test set and log detailed metrics."""
        self.logger.info("\n----------------  test_nodes ------------------")
        test_nodes_set = set(best_q_path + best_reward_path + top_k_reward_nodes)
        detailed_metrics_columns = []
        detailed_metrics_values = []
        if hasattr(self.task, "test_metrics_definition"):
            detailed_metrics_columns = ["node_id"] + [
                f"{metric['metric_name']}_{suffix}"
                for metric in self.task.test_metrics_definition
                for suffix in ["YES", "NO", "NA"]
            ]
        for node in nodes:
            if node in test_nodes_set:
                correct_results = np.array(
                    self.eval_and_log_node(
                        node, evaluate=True, log_metric=True, eval_type="test"
                    )
                )
                if len(correct_results.shape) == 2:
                    list_of_counts = self._record_counts(correct_results)
                    detailed_metrics_values.append([node.id] + list_of_counts)

        if len(detailed_metrics_values) > 0:
            self.tracker.log_table(
                "detailed_metrics",
                columns=detailed_metrics_columns,
                data=detailed_metrics_values,
            )

        self.logger.info("\n----------------  top_k_reward_nodes ------------------")
        for node in top_k_reward_nodes:
            self.eval_and_log_node(node, log_metric=True, eval_type="test")

        self.logger.info("\n----------------  best_reward_path ------------------")
        for node in best_reward_path:
            self.eval_and_log_node(node, log_metric=True, eval_type="test")

    def _track_metrics(
        self, trace_in_each_iter, nodes, optim_nodes,
        best_q_path, best_reward_path, selected_node,
        last_node_of_best_reward_path,
    ):
        """Send summary metrics and tables to the tracker."""
        self.tracker.set_summary("test_accuracy", selected_node.test_metric)
        self.tracker.set_summary(
            "last_node_test_accuracy", last_node_of_best_reward_path.test_metric
        )

        self._track_paths_table(trace_in_each_iter, nodes)
        self._track_nodes_table(nodes, best_q_path, best_reward_path, selected_node)
        self._track_optim_nodes_table(optim_nodes)

    def _track_paths_table(self, trace_in_each_iter, nodes):
        """Log the per-path node data table to the tracker."""
        columns = ["path_id"] + list(nodes[0].to_dict().keys())
        rows = []
        for i, path in enumerate(trace_in_each_iter):
            for node in path:
                row = self._node_dict_to_row(node)
                rows.append([i] + row)
        self.tracker.log_table("paths", columns=columns, data=rows)

    def _track_nodes_table(self, nodes, best_q_path, best_reward_path, selected_node):
        """Log the annotated nodes table and tree visualization to the tracker."""
        columns = [
            "node_id" if key == "id" else key for key in nodes[0].to_dict().keys()
        ]
        columns.extend(["best_q_path", "best_reward_path", "selected_node"])

        best_q_ids = {n.id for n in best_q_path}
        best_reward_ids = {n.id for n in best_reward_path}
        rows = []
        for node in nodes:
            row = self._node_dict_to_row(node)
            row.extend([
                node.id in best_q_ids,
                node.id in best_reward_ids,
                node.id == selected_node.id,
            ])
            rows.append(row)

        node_table = self.tracker.create_table(columns=columns, data=rows)
        self.tracker.log_plot_table(
            "nodes",
            vega_spec_name="rezazzr/tree_visualizer",
            data_table=node_table,
            fields={"node-id": "node_id", "node-parent": "parent"},
        )

    def _track_optim_nodes_table(self, optim_nodes):
        """Log the optimization nodes tree visualization to the tracker."""
        columns = list(optim_nodes[0].to_dict().keys())
        rows = []
        for optim_node in optim_nodes:
            node_dict = optim_node.to_dict()
            if node_dict["parent"] == -1:
                node_dict["parent"] = None
            rows.append(list(node_dict.values()))

        optim_table = self.tracker.create_table(columns=columns, data=rows)
        self.tracker.log_plot_table(
            "optim_nodes",
            vega_spec_name="tree_optim_visualizer",
            data_table=optim_table,
            fields={"node-id": "node_id", "node-parent": "parent"},
        )

    @staticmethod
    def _node_dict_to_row(node: MCTSNode) -> list:
        """Convert an MCTSNode to a tracker-ready row, normalizing parent=-1 to None."""
        node_dict = node.to_dict()
        if node_dict["parent"] == -1:
            node_dict["parent"] = None
        return list(node_dict.values())

    def save_to_json(self, mcts_output: dict):
        """Save MCTS output to a JSON file."""
        data_to_save = {}
        for key, value in mcts_output.items():
            if key == "all_paths":
                data_to_save[key] = [
                    [node.to_dict() for node in path] for path in value
                ]
            else:
                data_to_save[key] = [node.to_dict() for node in value]
        with open(os.path.join(self.log_dir, "data.json"), "w") as f:
            json.dump(data_to_save, f, indent=4)

    @staticmethod
    def _sort_helper(metric):
        if isinstance(metric, tuple):
            return metric[0]
        return metric

    @staticmethod
    def _record_counts(array: np.ndarray) -> list[int]:
        counts_list = []
        for col in range(array.shape[1]):
            counts = {1: 0, 0: 0, -1: 0}
            unique, counts_array = np.unique(array[:, col], return_counts=True)
            counts.update(dict(zip(unique, counts_array)))
            counts_list.extend([counts[1], counts[0], counts[-1]])
        return counts_list
