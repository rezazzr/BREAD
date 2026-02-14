"""MCTS reporting and output formatting.

Handles logging, evaluation, metric tracking, and JSON serialization
of MCTS search results. Separated from the core MCTS algorithm to
keep the search logic focused.
"""

import json
import logging
import os
from typing import List

import numpy as np

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
        self, node: MCTSNode, eval=False, log_metric=False, eval_type="test"
    ):
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

        if eval:
            if eval_type == "test":
                test_metric, eval_output = self.world_model.test_prompt(node.prompt)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported.")
            node.test_metric = test_metric
        if log_metric:
            if not isinstance(node.test_metric, tuple):
                self.logger.info(f"   {eval_type} metric: {node.test_metric:.4f}")
            else:
                self.logger.info(f"   {eval_type} metric: {node.test_metric}")
        self.logger.info("---------------------")
        if eval:
            return eval_output["correct"]
        else:
            return None

    def log_vars(self, vars_dict: dict):
        self.logger.info("-------------------- MCTS -----------------------")
        ignored_vars = {"task", "log_dir", "logger", "trace_in_each_iter", "root", "nodes"}
        for var_name, var_value in vars_dict.items():
            if var_name not in ignored_vars:
                self.logger.info(f"{var_name} : {var_value}")
        self.logger.info("-------------------------------------------")

    def log_path(self, path, eval=False, log_metric=False):
        for node in path:
            self.eval_and_log_node(node=node, eval=eval, log_metric=log_metric)

    def log_nodes(self, nodes, eval=False, log_metric=False, eval_type="test"):
        for node in nodes:
            self.eval_and_log_node(
                node, eval=eval, log_metric=log_metric, eval_type=eval_type
            )
        self.logger.info("\n")

    def log_paths(self, paths, eval=False, log_metric=False, eval_type="test"):
        for i, path in enumerate(paths):
            self.logger.info(f"\n----------------  path {i} ------------------")
            for node in path:
                self.eval_and_log_node(
                    node, eval=eval, log_metric=log_metric, eval_type=eval_type
                )

    def prepare_output(
        self,
        trace_in_each_iter: list[list[MCTSNode]],
        nodes: list[MCTSNode],
        optim_nodes: list[OptimNode],
        k: int,
    ) -> dict:
        """Prepare output: evaluate test nodes, log paths, and track metrics."""
        self.logger.info(
            "\n---------------------  all iteration paths ------------------------"
        )
        self.log_paths(trace_in_each_iter)
        self.logger.info("\n---------------------  all nodes ------------------------")
        self.log_nodes(nodes)

        # Build path statistics
        paths_nodes = []
        paths_qs = []
        paths_rewards = []
        paths_ucts = []

        for i, path in enumerate(trace_in_each_iter):
            path_nodes = []
            path_ids = []
            path_qs = []
            path_rewards = []
            path_ucts = []

            for node_p in path:
                node = nodes[node_p.id]
                path_ids.append(node.id)
                uct = self.uct_fn(node)
                node.uct = uct
                path_ucts.append(uct)
                path_nodes.append(node)
                path_qs.append(node.Q)
                path_rewards.append(node.reward)

            paths_nodes.append(path_nodes)
            paths_qs.append(path_qs)
            paths_rewards.append(path_rewards)
            paths_ucts.append(path_ucts)

            self.logger.info(f"path {i}: {path_ids} ")
            self.logger.info(
                f"mean values:   path_uct: {np.mean(path_ucts):.4f} | "
                f"path_q: {np.mean(path_qs):.4f} | path_reward: {np.mean(path_rewards):.4f}"
            )
            self.logger.info(f"path_ucts:  {path_ucts}")
            self.logger.info(f"paths_qs :  {paths_qs}")
            self.logger.info(f"path_reward : {path_rewards}")
            self.logger.info("---------------------------")

        qs_rank = np.argsort([np.mean(row) for row in paths_qs])[::-1].tolist()
        rewards_rank = np.argsort([np.mean(row) for row in paths_rewards])[::-1].tolist()

        best_q_path = paths_nodes[qs_rank[0]]
        best_reward_path = paths_nodes[rewards_rank[0]]
        top_k_reward_nodes = sorted(
            nodes, key=lambda node: node.reward, reverse=True
        )[:k]

        if len(self.world_model.test_dataloader) != 0:
            self._evaluate_test_nodes(
                nodes, best_q_path, best_reward_path, top_k_reward_nodes
            )

        selected_node = sorted(
            best_reward_path,
            key=lambda node: self._sort_helper(node.reward),
            reverse=True,
        )[0]
        last_node_of_best_reward_path = best_reward_path[-1]

        self._track_metrics(
            trace_in_each_iter, nodes, optim_nodes,
            best_q_path, best_reward_path, selected_node,
            last_node_of_best_reward_path,
        )

        return dict(
            all_paths=paths_nodes,
            all_nodes=nodes,
            best_q_path=best_q_path,
            best_reward_path=best_reward_path,
            top_k_reward_nodes=top_k_reward_nodes,
            best_reward_path_last_node=[last_node_of_best_reward_path],
            best_reward_path_selected_node=[selected_node],
        )

    def _evaluate_test_nodes(
        self, nodes, best_q_path, best_reward_path, top_k_reward_nodes
    ):
        """Evaluate selected nodes on test set and log detailed metrics."""
        self.logger.info("\n----------------  test nodes ------------------")
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
                        node, eval=True, log_metric=True, eval_type="test"
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

        self.logger.info("\n----------------  top_k_reward_nodes------------------")
        for node in top_k_reward_nodes:
            self.eval_and_log_node(node, eval=False, log_metric=True, eval_type="test")

        self.logger.info("\n----------------  best_reward_path------------------")
        for node in best_reward_path:
            self.eval_and_log_node(node, eval=False, log_metric=True, eval_type="test")

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

        # Path data table
        path_data_as_list = []
        path_data_columns = ["path_id"] + list(nodes[0].to_dict().keys())
        for i, path in enumerate(trace_in_each_iter):
            for node in path:
                node_dict = node.to_dict()
                if node_dict["parent"] == -1:
                    node_dict["parent"] = None
                path_data_as_list.append([i] + list(node_dict.values()))
        self.tracker.log_table("paths", columns=path_data_columns, data=path_data_as_list)

        # Nodes table with best-path annotations
        data_nodes_as_list = []
        data_nodes_columns = [
            key if key != "id" else "node_id" for key in nodes[0].to_dict().keys()
        ]
        data_nodes_columns.extend(["best_q_path", "best_reward_path", "selected_node"])
        best_reward_path_ids = {node.id for node in best_reward_path}
        best_q_path_ids = {node.id for node in best_q_path}
        for node in nodes:
            node_dict = node.to_dict()
            if node_dict["parent"] == -1:
                node_dict["parent"] = None
            data_nodes_as_list.append(
                list(node_dict.values())
                + [
                    node.id in best_q_path_ids,
                    node.id in best_reward_path_ids,
                    node.id == selected_node.id,
                ]
            )
        node_table = self.tracker.create_table(
            columns=data_nodes_columns, data=data_nodes_as_list
        )
        tree_fields = {"node-id": "node_id", "node-parent": "parent"}
        self.tracker.log_plot_table(
            "nodes",
            vega_spec_name="rezazzr/tree_visualizer",
            data_table=node_table,
            fields=tree_fields,
        )

        # Optim nodes tree
        data_optim_nodes_as_list = []
        data_optim_nodes_columns = list(optim_nodes[0].to_dict().keys())
        for optim_node in optim_nodes:
            node_dict = optim_node.to_dict()
            if node_dict["parent"] == -1:
                node_dict["parent"] = None
            data_optim_nodes_as_list.append(list(node_dict.values()))

        optim_node_table = self.tracker.create_table(
            columns=data_optim_nodes_columns, data=data_optim_nodes_as_list
        )
        self.tracker.log_plot_table(
            "optim_nodes",
            vega_spec_name="tree_optim_visualizer",
            data_table=optim_node_table,
            fields={"node-id": "node_id", "node-parent": "parent"},
        )

    def save_to_json(self, mcts_output: dict):
        """Save MCTS output to a JSON file."""
        data_to_save = {}
        paths = []
        for path in mcts_output["all_paths"]:
            paths.append([node.to_dict() for node in path])
        data_to_save["all_paths"] = paths

        for key in mcts_output:
            if key != "all_paths":
                data_to_save[key] = [node.to_dict() for node in mcts_output[key]]
        with open(os.path.join(self.log_dir, "data.json"), "w") as f:
            json.dump(data_to_save, f, indent=4)

    @staticmethod
    def _sort_helper(metric):
        if isinstance(metric, tuple):
            return metric[0]
        return metric

    @staticmethod
    def _record_counts(array: np.ndarray) -> List[int]:
        counts_list = []
        for col in range(array.shape[1]):
            counts = {1: 0, 0: 0, -1: 0}
            unique, counts_array = np.unique(array[:, col], return_counts=True)
            counts.update(dict(zip(unique, counts_array)))
            counts_list.extend([counts[1], counts[0], counts[-1]])
        return counts_list
