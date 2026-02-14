# The MCTS algorithm code is adapted from Reasoning with Language Model is Planning with World Model
# https://github.com/Ber666/llm-reasoners

import logging
import os
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from ..tracking import MetricsTracker
from ..utils import create_logger
from ..world_model.world_model import WorldModel
from .base_algo import OptimNode, SearchAlgo
from .mcts_reporter import MCTSReporter
from .mcts_tree_node import MCTSNode


class MCTS(SearchAlgo):

    def __init__(
        self,
        task,
        world_model: WorldModel,
        # mcts arguments
        expand_width=3,
        w_exp: float = 2.5,
        depth_limit: int = 8,
        min_depth: int = 2,
        iteration_num: int = 12,
        # log
        log=True,
        logger: Optional[logging.Logger] = None,
        log_dir=None,
        tracker: MetricsTracker = None,
        **kwargs,
    ) -> None:
        """
        MCTS search algorithm

        :param task: the specific task
        :param world_model: the MCTS world model for state transition
        :param expand_width: number of batches to be sampled
        :param w_exp: the weight of mcts exploration
        :param depth_limit: the max depth of a single MCTS path
        :param iteration_num: number of MCTS iterations
        :param logger: logger
        :param log_dir: logger directory to save the results
        :param tracker: experiment metrics tracker
        """

        self.task = task
        self.world_model = world_model
        self.tracker = tracker or MetricsTracker()

        self.expand_width = expand_width
        self.depth_limit = depth_limit
        self.w_exp = w_exp
        self.iteration_num = iteration_num
        self.min_depth = (
            min_depth  # Apply early stop only when depth is larger than min_depth
        )

        self.mcts_threshold = 0.0  # The highest reward node globally
        self.min_threshold = 0.0  # The root node's reward as a min threshold

        # output
        self.log_dir = log_dir if log_dir is not None else os.getcwd()
        self.logger = (
            logger
            if logger is not None
            else create_logger(self.log_dir, "mcts", log_mode="train")
        )

        self.k = 1  # top-k reward nodes
        self.trace_in_each_iter: list[list[MCTSNode]] = []
        self.root: Optional[MCTSNode] = None
        self.nodes: list[MCTSNode] = []
        self.optim_nodes: list[OptimNode] = []
        self.optim_nodes_ids_only: list[int] = []
        self.base_optim_node_id = -1
        self.num_gradient_accumulation = kwargs.get("num_gradient_accumulation", 1)
        self.log = log

        self.reporter = MCTSReporter(
            logger=self.logger,
            tracker=self.tracker,
            world_model=self.world_model,
            task=self.task,
            uct_fn=self._uct,
            log_dir=self.log_dir,
        )
        self.reporter.log_vars(vars(self))

    def get_optim_id(self) -> int:
        self.base_optim_node_id -= 1
        return self.base_optim_node_id

    def simulate_choice(self, x):
        return np.argmax(x)

    def increase_threshold(self, threshold):
        if threshold > self.mcts_threshold:
            self.mcts_threshold = threshold

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.depth >= self.depth_limit

    def early_stop(self, node: MCTSNode):
        return node.reward > self.mcts_threshold and node.depth > self.min_depth

    def _is_terminal_with_min_threshold(self, node: MCTSNode):
        if node.parent is None:
            min_threshold = self.min_threshold
        else:
            min_threshold = (self.min_threshold + node.parent.reward) / 2
        return node.reward < min_threshold and node.depth > self.min_depth

    def is_terminal_node(self, node: MCTSNode):
        return (
            self._is_terminal_with_depth_limit(node)
            or self._is_terminal_with_min_threshold(node)
            or node.is_terminal
        )

    def _uct(self, node: MCTSNode) -> float:
        if node.parent is None:
            N_parent = 0
        else:
            N_parent = len(node.parent.cum_rewards)
        return_value = node.Q + self.w_exp * np.sqrt(
            np.log(N_parent + 1) / max(1, len(node.cum_rewards))
        )
        return return_value.item()

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        return max(node.children or [], key=self._uct)

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """
        Selection:
            From root node, keep selecting child node based on UCT
        """

        path = []
        while True:
            path.append(node)
            node.visited += 1
            if len(node.children) == 0 or self.is_terminal_node(node):
                return path

            node = self._uct_select(node)
            if self.log:
                self.logger.info(
                    f"Select node {node.id}: depth {node.depth}, "
                    f"reward: {node.reward:.4f} utc: {self._uct(node=node)}"
                )

    def _expand(self, node: MCTSNode):
        """
        Expansion:
            Sample batches of data and perform state transition on the given node.
            Generate new child nodes and calculate their temporary reward.
        """
        if self.log:
            self.logger.info("Expanding:")
        if self.is_terminal_node(node):
            node.is_terminal = True
            return

        if self.log:
            self.logger.info(
                f"Expanding: node: {node.id}, depth {node.depth}, reward: {node.reward:.4f}"
            )

        node.expand_times += 1
        if node.id not in self.optim_nodes_ids_only:
            self.optim_nodes.append(
                OptimNode(
                    node_id=node.id,
                    parent=node.parent.id if node.parent is not None else None,
                    children_id=[],
                    prompt=node.prompt,
                    gradient=None,
                    kind="node",
                )
            )
            self.optim_nodes_ids_only.append(node.id)

        for _ in range(self.expand_width):
            batch = self.world_model.get_train_batch()
            children, gradient_descent_output = self.world_model.step(node, batch)
            optim_node_id = self.get_optim_id()
            self.optim_nodes.append(
                OptimNode(
                    node_id=optim_node_id,
                    parent=node.id,
                    children_id=[child.id for child in children],
                    prompt=gradient_descent_output["gradient_prompt"],
                    gradient=gradient_descent_output["gradient"],
                )
            )
            self.optim_nodes.extend(
                [
                    OptimNode(
                        node_id=child.id,
                        parent=optim_node_id,
                        children_id=[],
                        prompt=child.prompt,
                        gradient=None,
                        kind="node",
                    )
                    for child in children
                ]
            )
            self.optim_nodes_ids_only.extend(
                [child.id for child in children] + [optim_node_id]
            )

            for child_node in children:
                self.world_model.evaluate_child_node(node=child_node)
                child_node.is_terminal = self.is_terminal_node(child_node)

            self.nodes.extend(children)
            node.children.extend(children)

        if self.log:
            for child in node.children:
                self.logger.info(
                    f"child_node {child.id} (reward:{child.reward:.4f})"
                )

    def _simulate(self, path: list[MCTSNode]):
        """
        Simulation: simulate the last node in the selected path, stop if reaching terminal or early stop.
        """

        if self.log:
            self.logger.info("Simulating:")
        node = path[-1]

        while True:
            if self.early_stop(node):
                node.is_terminal = self.is_terminal_node(node)
                self.increase_threshold(node.reward)
                if self.log:
                    self.logger.info(
                        f"Early Stop: node {node.id}, reward: {node.reward}. "
                        f"MCTS threshold increases to {self.mcts_threshold}. Stop simulating.\n"
                    )
                return

            self.increase_threshold(node.reward)

            if self.is_terminal_node(node):
                return

            if len(node.children) == 0:
                self._expand(node)

            rewards = [child.reward for child in node.children]
            if len(rewards) != 0:
                node = node.children[self.simulate_choice(rewards)]
            else:
                node.is_terminal = True

            node.visited += 1
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode]) -> List[float]:
        """
        Back Propagation: Update the cumulated rewards of each node in the path.
        """
        if self.log:
            self.logger.info("Back propagating:")

        cum_rewards = []
        running_sum = 0.0

        # Traverse from leaf to root
        for node in reversed(path):
            running_sum += node.reward
            cum_rewards.append(running_sum)
            node.cum_rewards.append(running_sum)
            if self.log:
                self.logger.info(
                    f"node {node.id}: depth {node.depth}, new cum_reward: {node.cum_rewards[-1]:.4f}"
                )

        # Reverse to match the original path order (root to leaf)
        cum_rewards.reverse()
        return cum_rewards

    def iterate(self, node: MCTSNode) -> Tuple[list[MCTSNode], List[float]]:
        """
        MCTS iteration: Selection, Expansion, Simulation, Back-Propagation
        """
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_rewards = self._back_propagate(path)

        return path, cum_rewards

    def search(self, init_state: str):

        self.root = self.world_model.build_root(init_state)
        self.nodes.append(self.root)

        if self.min_threshold == 0:  # TODO: Experiment with this condition
            self.min_threshold = self.root.reward
            self.increase_threshold(self.root.reward)

        self.trace_in_each_iter = []
        for i in range(self.iteration_num):
            self.tracker.log({"iteration": i})
            if self.log:
                self.logger.info(
                    f"---------------------  iteration {i} ------------------------"
                )

            path, cum_rewards = self.iterate(self.root)
            self.trace_in_each_iter.append(deepcopy(path))

        mcts_output = self.reporter.prepare_output(
            trace_in_each_iter=self.trace_in_each_iter,
            nodes=self.nodes,
            optim_nodes=self.optim_nodes,
            k=self.k,
        )
        self.reporter.save_to_json(mcts_output=mcts_output)
        return self.trace_in_each_iter, mcts_output

    def __call__(self, init_state: str, **kwargs):

        MCTSNode.reset_id()

        iteration_paths, mcts_outputs = self.search(init_state)

        return iteration_paths, mcts_outputs
