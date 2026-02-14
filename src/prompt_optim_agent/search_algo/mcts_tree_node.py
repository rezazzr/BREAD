import itertools
from typing import List, Optional

import numpy as np


class MCTSNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        prompt: str,
        action: Optional[str],
        parent: "Optional[MCTSNode]" = None,
    ):
        """
        A node in the MCTS search tree

        :param prompt: the current state
        :param action: the action of the last optimization step,
            i.e., the state transition prompt from parent node to current node
        :param parent: the parent node, None if root of the tree
        """
        self.id = next(MCTSNode.id_iter)

        self.prompt = prompt
        self.action = action
        self.parent = parent
        self.is_terminal = False

        self.children: List[MCTSNode] = []
        self.cum_rewards: List[float] = []
        self.reward = 0.0
        self.test_metric = -1.0
        self.uct = 0.0

        self.visited = 0
        self.expand_times = 0

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return self.reward
        return np.mean(self.cum_rewards).item()

    def to_dict(self):
        return {
            "id": self.id,
            "depth": self.depth,
            "parent": -1 if self.parent is None else self.parent.id,
            "visited": self.visited,
            "expand_times": self.expand_times,
            "q": self.Q,
            "uct": self.uct,
            "prompt": self.prompt,
            "reward": self.reward,
            "test_metric": self.test_metric,
        }
