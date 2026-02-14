from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimNode:
    """
    A data class representing a node in an optimization graph.

    Attributes:
        node_id (float): Unique identifier for the node.
        parent (int): Identifier of the parent node.
        prompt (str): The prompt or description associated with the node.
        children_id (List[int]): List of identifiers for the children nodes.
        gradient (str): The gradient/feedback or direction for optimization.
        kind (str): Type of node - "optim" for optimization step, "node" for MCTS node.
    """

    node_id: float
    parent: Optional[int]
    prompt: str
    children_id: List[int] = field(default_factory=list)
    gradient: Optional[str] = None
    kind: str = "optim"

    def __str__(self) -> str:
        return (
            f"node_id={self.node_id}\nparent={self.parent}\n"
            f"children_id={self.children_id}\nprompt={self.prompt}\n"
            f"gradient={self.gradient}\nkind={self.kind}"
        )

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent": self.parent,
            "prompt": self.prompt,
            "gradient": self.gradient,
            "kind": self.kind,
        }


class SearchAlgo(ABC):
    def __init__(self, task, world_model, logger=None) -> None:
        self.task = task
        self.world_model = world_model
        self.logger = logger

    @abstractmethod
    def search(self):
        pass
