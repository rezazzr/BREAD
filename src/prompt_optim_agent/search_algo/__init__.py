from .ape import APE
from .mcts import MCTS
from .tras import TRAS

SEARCH_ALGOS = {"mcts": MCTS, "tras": TRAS, "ape": APE}


def get_search_algo(algo_name):
    if algo_name not in SEARCH_ALGOS:
        raise ValueError(
            f"Search algo '{algo_name}' is not supported. "
            f"Available: {list(SEARCH_ALGOS.keys())}"
        )
    return SEARCH_ALGOS[algo_name]


__all__ = ["get_search_algo", "MCTS", "TRAS", "APE"]
