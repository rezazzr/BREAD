from .mcts import MCTS

SEARCH_ALGOS = {"mcts": MCTS}


def get_search_algo(algo_name):
    if algo_name not in SEARCH_ALGOS:
        raise ValueError(
            f"Search algo '{algo_name}' is not supported. "
            f"Available: {list(SEARCH_ALGOS.keys())}"
        )
    return SEARCH_ALGOS[algo_name]


__all__ = ["get_search_algo", "MCTS"]
