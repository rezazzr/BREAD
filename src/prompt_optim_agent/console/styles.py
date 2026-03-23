"""Color palette and text helpers for console output."""

PHASE_COLORS: dict[str, str] = {
    "SELECT": "cyan",
    "EXPAND": "yellow",
    "FORWARD": "blue",
    "GRADIENT": "magenta",
    "OPTIMIZE": "green",
    "SIMULATE": "cyan bold",
    "BACKPROP": "yellow bold",
    "EVAL": "blue bold",
    "TEST": "red bold",
}


def truncate(text: str, max_len: int = 120) -> str:
    """Collapse whitespace and truncate with ellipsis."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3] + "..."


def reward_style(reward: float) -> str:
    """Return a Rich style string based on reward value."""
    if reward >= 0.8:
        return "bold green"
    if reward >= 0.5:
        return "bold yellow"
    return "bold red"
