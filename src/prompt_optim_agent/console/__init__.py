"""Rich console output for BREAD CLI."""

from .display import BreadConsole

_instance: BreadConsole | None = None


def init_console(enabled: bool = True) -> None:
    global _instance
    _instance = BreadConsole(enabled=enabled)


def get_console() -> BreadConsole:
    if _instance is None:
        init_console()
    return _instance
