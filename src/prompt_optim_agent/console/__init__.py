"""Rich console output for prompt optimization."""

from .display import TRASConsole

_instance: TRASConsole | None = None


def init_console(enabled: bool = True) -> None:
    global _instance
    _instance = TRASConsole(enabled=enabled)


def get_console() -> TRASConsole:
    if _instance is None:
        init_console()
    return _instance
