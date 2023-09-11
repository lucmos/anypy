from time import time
from typing import Dict


class ChronoRuntimeError(Exception):
    """Base exceptions for errors which happened inside Chronometer."""


class ChronoAlreadyStoppedError(ChronoRuntimeError):
    """Raised when trying to stop a stopped timer."""


class ChronoAlreadyStartedError(ChronoRuntimeError):
    """raised when trying to start a started timer."""


class Chrono:
    PREFIX: str = "| "
    START_PREFIX: str = "{prefix}Start {name}"
    STOP_PREFIX: str = "{prefix}Stop {name} [completed in {elapsed:.{precision}}]"
    ELAPSED_PREFIX: str = "{prefix}Lap {name} [completed in {elapsed:.{precision}}]"

    active_timers: Dict[str, "Chrono"] = {}

    def __init__(
        self,
        start: float,
        name: str,
        precision: int,
        _is_direct: bool = True,
    ):
        if _is_direct:
            raise ChronoRuntimeError("Do not initialize Chrono directly, use Chrono.start(name).")
        self.precision = precision
        self._start = start
        self.name = name

    def _current_prefix() -> str:
        return Chrono.PREFIX * len(Chrono.active_timers)

    def clock(self) -> float:
        return time() - self._start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def start(name: str, precision: int = 5):
        """Start the timer."""
        if name in Chrono.active_timers:
            raise ChronoAlreadyStartedError("Chrono already started.")
        current_prefixes = Chrono._current_prefix()
        Chrono.active_timers[name] = Chrono(time(), name, precision=precision, _is_direct=False)
        print(Chrono.START_PREFIX.format(prefix=current_prefixes, name=name))
        return Chrono.active_timers[name]

    def stop(self):
        """Stop the timer."""
        if self.name not in Chrono.active_timers:
            raise ChronoAlreadyStoppedError("Chrono already stopped.")
        del Chrono.active_timers[self.name]
        print(
            Chrono.STOP_PREFIX.format(
                prefix=Chrono._current_prefix(),
                name=self.name,
                elapsed=self.clock(),
                precision=self.precision,
            )
        )

    def elapsed(self) -> str:
        """Get the elapsed time."""
        if self.name not in Chrono.active_timers:
            raise ChronoAlreadyStoppedError("Chrono already stopped.")
        print(
            Chrono.ELAPSED_PREFIX.format(
                prefix=Chrono._current_prefix(),
                name=self.name,
                elapsed=self.clock(),
                precision=self.precision,
            )
        )
