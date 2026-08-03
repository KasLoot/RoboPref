"""Process-wide coordination for monitor-requested emergency shutdown.

The vision monitor can request an emergency stop, but its worker thread must
never terminate the process itself.  :class:`EmergencyStopCoordinator` latches
the first request, invokes the configured stop hook exactly once, and exposes a
``threading.Event`` that the main runtime can wait on.

``emergency_stop`` is intentionally only a placeholder.  It must be replaced
with an acknowledged, robot-specific stop API before this prototype is used on
physical hardware.  A vision-language model is not a safety-rated emergency
stop; an independent hardware/operator E-stop remains necessary.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Callable


LOGGER = logging.getLogger(__name__)


def emergency_stop(reason: str) -> None:
    """Placeholder for the robot's acknowledged emergency-stop command.

    The coordinator guarantees that this function is called at most once for a
    process.  This placeholder deliberately performs no hardware I/O.
    """

    LOGGER.critical("EMERGENCY STOP requested: %s", reason)


@dataclass(frozen=True, slots=True)
class EmergencyStopEvent:
    """Trusted envelope around the first accepted emergency-stop request."""

    reason: str
    publication_id: str | None
    observed_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("emergency-stop reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())
        if self.publication_id is not None:
            if (
                not isinstance(self.publication_id, str)
                or not self.publication_id.strip()
            ):
                raise ValueError(
                    "publication_id must be a non-empty string or None"
                )
            object.__setattr__(
                self,
                "publication_id",
                self.publication_id.strip(),
            )
        if isinstance(self.observed_at, bool) or not isinstance(
            self.observed_at,
            (int, float),
        ):
            raise ValueError("observed_at must be a finite number")
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be a finite number")
        object.__setattr__(self, "observed_at", observed_at)


class EmergencyStopCoordinator:
    """Latch and coordinate a process shutdown without calling ``sys.exit``.

    ``trigger`` is safe to call concurrently.  Exactly one caller wins the
    latch and invokes ``stop_function``; every later call returns ``False``.
    An internal latch is set before external code runs so publishers immediately
    stop admitting work.  The public shutdown event is set after the stop hook
    and best-effort state notification return, giving the main runtime an
    orderly handoff.  Exceptions from either callback are retained for
    diagnostics but do not undo the emergency latch.
    """

    def __init__(
        self,
        stop_function: Callable[[str], None] = emergency_stop,
        *,
        on_stop: Callable[[EmergencyStopEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(stop_function):
            raise TypeError("stop_function must be callable")
        if on_stop is not None and not callable(on_stop):
            raise TypeError("on_stop must be callable or None")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._stop_function = stop_function
        self._on_stop = on_stop
        self._clock = clock
        self._lock = threading.Lock()
        self._latch_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._event: EmergencyStopEvent | None = None
        self._stop_error: BaseException | None = None
        self._callback_error: BaseException | None = None

    @property
    def shutdown_event(self) -> threading.Event:
        """Event watched by the main runtime to initiate orderly shutdown."""

        return self._shutdown_event

    @property
    def event(self) -> EmergencyStopEvent | None:
        with self._lock:
            return self._event

    @property
    def latched(self) -> bool:
        return self._latch_event.is_set()

    @property
    def stop_error(self) -> BaseException | None:
        with self._lock:
            return self._stop_error

    @property
    def callback_error(self) -> BaseException | None:
        with self._lock:
            return self._callback_error

    def trigger(
        self,
        reason: str,
        *,
        publication_id: str | None = None,
        observed_at: float | None = None,
    ) -> bool:
        """Latch the first request and invoke the stop integration exactly once.

        Returns ``True`` only to the caller that acquired the latch.  The
        configured actions run in the caller's thread; no process-exit function
        is used here.
        """

        if observed_at is None:
            observed_at = self._clock()
        event = EmergencyStopEvent(
            reason=reason,
            publication_id=publication_id,
            observed_at=observed_at,
        )
        with self._lock:
            if self._event is not None:
                return False
            self._event = event
            # Latch before any external callback.  This prevents a callback
            # failure or a concurrent publisher from admitting more work.
            self._latch_event.set()

        try:
            self._stop_function(event.reason)
        except BaseException as error:  # stopping must remain latched
            with self._lock:
                self._stop_error = error
            LOGGER.exception("The emergency-stop integration raised an error")

        if self._on_stop is not None:
            try:
                self._on_stop(event)
            except BaseException as error:  # shutdown must still proceed
                with self._lock:
                    self._callback_error = error
                LOGGER.exception("The emergency-stop notification raised an error")
        # Signal the main runtime only after the stop hook and best-effort state
        # notification have returned.  The independent latch above already
        # prevents any new publication while these actions run.
        self._shutdown_event.set()
        return True
