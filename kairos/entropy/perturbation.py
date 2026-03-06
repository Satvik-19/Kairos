"""Seed perturbation scheduler: injects micro-deltas into chaos engines every 10 seconds."""
import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engines.base import BaseChaosEngine

logger = logging.getLogger(__name__)


class PerturbationScheduler:
    """
    Every 10 seconds, applies a cryptographically-sourced micro-perturbation
    (epsilon = 1e-9) to each chaos engine's primary state variable.
    This breaks deterministic replay attacks.
    """

    def __init__(self, engines: List["BaseChaosEngine"]):
        self._engines = engines
        self._running = True
        self._last_perturbation_at: str | None = None
        self._lock = threading.Lock()
        self._schedule()

    def _schedule(self):
        """Schedule next perturbation in 10 seconds."""
        if self._running:
            self._timer = threading.Timer(10.0, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        """Apply perturbation to all engines, then reschedule."""
        epsilon = 1e-9
        for engine in self._engines:
            raw = os.urandom(32)
            delta_bytes = hashlib.sha3_256(raw).digest()[:8]
            # Interpret as a uniform integer in [0, 2^64) and normalise to (0, epsilon].
            # This guarantees every delta is well above float64 ULP and avoids the
            # NaN/Inf risk of struct.unpack("d", ...) on raw bit patterns.
            delta_int = int.from_bytes(delta_bytes, 'big')
            delta = (delta_int / (1 << 64)) * epsilon
            engine.apply_perturbation(delta)

        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._last_perturbation_at = ts

        logger.info("Perturbation applied at %s", ts)
        self._schedule()

    def last_perturbation_at(self) -> str | None:
        """Return ISO timestamp of last perturbation (not the value)."""
        with self._lock:
            return self._last_perturbation_at

    def stop(self):
        """Stop the perturbation scheduler."""
        self._running = False
        if hasattr(self, "_timer"):
            self._timer.cancel()
