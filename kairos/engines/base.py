"""Abstract base class for all chaos engines."""
import threading
import time
from abc import ABC, abstractmethod


class BaseChaosEngine(ABC):
    """Abstract base class for chaos engines. Each engine runs in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    @abstractmethod
    def tick(self):
        """Advance simulation one step."""

    @abstractmethod
    def get_state(self) -> dict:
        """Return current state as a dict."""

    @abstractmethod
    def get_entropy_bytes(self) -> bytes:
        """Return current state packed as bytes."""

    @abstractmethod
    def apply_perturbation(self, delta: float):
        """Apply a micro-delta to the primary state variable."""

    def _run_loop(self):
        """Thread target: calls tick() every 20ms."""
        while self._running:
            self.tick()
            time.sleep(0.02)

    def stop(self):
        """Stop the background thread."""
        self._running = False
