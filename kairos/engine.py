"""
EntropyEngine — public API for Kairos.

Runs three chaos engines in background threads, mixes their output into
a cryptographic entropy pool, and exposes token generation + health metrics.
No event loop dependency. Works in any Python context.
"""
import base64
import threading
import time
import uuid as _uuid_module

from .engines.double_pendulum import DoublePendulumEngine
from .engines.lorenz import LorenzEngine
from .engines.reaction_diffusion import ReactionDiffusionEngine
from .entropy.health import HealthMonitor
from .entropy.mixer import CryptoMixer
from .entropy.perturbation import PerturbationScheduler
from .entropy.pool import EntropyPool


class EntropyEngine:
    """
    Kairos entropy engine. Starts immediately on instantiation.
    Three chaos engines run in daemon threads and continuously feed
    a ring-buffer entropy pool.
    """

    def __init__(self):
        # Chaos engines (each starts its own daemon thread)
        self.pendulum = DoublePendulumEngine()
        self.lorenz = LorenzEngine()
        self.rd = ReactionDiffusionEngine()

        # Entropy subsystem
        self.pool = EntropyPool(size=1024)
        self.mixer = CryptoMixer()
        self.health_monitor = HealthMonitor(self.pool)
        self.perturber = PerturbationScheduler([self.pendulum, self.lorenz, self.rd])

        # Metrics
        self._tokens_generated = 0
        self._tokens_lock = threading.Lock()
        self._hashes_count = 0
        self._hashes_lock = threading.Lock()
        self._bytes_fed = 0
        self._bytes_lock = threading.Lock()
        self._start_time = time.time()

        # Feed loop
        self._running = True
        self._feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._feed_thread.start()

    def _feed_loop(self):
        """Every 50ms: collect bytes from all three engines, mix, feed to pool."""
        while self._running:
            mixed = self.mixer.mix(
                self.pendulum.get_entropy_bytes(),
                self.lorenz.get_entropy_bytes(),
                self.rd.get_entropy_bytes(),
            )
            self.pool.feed(mixed)
            with self._hashes_lock:
                self._hashes_count += 1
            with self._bytes_lock:
                self._bytes_fed += len(mixed)

            # Update ML state windows (lightweight: pendulum + lorenz only)
            try:
                self.health_monitor.ml_inference.update_state(
                    {
                        'pendulum': self.pendulum.get_state(),
                        'lorenz':   self.lorenz.get_state(),
                    },
                    mixed,
                )
            except Exception:
                pass  # never let ML code kill the feed loop

            time.sleep(0.05)

    def token(self, length: int = 32, fmt: str = "hex") -> str:
        """Generate a cryptographic token of the given byte length."""
        raw = self.mixer.derive_token(self.pool.read(64), length)
        with self._tokens_lock:
            self._tokens_generated += 1
        if fmt == "hex":
            return raw.hex()
        if fmt == "base64":
            return base64.b64encode(raw).decode()
        if fmt == "uuid":
            return str(_uuid_module.UUID(bytes=raw[:16]))
        return raw.hex()

    def api_key(self) -> str:
        """Generate a 'krs_' prefixed API key (40 hex chars)."""
        return "krs_" + self.token(20, "hex")

    def nonce(self) -> str:
        """Generate a single-use nonce (16 bytes hex)."""
        return self.token(16, "hex")

    def seed_bytes(self, n: int) -> bytes:
        """Return n raw bytes suitable for seeding other RNGs."""
        return self.mixer.derive_token(self.pool.read(64), n)

    def health(self) -> dict:
        """Return cached entropy health metrics."""
        return self.health_monitor.get_cached()

    def get_engine_states(self) -> dict:
        """Return current state snapshots from all three chaos engines."""
        return {
            "pendulum": self.pendulum.get_state(),
            "lorenz": self.lorenz.get_state(),
            "reaction_diffusion": self.rd.get_state(),
        }

    def tokens_generated(self) -> int:
        """Return total tokens generated since startup."""
        with self._tokens_lock:
            return self._tokens_generated

    def hashes_per_second(self) -> float:
        """Return approximate SHA3 mix cycles per second."""
        elapsed = time.time() - self._start_time
        if elapsed < 0.001:
            return 0.0
        with self._hashes_lock:
            return round(self._hashes_count / elapsed, 1)

    def entropy_rate_bps(self) -> float:
        """Return bytes fed to pool per second."""
        elapsed = time.time() - self._start_time
        if elapsed < 0.001:
            return 0.0
        with self._bytes_lock:
            return round(self._bytes_fed / elapsed, 1)

    def uptime_seconds(self) -> float:
        """Return seconds since engine started."""
        return round(time.time() - self._start_time, 1)

    def shutdown(self):
        """Gracefully stop all background threads."""
        self._running = False
        self.pendulum.stop()
        self.lorenz.stop()
        self.rd.stop()
        self.perturber.stop()
        self.health_monitor.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
