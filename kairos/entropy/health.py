"""Entropy health monitor: Shannon entropy, uniformity, duplicate rate.
After training, delegates to KairosMLInference for enhanced scoring.
"""
import math
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pool import EntropyPool

# Optional ML enhancement — gracefully absent before training
try:
    from ml.inference import KairosMLInference as _KairosMLInference
except Exception:
    _KairosMLInference = None


class HealthMonitor:
    """
    Evaluates the statistical quality of the entropy pool every 5 seconds.
    Results are cached and accessible via get_cached().
    When ML models are trained, the cached result is augmented with:
      ml_active, health_confidence, anomaly_score, is_anomaly,
      prediction_resistance  (see ml/inference.py).
    """

    def __init__(self, pool: "EntropyPool"):
        self._pool = pool
        self._cached: dict = {
            "entropy_score":           0.0,
            "distribution_uniformity": 0.0,
            "duplicate_rate":          0.0,
            "health_status":           "degraded",
            "last_evaluated":          None,
        }
        self._lock = threading.Lock()

        # ML inference helper — instantiated once; silently no-ops if untrained
        if _KairosMLInference is not None:
            self.ml_inference = _KairosMLInference()
        else:
            self.ml_inference = _RuleBasedFallback()

        self._running = True
        self._schedule()

    def _schedule(self):
        """Schedule next evaluation in 5 seconds."""
        if self._running:
            self._timer = threading.Timer(5.0, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        """Compute metrics, enrich with ML fields, cache result."""
        pool_bytes = self._pool.read_all()
        result     = self.evaluate(pool_bytes)

        # Add pool fill so ML classifier can use it
        result['pool_fill_percent'] = round(self._pool.fill_percent(), 2)

        # Enrich with ML-derived fields (no-op if models not trained)
        try:
            ml_result = self.ml_inference.evaluate(result.copy())
            result.update(ml_result)
        except Exception:
            pass  # never let ML code stop the health timer

        with self._lock:
            self._cached = result
        self._schedule()

    def evaluate(self, pool_bytes: bytes) -> dict:
        """
        Compute entropy health metrics from raw bytes.
        Returns dict with entropy_score, distribution_uniformity,
        duplicate_rate, and health_status.
        """
        n = len(pool_bytes)
        if n == 0:
            return {
                "entropy_score":           0.0,
                "distribution_uniformity": 0.0,
                "duplicate_rate":          0.0,
                "health_status":           "critical",
                "last_evaluated":          datetime.now(timezone.utc).isoformat(),
            }

        # --- Shannon entropy ---
        counts = [0] * 256
        for b in pool_bytes:
            counts[b] += 1

        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / n
                entropy -= p * math.log2(p)
        entropy_score = entropy / 8.0   # normalise to [0, 1]

        # --- Byte distribution uniformity (chi-squared) ---
        expected    = n / 256.0
        chi_sq      = sum((c - expected) ** 2 / expected for c in counts)
        chi_sq_max  = ((n - expected) ** 2 / expected) + 255 * (expected ** 2 / expected)
        uniformity  = max(0.0, 1.0 - (chi_sq / chi_sq_max)) if chi_sq_max > 0 else 1.0

        # --- Duplicate rate (repeated 32-byte chunks) ---
        chunk_size   = 32
        chunks       = [pool_bytes[i: i + chunk_size]
                        for i in range(0, n - chunk_size + 1, chunk_size)]
        total_chunks = len(chunks)
        if total_chunks > 1:
            seen   = set()
            dupes  = 0
            for chunk in chunks:
                key = bytes(chunk)
                if key in seen:
                    dupes += 1
                seen.add(key)
            duplicate_rate = dupes / total_chunks
        else:
            duplicate_rate = 0.0

        # --- Health status (rule-based; overridden by ML when active) ---
        if entropy_score >= 0.99:
            health_status = "excellent"
        elif entropy_score >= 0.95:
            health_status = "good"
        elif entropy_score >= 0.90:
            health_status = "degraded"
        else:
            health_status = "critical"

        return {
            "entropy_score":           round(entropy_score, 6),
            "distribution_uniformity": round(uniformity,    6),
            "duplicate_rate":          round(duplicate_rate, 8),
            "health_status":           health_status,
            "last_evaluated":          datetime.now(timezone.utc).isoformat(),
        }

    def get_cached(self) -> dict:
        """Return the last computed health result."""
        with self._lock:
            return dict(self._cached)

    def stop(self):
        """Stop the background evaluation timer."""
        self._running = False
        if hasattr(self, "_timer"):
            self._timer.cancel()


class _RuleBasedFallback:
    """
    Minimal stand-in when ml.inference cannot be imported.
    Implements the same .evaluate() and .update_state() interface.
    """

    def update_state(self, chaos_states: dict, hash_bytes: bytes):
        pass

    def evaluate(self, metrics: dict) -> dict:
        metrics['ml_active'] = False
        return metrics
