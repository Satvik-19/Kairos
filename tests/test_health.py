"""Tests for HealthMonitor."""
import os
import time

from kairos.entropy.health import HealthMonitor
from kairos.entropy.pool import EntropyPool


class TestHealthMonitor:
    def test_evaluate_returns_required_keys(self):
        pool = EntropyPool()
        monitor = HealthMonitor(pool)
        result = monitor.evaluate(os.urandom(1024))
        monitor.stop()
        for key in ("entropy_score", "distribution_uniformity", "duplicate_rate", "health_status"):
            assert key in result

    def test_uniform_distribution_high_score(self):
        """Uniform random bytes should produce entropy_score near 1.0."""
        pool = EntropyPool()
        monitor = HealthMonitor(pool)
        data = os.urandom(1024)
        result = monitor.evaluate(data)
        monitor.stop()
        assert result["entropy_score"] >= 0.95, f"Score too low: {result['entropy_score']}"

    def test_degenerate_zeros_critical(self):
        """All-zero bytes should produce a critical health status."""
        pool = EntropyPool()
        monitor = HealthMonitor(pool)
        data = b"\x00" * 1024
        result = monitor.evaluate(data)
        monitor.stop()
        assert result["entropy_score"] < 0.1
        assert result["health_status"] == "critical"

    def test_health_status_thresholds(self):
        """Test that status labels match score thresholds."""
        pool = EntropyPool()
        monitor = HealthMonitor(pool)

        # Manually craft near-uniform data for excellent status
        data = os.urandom(4096)
        result = monitor.evaluate(data)
        monitor.stop()
        score = result["entropy_score"]
        if score >= 0.99:
            assert result["health_status"] == "excellent"
        elif score >= 0.95:
            assert result["health_status"] == "good"
        elif score >= 0.90:
            assert result["health_status"] == "degraded"
        else:
            assert result["health_status"] == "critical"

    def test_get_cached_returns_dict(self):
        pool = EntropyPool()
        pool.feed(os.urandom(1024))
        monitor = HealthMonitor(pool)
        time.sleep(0.1)
        cached = monitor.get_cached()
        monitor.stop()
        assert isinstance(cached, dict)
        assert "health_status" in cached

    def test_duplicate_rate_zero_for_random(self):
        """Random data should have near-zero duplicate rate."""
        pool = EntropyPool()
        monitor = HealthMonitor(pool)
        data = os.urandom(1024)
        result = monitor.evaluate(data)
        monitor.stop()
        assert result["duplicate_rate"] < 0.01
