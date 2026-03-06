"""Tests for PerturbationScheduler."""
import math
import time

from kairos.engines.double_pendulum import DoublePendulumEngine
from kairos.engines.lorenz import LorenzEngine
from kairos.entropy.perturbation import PerturbationScheduler


class TestPerturbationScheduler:
    def test_perturbation_changes_engine_state(self):
        """After calling _run(), engine state should be different."""
        pendulum = DoublePendulumEngine()
        lorenz = LorenzEngine()
        time.sleep(0.05)  # Let engines warm up

        # Record state before perturbation
        before_t1 = pendulum._theta1
        before_x = lorenz._x

        scheduler = PerturbationScheduler([pendulum, lorenz])
        # Fire manually instead of waiting 10s
        scheduler._running = False  # prevent auto-reschedule
        scheduler._run()

        after_t1 = pendulum._theta1
        after_x = lorenz._x

        pendulum.stop()
        lorenz.stop()

        assert after_t1 != before_t1 or after_x != before_x, (
            "Perturbation did not change any engine state"
        )

    def test_delta_within_epsilon_bounds(self):
        """Delta magnitude should be within epsilon = 1e-9."""
        import hashlib
        import os
        import struct
        import sys

        epsilon = 1e-9
        for _ in range(100):
            raw = os.urandom(32)
            delta_bytes = hashlib.sha3_256(raw).digest()[:8]
            delta_float = struct.unpack("d", delta_bytes)[0]
            delta = (delta_float / sys.float_info.max) * epsilon
            assert abs(delta) <= epsilon, f"Delta {delta} exceeds epsilon {epsilon}"
            assert not math.isnan(delta)
            assert not math.isinf(delta)

    def test_last_perturbation_at_is_set(self):
        """After _run(), last_perturbation_at should return an ISO timestamp string."""
        pendulum = DoublePendulumEngine()
        scheduler = PerturbationScheduler([pendulum])
        scheduler._running = False
        scheduler._run()
        ts = scheduler.last_perturbation_at()
        pendulum.stop()
        assert ts is not None
        assert "T" in ts  # ISO format

    def test_stop_cancels_timer(self):
        """Stop should not raise and should prevent further fires."""
        pendulum = DoublePendulumEngine()
        scheduler = PerturbationScheduler([pendulum])
        scheduler.stop()
        pendulum.stop()
        assert not scheduler._running
