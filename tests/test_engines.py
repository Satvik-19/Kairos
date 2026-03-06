"""Tests for chaos engines."""
import math
import time

import pytest

from kairos.engines.double_pendulum import DoublePendulumEngine
from kairos.engines.lorenz import LorenzEngine
from kairos.engines.reaction_diffusion import ReactionDiffusionEngine


class TestDoublePendulumEngine:
    def setup_method(self):
        self.engine = DoublePendulumEngine()
        time.sleep(0.1)  # Let the engine run a few ticks

    def teardown_method(self):
        self.engine.stop()

    def test_initializes(self):
        assert self.engine is not None
        assert self.engine._running is True

    def test_get_entropy_bytes_length(self):
        data = self.engine.get_entropy_bytes()
        assert len(data) == 32  # 4 doubles × 8 bytes

    def test_get_state_keys(self):
        state = self.engine.get_state()
        for key in ("theta1", "theta2", "omega1", "omega2", "x1", "y1", "x2", "y2"):
            assert key in state

    def test_apply_perturbation_no_crash(self):
        self.engine.apply_perturbation(1e-9)
        state = self.engine.get_state()
        assert not math.isnan(state["theta1"])
        assert not math.isinf(state["theta1"])

    def test_100_ticks_no_nan(self):
        for _ in range(100):
            self.engine.tick()
        state = self.engine.get_state()
        for v in state.values():
            assert not math.isnan(v), f"NaN in state: {state}"
            assert not math.isinf(v), f"Inf in state: {state}"


class TestLorenzEngine:
    def setup_method(self):
        self.engine = LorenzEngine()
        time.sleep(0.1)

    def teardown_method(self):
        self.engine.stop()

    def test_initializes(self):
        assert self.engine is not None

    def test_get_entropy_bytes_length(self):
        data = self.engine.get_entropy_bytes()
        assert len(data) == 24  # 3 doubles × 8 bytes

    def test_get_state_keys(self):
        state = self.engine.get_state()
        for key in ("x", "y", "z"):
            assert key in state

    def test_apply_perturbation_no_crash(self):
        self.engine.apply_perturbation(1e-9)
        state = self.engine.get_state()
        assert not math.isnan(state["x"])

    def test_100_ticks_no_nan(self):
        for _ in range(100):
            self.engine.tick()
        state = self.engine.get_state()
        for v in state.values():
            assert not math.isnan(v)
            assert not math.isinf(v)


class TestReactionDiffusionEngine:
    def setup_method(self):
        self.engine = ReactionDiffusionEngine()
        time.sleep(0.1)

    def teardown_method(self):
        self.engine.stop()

    def test_initializes(self):
        assert self.engine is not None

    def test_get_entropy_bytes_not_empty(self):
        data = self.engine.get_entropy_bytes()
        assert len(data) == 64 * 64 * 8  # float64

    def test_get_state_has_grid_b64(self):
        state = self.engine.get_state()
        assert "grid_b64" in state
        assert isinstance(state["grid_b64"], str)
        assert len(state["grid_b64"]) > 0

    def test_apply_perturbation_no_crash(self):
        self.engine.apply_perturbation(1e-9)

    def test_100_ticks_no_nan(self):
        import numpy as np
        for _ in range(100):
            self.engine.tick()
        with self.engine._lock:
            assert not np.any(np.isnan(self.engine._V))
            assert not np.any(np.isinf(self.engine._V))
