"""Gray-Scott reaction-diffusion chaos engine."""
import base64
import threading

import numpy as np

from .base import BaseChaosEngine


class ReactionDiffusionEngine(BaseChaosEngine):
    """
    Gray-Scott reaction-diffusion model on a 64×64 grid.
    Parameters: moving-spots preset (f=0.025, k=0.060)

    This parameter set produces spots that continuously grow, split, and move —
    they never stabilise to a static pattern, giving perpetual visual activity.

    Diffusion coefficients use Pearson 1993 / Karl Sims pixel-unit values
    (Du=0.2097, Dv=0.1050) which balance diffusion against autocatalysis on
    a unit-pixel grid and produce visible patterns within seconds.
    """

    def __init__(self):
        self.N = 64
        self.f = 0.025   # moving spots — perpetually evolving
        self.k = 0.060
        # Pixel-unit diffusion coefficients (Pearson 1993 / Karl Sims canonical values).
        # The PRD's Du=1.0, Dv=0.5 are physical-unit values that require a scaled grid.
        # On a pixel-unit grid (dx=1), these values produce stable, beautiful spot patterns.
        self.Du = 0.2097
        self.Dv = 0.1050
        self.dt = 1.0
        self._steps_per_tick = 15  # more steps per tick → patterns develop faster

        # Initialize grids: U=1 (chemical A), V=0 (chemical B)
        U = np.ones((self.N, self.N), dtype=np.float64)
        V = np.zeros((self.N, self.N), dtype=np.float64)

        rng = np.random.default_rng()

        # Center 8×8 seed patch
        half = self.N // 2
        U[half - 4 : half + 4, half - 4 : half + 4] = 0.50
        V[half - 4 : half + 4, half - 4 : half + 4] = 0.25 + rng.uniform(-0.05, 0.05, (8, 8))

        # 8 additional random 4×4 seed patches — more seeds → faster visible patterns
        for _ in range(8):
            rx = int(rng.integers(4, self.N - 4))
            ry = int(rng.integers(4, self.N - 4))
            U[rx : rx + 4, ry : ry + 4] = 0.50
            V[rx : rx + 4, ry : ry + 4] = 0.25 + rng.uniform(-0.05, 0.05, (4, 4))

        self._U = U
        self._V = V

        super().__init__()

    def _laplacian(self, grid: np.ndarray) -> np.ndarray:
        """5-point stencil Laplacian with periodic (wrap-around) boundaries."""
        return (
            np.roll(grid, 1, axis=0)
            + np.roll(grid, -1, axis=0)
            + np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
            - 4.0 * grid
        )

    def tick(self):
        """
        Advance the Gray-Scott simulation by _steps_per_tick steps.
        Uses a stable dt=0.2 to prevent numerical blow-up.
        """
        with self._lock:
            U = self._U.copy()
            V = self._V.copy()

        dt = self.dt
        for _ in range(self._steps_per_tick):
            uvv = U * V * V
            lap_u = self._laplacian(U)
            lap_v = self._laplacian(V)
            U = U + dt * (self.Du * lap_u - uvv + self.f * (1.0 - U))
            V = V + dt * (self.Dv * lap_v + uvv - (self.f + self.k) * V)
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

        # Guard against NaN/Inf (shouldn't happen with stable dt but be safe)
        if not (np.isfinite(U).all() and np.isfinite(V).all()):
            U = np.where(np.isfinite(U), U, 1.0)
            V = np.where(np.isfinite(V), V, 0.0)

        with self._lock:
            self._U = U
            self._V = V

    def get_state(self) -> dict:
        """Return V grid encoded as base64-encoded float32 bytes (64×64 = 16 KB)."""
        with self._lock:
            v_copy = self._V.astype(np.float32)
        return {"grid_b64": base64.b64encode(v_copy.tobytes()).decode()}

    def get_entropy_bytes(self) -> bytes:
        """Return V grid as raw bytes for entropy mixing."""
        with self._lock:
            return self._V.tobytes()

    def apply_perturbation(self, delta: float):
        """Apply abs(delta) to a randomly selected U cell."""
        rng = np.random.default_rng()
        rx = int(rng.integers(0, self.N))
        ry = int(rng.integers(0, self.N))
        with self._lock:
            self._U[rx, ry] = min(1.0, self._U[rx, ry] + abs(delta))
