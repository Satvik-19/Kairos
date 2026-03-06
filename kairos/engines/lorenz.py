"""Lorenz attractor chaos engine using RK4 integration."""
import struct
import threading

from .base import BaseChaosEngine


class LorenzEngine(BaseChaosEngine):
    """
    Lorenz attractor with RK4 integration.
    State: [x, y, z]
    Parameters: sigma=10, rho=28, beta=8/3
    """

    def __init__(self):
        # Parameters
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.dt = 0.005

        # Initial conditions
        self._x = 0.1
        self._y = 0.0
        self._z = 0.0

        super().__init__()

    def _derivatives(self, x, y, z):
        """Lorenz system equations."""
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return dx, dy, dz

    def _rk4_step(self, x, y, z):
        """Single RK4 step."""
        dt = self.dt
        k1 = self._derivatives(x, y, z)
        k2 = self._derivatives(x + dt / 2 * k1[0], y + dt / 2 * k1[1], z + dt / 2 * k1[2])
        k3 = self._derivatives(x + dt / 2 * k2[0], y + dt / 2 * k2[1], z + dt / 2 * k2[2])
        k4 = self._derivatives(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        nx = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ny = y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        nz = z + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        return nx, ny, nz

    def tick(self):
        """Advance 3 Lorenz steps per tick for speed."""
        with self._lock:
            x, y, z = self._x, self._y, self._z

        for _ in range(3):
            x, y, z = self._rk4_step(x, y, z)

        with self._lock:
            self._x = x
            self._y = y
            self._z = z

    def get_state(self) -> dict:
        """Return current x, y, z coordinates."""
        with self._lock:
            return {"x": self._x, "y": self._y, "z": self._z}

    def get_entropy_bytes(self) -> bytes:
        """Pack state as 3 doubles (24 bytes)."""
        with self._lock:
            return struct.pack("3d", self._x, self._y, self._z)

    def apply_perturbation(self, delta: float):
        """Apply micro-delta to x."""
        with self._lock:
            self._x += delta
