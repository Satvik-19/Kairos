"""Double pendulum chaos engine using RK4 integration."""
import math
import struct
import threading

from .base import BaseChaosEngine


class DoublePendulumEngine(BaseChaosEngine):
    """
    Double pendulum with RK4 integration.
    State: [theta1, theta2, omega1, omega2]
    """

    def __init__(self):
        # Physical parameters
        self.m1 = 1.0
        self.m2 = 1.0
        self.L1 = 1.0
        self.L2 = 1.0
        self.g = 9.81
        self.dt = 0.01

        # Initial conditions (set before super().__init__ so thread is safe)
        self._theta1 = math.pi / 2 + 0.5
        self._theta2 = math.pi / 3
        self._omega1 = 0.0
        self._omega2 = 0.0

        super().__init__()

    def _derivatives(self, theta1, theta2, omega1, omega2):
        """Compute d/dt [theta1, theta2, omega1, omega2] for the double pendulum."""
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g
        delta = theta2 - theta1

        denom1 = (2 * m1 + m2 - m2 * math.cos(2 * delta)) * L1
        denom2 = (2 * m1 + m2 - m2 * math.cos(2 * delta)) * L2

        domega1 = (
            -g * (2 * m1 + m2) * math.sin(theta1)
            - m2 * g * math.sin(theta1 - 2 * theta2)
            - 2
            * math.sin(delta)
            * m2
            * (omega2 ** 2 * L2 + omega1 ** 2 * L1 * math.cos(delta))
        ) / denom1

        domega2 = (
            2
            * math.sin(delta)
            * (
                omega1 ** 2 * L1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(theta1)
                + omega2 ** 2 * L2 * m2 * math.cos(delta)
            )
        ) / denom2

        return omega1, omega2, domega1, domega2

    def tick(self):
        """Advance simulation one RK4 step."""
        with self._lock:
            t1, t2, w1, w2 = self._theta1, self._theta2, self._omega1, self._omega2

        dt = self.dt
        k1 = self._derivatives(t1, t2, w1, w2)
        k2 = self._derivatives(
            t1 + dt / 2 * k1[0],
            t2 + dt / 2 * k1[1],
            w1 + dt / 2 * k1[2],
            w2 + dt / 2 * k1[3],
        )
        k3 = self._derivatives(
            t1 + dt / 2 * k2[0],
            t2 + dt / 2 * k2[1],
            w1 + dt / 2 * k2[2],
            w2 + dt / 2 * k2[3],
        )
        k4 = self._derivatives(
            t1 + dt * k3[0],
            t2 + dt * k3[1],
            w1 + dt * k3[2],
            w2 + dt * k3[3],
        )

        new_t1 = t1 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        new_t2 = t2 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        new_w1 = w1 + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        new_w2 = w2 + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        with self._lock:
            self._theta1 = new_t1
            self._theta2 = new_t2
            self._omega1 = new_w1
            self._omega2 = new_w2

    def get_state(self) -> dict:
        """Return state with angles and cartesian positions."""
        with self._lock:
            t1, t2 = self._theta1, self._theta2
            w1, w2 = self._omega1, self._omega2

        x1 = self.L1 * math.sin(t1)
        y1 = -self.L1 * math.cos(t1)
        x2 = x1 + self.L2 * math.sin(t2)
        y2 = y1 - self.L2 * math.cos(t2)

        return {
            "theta1": t1,
            "theta2": t2,
            "omega1": w1,
            "omega2": w2,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    def get_entropy_bytes(self) -> bytes:
        """Pack state as 4 doubles (32 bytes)."""
        with self._lock:
            return struct.pack("4d", self._theta1, self._theta2, self._omega1, self._omega2)

    def apply_perturbation(self, delta: float):
        """Apply micro-delta to theta1."""
        with self._lock:
            self._theta1 += delta
