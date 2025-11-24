# ============================================================
# physics_simulator/core.py
# Physik-Kern: Body, Connection, CollisionEvent, Simulator,
# physikalische Konstanten und numerische Integration
# ============================================================

import math
from dataclasses import dataclass
from typing import List, Optional, Callable

import numpy as np

# ============================================================
# Physikalische Konstanten (SI)
# ============================================================

G = 6.67430e-11                    # Gravitationskonstante
k_e = 8.9875517923e9               # Coulomb-Konstante
c = 299792458.0                    # Lichtgeschwindigkeit
softening = 1e-9                   # Softening für numerische Stabilität

# Convenience units
AU = 1.495978707e11                # Astronomische Einheit
KM = 1000.0
M_sun = 1.98847e30
DAY = 86400.0


# ============================================================
# Datenstrukturen
# ============================================================

@dataclass
class Body:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    mass: float
    t0: float
    dt: float
    t_end: float
    charge: float = 0.0
    radius: float = 0.0
    color: str = None


@dataclass
class Connection:
    i: int
    j: int
    typ: str
    strength: float
    rest_length: float


@dataclass
class CollisionEvent:
    time: float
    body1: int
    body2: int
    momentum_before: float
    momentum_after: float
    energy_before: float
    energy_after: float


# ============================================================
# Simulator
# ============================================================

class Simulator:
    def __init__(
        self,
        bodies: List[Body],
        connections: Optional[List[Connection]] = None,
        forces: Optional[List[Callable]] = None,
        restitution: float = 1.0,
        drag: float = 0.0,
    ):
        self.bodies = bodies
        self.connections = connections if connections else []
        self.forces = forces if forces else []
        self.restitution = restitution
        self.drag = drag

        self.collision_events: List[CollisionEvent] = []

    # --------------------------------------------------------
    # Hilfsfunktionen für Kräfte
    # --------------------------------------------------------

    def gravitational_force(self, i, j):
        bi, bj = self.bodies[i], self.bodies[j]
        r = bj.pos - bi.pos
        d = np.linalg.norm(r) + softening
        return G * bi.mass * bj.mass * r / (d ** 3)

    def electric_force(self, i, j):
        bi, bj = self.bodies[i], self.bodies[j]
        r = bj.pos - bi.pos
        d = np.linalg.norm(r) + softening
        return k_e * bi.charge * bj.charge * r / (d ** 3)

    def spring_force(self, b1, b2, strength, rest_length):
        r = b2.pos - b1.pos
        d = np.linalg.norm(r) + softening
        return -strength * (d - rest_length) * (r / d)

    # --------------------------------------------------------
    # Kollisionserkennung und -auflösung
    # --------------------------------------------------------

    def detect_collision(self, i, j):
        bi, bj = self.bodies[i], self.bodies[j]
        dist = np.linalg.norm(bi.pos - bj.pos)
        r_sum = (getattr(bi, "radius", 0.0) or 0.0) + (getattr(bj, "radius", 0.0) or 0.0)
        if r_sum <= 0:
            r_sum = (bi.mass + bj.mass) ** (1/3) * 0.001  # fallback auf Massen-basierten Näherungsradius
        return dist < r_sum

    def resolve_collision(self, i, j, time):
        bi, bj = self.bodies[i], self.bodies[j]

        m1, m2 = bi.mass, bj.mass
        v1, v2 = bi.vel, bj.vel

        momentum_before = m1 * np.linalg.norm(v1) + m2 * np.linalg.norm(v2)
        energy_before = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)

        v_rel = v1 - v2
        v1_new = v1 - (1 + self.restitution) * (m2 / (m1 + m2)) * v_rel
        v2_new = v2 + (1 + self.restitution) * (m1 / (m1 + m2)) * v_rel

        bi.vel = v1_new
        bj.vel = v2_new

        momentum_after = m1 * np.linalg.norm(v1_new) + m2 * np.linalg.norm(v2_new)
        energy_after = 0.5 * m1 * np.dot(v1_new, v1_new) + 0.5 * m2 * np.dot(v2_new, v2_new)

        self.collision_events.append(
            CollisionEvent(time, i, j, momentum_before, momentum_after, energy_before, energy_after)
        )

    # --------------------------------------------------------
    # Hauptintegrationsroutine
    # --------------------------------------------------------

    def run(self):
        if not self.bodies:
            return {}

        dt = min(b.dt for b in self.bodies)
        t_end = min(b.t_end for b in self.bodies)

        times = []
        positions = {i: [] for i in range(len(self.bodies))}
        velocities = {i: [] for i in range(len(self.bodies))}
        energies = []

        t = 0.0
        while t <= t_end:
            times.append(t)

            # Speicherung
            for i, b in enumerate(self.bodies):
                positions[i].append(b.pos.copy())
                velocities[i].append(b.vel.copy())

            # Kräfte berechnen
            accel = [np.zeros(3) for _ in self.bodies]

            # Paar-Kräfte
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    if self.bodies[i].mass > 0 and self.bodies[j].mass > 0:
                        Fg = self.gravitational_force(i, j)
                        accel[i] += Fg / self.bodies[i].mass
                        accel[j] -= Fg / self.bodies[j].mass

                    if self.bodies[i].charge or self.bodies[j].charge:
                        Fe = self.electric_force(i, j)
                        accel[i] += Fe / self.bodies[i].mass
                        accel[j] -= Fe / self.bodies[j].mass

            # Federkräfte
            for c in self.connections:
                b1 = self.bodies[c.i]
                b2 = self.bodies[c.j]
                Fs = self.spring_force(b1, b2, c.strength, c.rest_length)
                accel[c.i] += Fs / b1.mass
                accel[c.j] -= Fs / b2.mass

            # Luftwiderstand
            if self.drag > 0:
                for i, b in enumerate(self.bodies):
                    accel[i] -= self.drag * b.vel / b.mass

            # Externe Kräfte
            for f in self.forces:
                f(self.bodies, accel, t)

            # Euler-Integration
            for i, b in enumerate(self.bodies):
                b.vel += accel[i] * dt
                b.pos += b.vel * dt

            # Kollisionen
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    if self.detect_collision(i, j):
                        self.resolve_collision(i, j, t)

            # Energie
            total_E = sum(0.5 * b.mass * np.dot(b.vel, b.vel)
                          for b in self.bodies)
            energies.append(total_E)

            t += dt

        return {
            "times": np.array(times),
            "positions": {k: np.array(v) for k, v in positions.items()},
            "velocities": {k: np.array(v) for k, v in velocities.items()},
            "energies": np.array(energies),
        }
