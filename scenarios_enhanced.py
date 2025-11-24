# ============================================================
# scenarios_enhanced.py — Erweiterte Physik-Presets für UI
# ============================================================
from __future__ import annotations

import math
import numpy as np

try:
    from core import Body, Connection, G
except ImportError:
    from .core import Body, Connection, G  # type: ignore


def scenario_charged_pair(high_charge=1e-6, v=20.0):
    a = Body('A', pos=np.array([-0.5, 0.0, 0.0]), vel=np.array([v, 0.0, 0.0]),
             mass=1.0, charge=high_charge, t0=0.0, dt=0.001, t_end=2.0, radius=0.0, color='red')
    b = Body('B', pos=np.array([0.5, 0.0, 0.0]), vel=np.array([-v, 0.0, 0.0]),
             mass=1.0, charge=-high_charge, t0=0.0, dt=0.001, t_end=2.0, radius=0.0, color='blue')
    return [a, b], [], f"Geladenes Paar (±{high_charge} C)"


def scenario_three_charges_demo(scale_uC=1.0):
    q1 = +3.0e-6 * scale_uC
    q2 = -1.0e-6 * scale_uC
    q3 = -3.0e-6 * scale_uC
    a = Body('Q1', pos=np.array([-1.0, 0.0, 0.0]), vel=np.zeros(3), mass=1.0,
             charge=q1, t0=0.0, dt=0.001, t_end=2.0, radius=0.0, color='red')
    b = Body('Q2', pos=np.array([0.0, 0.0, 0.0]), vel=np.zeros(3), mass=1.0,
             charge=q2, t0=0.0, dt=0.001, t_end=2.0, radius=0.0, color='blue')
    c = Body('Q3', pos=np.array([1.0, 0.0, 0.0]), vel=np.zeros(3), mass=1.0,
             charge=q3, t0=0.0, dt=0.001, t_end=2.0, radius=0.0, color='red')
    return [a, b, c], [], "Drei Ladungen (+3µC, -1µC, -3µC)"


def scenario_elastic_collision():
    a = Body('A', pos=np.array([-1.0, 0.0, 0.0]), vel=np.array([2.0, 0.0, 0.0]),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, radius=0.0, color='red')
    b = Body('B', pos=np.array([1.0, 0.0, 0.0]), vel=np.array([-2.0, 0.0, 0.0]),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, radius=0.0, color='blue')
    return [a, b], [], "Elastischer Stoß (e=1.0)"


def scenario_inelastic_collision():
    a = Body('A', pos=np.array([-1.0, 0.0, 0.0]), vel=np.array([2.0, 0.0, 0.0]),
             mass=2.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, radius=0.0, color='red')
    b = Body('B', pos=np.array([1.0, 0.0, 0.0]), vel=np.array([-1.0, 0.0, 0.0]),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, radius=0.0, color='blue')
    return [a, b], [], "Inelastischer Stoß (e<1.0)"


def scenario_spring_system():
    a = Body('A', pos=np.array([-1.0, 0.0, 0.0]), vel=np.zeros(3),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=5.0, radius=0.0, color='red')
    b = Body('B', pos=np.array([1.0, 0.0, 0.0]), vel=np.zeros(3),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=5.0, radius=0.0, color='blue')
    conn = [Connection(0, 1, 'elastic', 10.0, 2.0)]
    return [a, b], conn, "Federsystem"


def scenario_planetary_scaled(scale_mass=1e14, scale_distance=1.0, add_second=True,
                              t_end=50.0, dt=0.001, pin_sun=False):
    M = float(scale_mass)
    sun = Body('Sun', pos=np.array([0.0, 0.0, 0.0]), vel=np.array([0.0, 0.0, 0.0]),
               mass=M, charge=0.0, t0=0.0, dt=float(dt), t_end=float(t_end), radius=0.0, color='yellow')
    r1 = 1.0 * float(scale_distance)
    v1 = math.sqrt(G * M / r1)
    p1 = Body('P1', pos=np.array([r1, 0.0, 0.0]), vel=np.array([0.0, v1, 0.0]),
              mass=1.0, charge=0.0, t0=0.0, dt=float(dt), t_end=float(t_end), radius=0.0, color='blue')
    bodies = [sun, p1]
    if add_second:
        r2 = 1.6 * float(scale_distance)
        v2 = math.sqrt(G * M / r2)
        p2 = Body('P2', pos=np.array([r2, 0.0, 0.0]), vel=np.array([0.0, v2, 0.0]),
                  mass=0.5, charge=0.0, t0=0.0, dt=float(dt), t_end=float(t_end), radius=0.0, color='green')
        bodies.append(p2)
    if not pin_sun:
        total_p = np.zeros(3)
        for b in bodies:
            total_p += b.mass * b.vel
        sun.vel = - total_p / sun.mass
    return bodies, [], f"Planetensystem (M={scale_mass})"


PRESETS_ENHANCED = {
    'Geladenes Paar': scenario_charged_pair,
    'Drei Ladungen': scenario_three_charges_demo,
    'Elastischer Stoß': scenario_elastic_collision,
    'Inelastischer Stoß': scenario_inelastic_collision,
    'Federsystem': scenario_spring_system,
    'Planetensystem': scenario_planetary_scaled,
}
