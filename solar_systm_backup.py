# ============================================================
# solar_systm.py — Einfaches Sonne-Erde-Zweikörpersystem (2D)
# ============================================================
from __future__ import annotations

import numpy as np

G = 6.67430e-11        # Gravitationskonstante
AU = 1.495978707e11    # Astronomische Einheit [m]
M_SUN = 1.98847e30     # Masse Sonne [kg]
M_EARTH = 5.972e24     # Masse Erde [kg]
DAY = 86400.0          # Sekunde pro Tag


def two_body_orbit(n_steps: int = 365, dt: float = DAY):
    """
    Integriert ein einfaches Sonne-Erde-System mit symplektischem Euler.
    Returns: (rs, vs) mit shape (n_steps, 2) je (x,y) in Meter.
    """
    # Anfangsbedingungen: Erde auf x-Achse, Startgeschwindigkeit in y-Richtung
    r = np.array([AU, 0.0], dtype=float)
    v = np.array([0.0, 29780.0], dtype=float)  # ~29.78 km/s

    rs = np.zeros((n_steps, 2), dtype=float)
    vs = np.zeros((n_steps, 2), dtype=float)

    for i in range(n_steps):
        rs[i] = r
        vs[i] = v

        dist = np.linalg.norm(r) + 1e-12
        a = -G * M_SUN * r / dist**3

        # symplektischer Euler
        v = v + a * dt
        r = r + v * dt

    return rs, vs
