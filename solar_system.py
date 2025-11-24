# ============================================================
# physics_simulator/solar_system.py
# Einfache 2D-Visualisierung eines Sonnensystems mittels Matplotlib
# (für externen Start durch launch_solar_system)
# ============================================================

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception as e:
    raise RuntimeError("Matplotlib wird für solar_system.py benötigt.") from e

# Hybrid-Imports für direkte Ausführung und Modul-Import
try:
    from .core import G, AU, M_sun, DAY
except ImportError:
    from core import G, AU, M_sun, DAY


def two_body_orbit(n_steps=365, dt=DAY):
    """
    Einfache numerische Integration eines Sonne-Erde-Systems in 2D.
    """
    m_sun = M_sun
    m_earth = 5.972e24

    # Anfangsbedingungen
    r = np.array([AU, 0.0])
    v = np.array([0.0, 29780.0])

    rs = np.zeros((n_steps, 2))
    vs = np.zeros((n_steps, 2))

    for i in range(n_steps):
        rs[i] = r
        vs[i] = v

        dist = np.linalg.norm(r) + 1e-9
        a = -G * m_sun * r / dist**3
        v = v + a * dt
        r = r + v * dt

    return rs, vs


def main():
    rs, _ = two_body_orbit(n_steps=365, dt=DAY)
    x = rs[:, 0] / AU
    y = rs[:, 1] / AU

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title("Einfaches Sonne-Erde-System")

    sun, = ax.plot([0], [0], "yo", markersize=10, label="Sonne")
    earth, = ax.plot([], [], "bo", markersize=5, label="Erde")
    orbit, = ax.plot([], [], "b--", linewidth=1, alpha=0.5)
    ax.legend()

    def init():
        earth.set_data([], [])
        orbit.set_data([], [])
        return earth, orbit

    def update(frame):
        earth.set_data([x[frame]], [y[frame]])
        orbit.set_data(x[:frame+1], y[:frame+1])
        return earth, orbit

    ani = FuncAnimation(fig, update, frames=len(x),
                        init_func=init, interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
