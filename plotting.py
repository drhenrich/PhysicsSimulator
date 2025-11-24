# ============================================================
# physics_simulator/plotting.py
# Plot-Funktionen (3D-Trajektorien, Energie, Kollisionen, Potenzial)
# ============================================================

import numpy as np

# Hybrid-Imports für direkte Ausführung und Modul-Import
try:
    from .core import k_e
except ImportError:
    from core import k_e

# Plotly
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Matplotlib (Fallback)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ------------------------------------------------------------
# 3D-Trajektorien
# ------------------------------------------------------------

def plot_trajectory_3d(bodies, data, coordinate_system: str = "cartesian"):
    """
    Stellt die Trajektorien aller Körper in 3D dar.
    coordinate_system: 'cartesian' (derzeit nur kartesisch implementiert).
    """
    positions = data.get("positions", {})

    if HAS_PLOTLY:
        fig = go.Figure()
        for i, b in enumerate(bodies):
            pos = positions.get(i)
            if pos is None:
                continue
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                name=b.name
            ))
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode="markers",
                name=f"{b.name} start",
                marker=dict(size=4)
            ))
        fig.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
            ),
            title="3D-Trajektorien"
        )
        return fig

    elif HAS_MATPLOTLIB:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, b in enumerate(bodies):
            pos = positions.get(i)
            if pos is None:
                continue
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            ax.plot(x, y, z, label=b.name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        ax.set_title("3D-Trajektorien")
        return fig

    else:
        raise RuntimeError("Weder Plotly noch Matplotlib verfügbar.")


# ------------------------------------------------------------
# Energie- und Impulserhaltung
# ------------------------------------------------------------

def plot_conservation_laws(bodies, data):
    """
    Plottet die im Simulator berechnete Gesamtenergie als Funktion der Zeit.
    """
    times = data.get("times", None)
    energies = data.get("energies", None)

    if times is None or energies is None:
        raise ValueError("Daten enthalten keine 'times' oder 'energies'.")

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=energies,
            mode="lines",
            name="Gesamtenergie"
        ))
        fig.update_layout(
            xaxis_title="t",
            yaxis_title="E",
            title="Energieverlauf"
        )
        return fig

    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots()
        ax.plot(times, energies, label="Gesamtenergie")
        ax.set_xlabel("t")
        ax.set_ylabel("E")
        ax.set_title("Energieverlauf")
        ax.legend()
        return fig

    else:
        raise RuntimeError("Weder Plotly noch Matplotlib verfügbar.")


# ------------------------------------------------------------
# Kollisionsanalyse
# ------------------------------------------------------------

def plot_collision_analysis(collision_events):
    """
    Stellt Änderungen in Impuls und Energie bei Kollisionen dar.
    collision_events: Liste von CollisionEvent-Instanzen.
    """
    if not collision_events:
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.update_layout(
                title="Keine Kollisionen registriert."
            )
            return fig
        elif HAS_MATPLOTLIB:
            fig, ax = plt.subplots()
            ax.set_title("Keine Kollisionen registriert.")
            return fig
        else:
            raise RuntimeError("Keine Plot-Bibliothek verfügbar.")

    times = [ev.time for ev in collision_events]
    dP = [ev.momentum_after - ev.momentum_before for ev in collision_events]
    dE = [ev.energy_after - ev.energy_before for ev in collision_events]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=times, y=dP, name="Δ Impuls"))
        fig.add_trace(go.Bar(x=times, y=dE, name="Δ Energie"))
        fig.update_layout(
            barmode="group",
            xaxis_title="t (Kollision)",
            title="Kollisionsanalyse"
        )
        return fig

    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots()
        ax.bar(times, dP, label="Δ Impuls")
        ax.bar(times, dE, label="Δ Energie", alpha=0.7)
        ax.set_xlabel("t (Kollision)")
        ax.set_title("Kollisionsanalyse")
        ax.legend()
        return fig

    else:
        raise RuntimeError("Keine Plot-Bibliothek verfügbar.")


# ------------------------------------------------------------
# Elektrostatisches Potenzialfeld (2D)
# ------------------------------------------------------------

def plot_potential_field(bodies, x_range=(-2, 2), y_range=(-2, 2), resolution=50):
    """
    Berechnet und visualisiert das 2D-Potenzialfeld einer Menge von Punktladungen
    im z=0-Plane.
    """
    charges = [b for b in bodies if abs(b.charge) > 0]

    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    V = np.zeros_like(X)

    for b in charges:
        rx = X - b.pos[0]
        ry = Y - b.pos[1]
        r = np.sqrt(rx ** 2 + ry ** 2) + 1e-9
        V += k_e * b.charge / r

    if HAS_PLOTLY:
        fig = go.Figure(data=go.Contour(
            x=xs,
            y=ys,
            z=V,
            contours=dict(showlabels=True),
            colorbar=dict(title="V")
        ))
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            title="Elektrostatisches Potenzialfeld (z=0)"
        )
        return fig

    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, V, levels=30)
        fig.colorbar(cs, ax=ax, label="V")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Elektrostatisches Potenzialfeld (z=0)")
        return fig

    else:
        raise RuntimeError("Keine Plot-Bibliothek verfügbar.")
