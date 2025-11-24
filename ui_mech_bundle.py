from __future__ import annotations
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sim_core_bundle import Simulator, Body

# Simple orbit helper
G = 6.67430e-11
AU = 1.495978707e11
M_sun = 1.98847e30
DAY = 86400.0

def two_body_orbit(n_steps: int = 365, dt: float = DAY):
    r = np.array([AU, 0.0], dtype=float)
    v = np.array([0.0, 29780.0], dtype=float)
    rs = np.zeros((n_steps, 2), dtype=float)
    vs = np.zeros((n_steps, 2), dtype=float)
    for i in range(n_steps):
        rs[i] = r; vs[i] = v
        dist = np.linalg.norm(r) + 1e-12
        a = -G * M_sun * r / dist**3
        v = v + a * dt
        r = r + v * dt
    return rs, vs

def render_mech_astro_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    st.subheader(tr("Mechanik & Himmelsmechanik", "Mechanics & Celestial Mechanics"))
    mech_tab, astro_tab = st.tabs([tr("Mechanik (2D)", "Mechanics (2D)"), tr("Himmelsmechanik (2D)", "Celestial mechanics (2D)")])

    with mech_tab:
        st.info(tr("Einfache 2D-Darstellung. Erweiterte Presets unten im Haupttab.", "Simple 2D view. Advanced presets below in main tab."))

    with astro_tab:
        st.markdown(tr("#### Himmelsmechanik: Sonne-Erde-Zweikörper-System", "Celestial mechanics: Sun-Earth two-body"))
        n_steps = st.slider(tr("Zeitschritte", "Time steps"), 90, 1000, 365, step=10, key="astro_steps")
        dt_days = st.number_input(tr("Zeitschritt [Tage]", "Timestep [days]"), min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="astro_dt_days")
        if st.button(tr("▶️ Orbit simulieren", "▶️ Simulate orbit"), use_container_width=True):
            st.info(tr("⏳ Berechne Orbit...", "⏳ Computing orbit..."))
            rs, _ = two_body_orbit(n_steps=int(n_steps), dt=float(dt_days)*86400.0)
            x = rs[:,0]/AU; y = rs[:,1]/AU
            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(0,0,"yo", markersize=12, label="Sun")
            ax.plot(x, y, "b-", label="Earth")
            ax.plot(x[0], y[0], "bo", markersize=6, label="Start")
            ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
            ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig)
