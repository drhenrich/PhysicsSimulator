from __future__ import annotations

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt


def render_mech_astro_tab():
    """Mechanik & Astromechanik (Safe Mode) mit Simulation & Visualisierung"""
    
    st.subheader("Mechanik & Astromechanik — Safe Mode")
    
    mech_tab, astro_tab = st.tabs(["Mechanik (2D)", "Astromechanik (2D)"])
    
    def simulate_mechanics(objects, t_end, dt, interaction_type="Gravitation", connection_type="Keine"):
        """Simuliere Objektbewegungen mit Physik"""
        num_objects = len(objects)
        
        # Zeitschritte
        n_steps = int(t_end / dt)
        
        # Initialisiere Positionen und Geschwindigkeiten
        positions = np.array([[obj["x"], obj["y"]] for obj in objects])  # (N, 2)
        velocities = np.array([[obj["vx"], obj["vy"]] for obj in objects])  # (N, 2)
        masses = np.array([obj["mass"] for obj in objects])  # (N,)
        charges = np.array([obj["charge"] for obj in objects])  # (N,)
        
        # Speichere Trajektorien
        trajectory = [positions.copy()]
        
        G = 6.674e-11  # Gravitationskonstante
        k_e = 8.988e9  # Coulomb-Konstante
        
        for step in range(n_steps):
            forces = np.zeros_like(positions)
            
            # Berechne Kräfte zwischen allen Objektpaaren
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec) + 1e-10  # Avoid division by zero
                    r_hat = r_vec / r
                    
                    if interaction_type == "Gravitation":
                        # Gravitationskraft
                        F = G * masses[i] * masses[j] / (r**2 + 1e-10)
                        forces[i] += F * r_hat
                        forces[j] -= F * r_hat
                    
                    elif interaction_type == "Elektrodynamisch":
                        # Coulomb-Kraft
                        if charges[i] != 0 and charges[j] != 0:
                            F = k_e * charges[i] * charges[j] / (r**2 + 1e-10)
                            forces[i] += F * r_hat
                            forces[j] -= F * r_hat
            
            # Update Geschwindigkeiten und Positionen
            for i in range(num_objects):
                if masses[i] > 0:
                    accelerations = forces[i] / masses[i]
                    velocities[i] += accelerations * dt
                    positions[i] += velocities[i] * dt
            
            trajectory.append(positions.copy())
        
        return np.array(trajectory), positions, velocities
    
    def plot_trajectory(trajectory):
        """Plotte die Trajektorie der Objekte"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        num_objects = trajectory.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, num_objects))
        
        for obj_id in range(num_objects):
            x_traj = trajectory[:, obj_id, 0]
            y_traj = trajectory[:, obj_id, 1]
            ax.plot(x_traj, y_traj, "-", color=colors[obj_id], label=f"Objekt {obj_id + 1}", linewidth=1.5, alpha=0.7)
            ax.plot(x_traj[-1], y_traj[-1], "o", color=colors[obj_id], markersize=8)  # Endposition
        
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        ax.set_title("Objekttrajektorien", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.axis("equal")
        
        plt.tight_layout()
        return fig
    
    def plot_positions(positions, objects):
        """Plotte die Endpositionen und Geschwindigkeiten"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        num_objects = len(objects)
        colors = plt.cm.tab10(np.linspace(0, 1, num_objects))
        
        for i in range(num_objects):
            x, y = positions[i]
            vx, vy = objects[i]["vx"], objects[i]["vy"]
            mass = objects[i]["mass"]
            
            # Größe basierend auf Masse (skaliert)
            size = 100 * np.log10(mass + 1)
            
            ax.scatter(x, y, s=size, color=colors[i], alpha=0.7, edgecolors="black", linewidth=1.5)
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                scale_factor = max(abs(x), abs(y)) * 0.1 if max(abs(x), abs(y)) > 0 else 1
                ax.arrow(x, y, vx * scale_factor, vy * scale_factor, head_width=scale_factor * 0.1, 
                        head_length=scale_factor * 0.1, fc=colors[i], ec=colors[i], alpha=0.5)
            ax.text(x, y + max(abs(x), abs(y)) * 0.05, f"Obj {i + 1}", fontsize=9, ha="center")
        
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        ax.set_title("Endpositionen & Geschwindigkeitsrichtung", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        
        plt.tight_layout()
        return fig
    
    def plot_electric_field(positions, charges, grid_size=80):
        """Plotte elektrisches Feldlinien und Potentialfeld mit besserer Auflösung"""
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        k_e = 8.988e9  # Coulomb-Konstante
        
        # Bestimme Grid-Grenzen mit Padding
        x_min, x_max = positions[:, 0].min() - 2, positions[:, 0].max() + 2
        y_min, y_max = positions[:, 1].min() - 2, positions[:, 1].max() + 2
        
        # Erstelle Grid mit höherer Auflösung
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Berechne elektrisches Feld und Potential
        E_x = np.zeros_like(X, dtype=float)
        E_y = np.zeros_like(Y, dtype=float)
        V = np.zeros_like(X, dtype=float)
        
        for i in range(len(charges)):
            if charges[i] != 0:
                dx = X - positions[i, 0]
                dy = Y - positions[i, 1]
                r = np.sqrt(dx**2 + dy**2) + 1e-8
                
                # Elektrisches Feld: E = k*q/r^2 * r_hat
                E_mag = k_e * abs(charges[i]) / (r**2 + 1e-8)
                
                # Normalisiere für bessere Visualisierung
                E_mag = np.clip(E_mag, 0, np.percentile(E_mag, 95))
                
                r_hat_x = dx / (r + 1e-8)
                r_hat_y = dy / (r + 1e-8)
                
                if charges[i] > 0:  # Positive Ladung: Feld zeigt nach außen
                    E_x += E_mag * r_hat_x
                    E_y += E_mag * r_hat_y
                else:  # Negative Ladung: Feld zeigt nach innen
                    E_x -= E_mag * r_hat_x
                    E_y -= E_mag * r_hat_y
                
                # Potential: V = k*q/r
                V += k_e * charges[i] / (r + 1e-8)
        
        # ===== PLOT 1: FELDLINIEN =====
        E_mag = np.sqrt(E_x**2 + E_y**2) + 1e-10
        
        # Normalisiere Feldvektoren für bessere Visualisierung
        E_x_norm = E_x / (E_mag + 1e-10)
        E_y_norm = E_y / (E_mag + 1e-10)
        
        # Quiver mit besseren Parametern
        skip = max(1, grid_size // 12)
        quiver = ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                           E_x_norm[::skip, ::skip], E_y_norm[::skip, ::skip], 
                           E_mag[::skip, ::skip], cmap="viridis", scale=25, scale_units="xy", width=0.003)
        
        # Zeige Ladungen
        for i, (pos, q) in enumerate(zip(positions, charges)):
            if q != 0:
                color = 'red' if q > 0 else 'blue'
                size = 200 * abs(q) * 1e6  # Skaliert mit Ladungsbetrag
                ax1.scatter(pos[0], pos[1], s=size, c=color, edgecolors='black', 
                           linewidth=2, alpha=0.9, zorder=5, marker='+' if q > 0 else '_')
                ax1.text(pos[0], pos[1] + 0.3, f"q={q:.1e}\nC", fontsize=9, ha="center", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        ax1.set_xlabel("x [m]", fontsize=12)
        ax1.set_ylabel("y [m]", fontsize=12)
        ax1.set_title("Elektrische Feldlinien", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")
        cbar1 = plt.colorbar(quiver, ax=ax1)
        cbar1.set_label("Feldstärke [N/C]", fontsize=11)
        
        # ===== PLOT 2: POTENTIALFELD =====
        # Begrenzte Potentialberechnung für bessere Visualisierung
        V_clipped = np.clip(V, np.percentile(V, 1), np.percentile(V, 99))
        
        # Contourf mit mehr Levels
        levels = np.linspace(V_clipped.min(), V_clipped.max(), 30)
        contourf = ax2.contourf(X, Y, V_clipped, levels=levels, cmap="RdBu_r", alpha=0.9)
        
        # Contour-Linien
        contour_lines = ax2.contour(X, Y, V_clipped, levels=15, colors="black", alpha=0.2, linewidths=0.5)
        ax2.clabel(contour_lines, inline=True, fontsize=7, fmt="%.1e")
        
        # Zeige Ladungen
        for i, (pos, q) in enumerate(zip(positions, charges)):
            if q != 0:
                color = 'red' if q > 0 else 'blue'
                size = 200 * abs(q) * 1e6
                ax2.scatter(pos[0], pos[1], s=size, c=color, edgecolors='black', 
                           linewidth=2, alpha=0.9, zorder=5, marker='+' if q > 0 else '_')
        
        ax2.set_xlabel("x [m]", fontsize=12)
        ax2.set_ylabel("y [m]", fontsize=12)
        ax2.set_title("Elektrisches Potential", fontsize=14, fontweight="bold")
        ax2.set_aspect("equal")
        
        cbar2 = plt.colorbar(contourf, ax=ax2)
        cbar2.set_label("Potential [V]", fontsize=11)
        
        plt.tight_layout()
        return fig
    
    # ============================================
    # MECHANIK TAB
    # ============================================
    with mech_tab:
        st.markdown("#### Objekt-Konfiguration")
        
        # Anzahl Objekte
        col1, col2 = st.columns([1, 3])
        with col1:
            num_objects = st.slider("Anzahl Objekte", 1, 10, 2, key="mech_num_objects")
        
        # Initialisiere Objekt-Storage in session_state
        if "mech_objects" not in st.session_state:
            st.session_state.mech_objects = {}
        
        # Objekt-Editoren
        st.markdown("---")
        
        for obj_id in range(num_objects):
            with st.expander(f"🔵 Objekt {obj_id + 1}", expanded=(obj_id == 0)):
                col1, col2, col3 = st.columns(3)
                
                # Position (x, y, z)
                with col1:
                    st.markdown("**Position [m]**")
                    x = st.number_input(f"x##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("x", 0.0), step=0.1, key=f"mech_x_{obj_id}")
                    y = st.number_input(f"y##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("y", 0.0), step=0.1, key=f"mech_y_{obj_id}")
                    z = st.number_input(f"z##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("z", 0.0), step=0.1, key=f"mech_z_{obj_id}")
                
                # Geschwindigkeit (vx, vy, vz)
                with col2:
                    st.markdown("**Geschwindigkeit [m/s]**")
                    vx = st.number_input(f"vx##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("vx", 0.0), step=0.1, key=f"mech_vx_{obj_id}")
                    vy = st.number_input(f"vy##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("vy", 0.0), step=0.1, key=f"mech_vy_{obj_id}")
                    vz = st.number_input(f"vz##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("vz", 0.0), step=0.1, key=f"mech_vz_{obj_id}")
                
                # Masse & Ladung
                with col3:
                    st.markdown("**Physikalische Eigenschaften**")
                    mass = st.number_input(f"Masse [kg]##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("mass", 1.0), min_value=0.001, step=0.1, key=f"mech_mass_{obj_id}")
                    charge = st.number_input(f"Ladung [C]##obj{obj_id}", value=st.session_state.mech_objects.get(obj_id, {}).get("charge", 0.0), step=1e-6, format="%.1e", key=f"mech_charge_{obj_id}")
                
                st.markdown("---")
                
                # Wechselwirkungen & Verbindungen
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Wechselwirkung**")
                    interaction = st.radio(
                        "Art der Wechselwirkung",
                        ["Gravitation", "Elektrodynamisch"],
                        key=f"mech_interaction_{obj_id}",
                        index=0 if st.session_state.mech_objects.get(obj_id, {}).get("interaction", "Gravitation") == "Gravitation" else 1
                    )
                
                with col2:
                    st.markdown("**Verbindung**")
                    connection_type = st.selectbox(
                        "Verbindungstyp",
                        ["Keine", "Starr", "Elastisch"],
                        key=f"mech_connection_{obj_id}",
                        index=0 if st.session_state.mech_objects.get(obj_id, {}).get("connection", "Keine") == "Keine" else (1 if st.session_state.mech_objects.get(obj_id, {}).get("connection", "Keine") == "Starr" else 2)
                    )
                    
                    if connection_type == "Elastisch":
                        spring_strength = st.slider(
                            "Federstärke [N/m]",
                            min_value=1.0,
                            max_value=1000.0,
                            value=st.session_state.mech_objects.get(obj_id, {}).get("spring_strength", 100.0),
                            step=10.0,
                            key=f"mech_spring_{obj_id}"
                        )
                    else:
                        spring_strength = 0.0
                
                # Speichere Objekt-Daten
                st.session_state.mech_objects[obj_id] = {
                    "x": x, "y": y, "z": z,
                    "vx": vx, "vy": vy, "vz": vz,
                    "mass": mass,
                    "charge": charge,
                    "interaction": interaction,
                    "connection": connection_type,
                    "spring_strength": spring_strength
                }
        
        st.markdown("---")
        
        # Simulationsparameter
        st.markdown("#### Simulationsparameter")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            N = st.selectbox("Raster N (Anzeige)", [128, 256, 384, 512], index=1, key="mechN")
        
        with col2:
            t_end = st.number_input("t_end [s]", min_value=0.1, max_value=20.0, value=10.0, step=0.5, key="mech_t_end")
        
        with col3:
            dt = st.number_input("dt [s]", min_value=0.001, max_value=0.1, value=0.02, step=0.001, key="mech_dt")
        
        if st.button("▶️ Simulation starten##mech", use_container_width=True):
            st.info("⏳ Berechne Simulation...")
            
            # Sammle Objekt-Daten
            objects = [st.session_state.mech_objects[i] for i in range(num_objects)]
            
            # Bestimme Wechselwirkungstyp vom ersten Objekt
            interaction_type = objects[0].get("interaction", "Gravitation")
            
            # Führe Simulation durch
            trajectory, final_positions, final_velocities = simulate_mechanics(
                objects, t_end, dt, interaction_type=interaction_type
            )
            
            st.success("✅ Simulation abgeschlossen!")
            
            # Zeige Ergebnisse
            st.markdown("#### Simulationsergebnisse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Trajektorie-Plot**")
                fig_traj = plot_trajectory(trajectory)
                st.pyplot(fig_traj)
            
            with col2:
                st.markdown("**Endpositionen & Geschwindigkeiten**")
                fig_pos = plot_positions(final_positions, objects)
                st.pyplot(fig_pos)
            
            # Zeige elektrisches Feld, wenn Ladungen vorhanden
            charges = np.array([obj["charge"] for obj in objects])
            if interaction_type == "Elektrodynamisch" and np.any(charges != 0):
                st.markdown("#### Elektrisches Feld & Potential")
                fig_field = plot_electric_field(final_positions, charges, grid_size=80)
                st.pyplot(fig_field)
            
            # Statistiken
            st.markdown("---")
            st.markdown("#### Simulationsstatistiken")
            
            for i in range(num_objects):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"Obj {i + 1} - Start X", f"{objects[i]['x']:.2f} m")
                with col2:
                    st.metric(f"End X", f"{final_positions[i, 0]:.2f} m")
                with col3:
                    st.metric(f"Start vx", f"{objects[i]['vx']:.2f} m/s")
                with col4:
                    st.metric(f"End vx", f"{final_velocities[i, 0]:.2f} m/s")
    
    # ============================================
    # ASTROMECHANIK TAB
    # ============================================
    with astro_tab:
        st.markdown("#### Himmelskörper-Konfiguration")
        
        num_bodies = st.slider("Anzahl Himmelskörper", 1, 5, 2, key="astro_num_bodies")
        
        # Initialisiere Himmelskörper-Storage
        if "astro_bodies" not in st.session_state:
            st.session_state.astro_bodies = {}
        
        for body_id in range(num_bodies):
            with st.expander(f"🌍 Himmelskörper {body_id + 1}", expanded=(body_id == 0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Position [m]**")
                    x = st.number_input(f"x##astro{body_id}", value=st.session_state.astro_bodies.get(body_id, {}).get("x", 0.0), step=1e10, key=f"astro_x_{body_id}")
                    y = st.number_input(f"y##astro{body_id}", value=st.session_state.astro_bodies.get(body_id, {}).get("y", 1e11), step=1e10, key=f"astro_y_{body_id}")
                
                with col2:
                    st.markdown("**Geschwindigkeit [m/s]**")
                    vx = st.number_input(f"vx##astro{body_id}", value=st.session_state.astro_bodies.get(body_id, {}).get("vx", 0.0), step=1e3, key=f"astro_vx_{body_id}")
                    vy = st.number_input(f"vy##astro{body_id}", value=st.session_state.astro_bodies.get(body_id, {}).get("vy", 3e4), step=1e3, key=f"astro_vy_{body_id}")
                
                col1, col2 = st.columns(2)
                with col1:
                    mass = st.number_input(f"Masse [kg]##astro{body_id}", value=st.session_state.astro_bodies.get(body_id, {}).get("mass", 1e24), min_value=1e20, step=1e23, key=f"astro_mass_{body_id}")
                
                with col2:
                    name = st.text_input(f"Name##astro{body_id}", value=st.session_state.astro_bodies.get(body_id, {}).get("name", f"Body {body_id + 1}"), key=f"astro_name_{body_id}")
                
                # Speichere Himmelskörper-Daten
                st.session_state.astro_bodies[body_id] = {
                    "x": x, "y": y,
                    "vx": vx, "vy": vy,
                    "mass": mass,
                    "name": name
                }
        
        st.markdown("---")
        
        # Astro-Simulationsparameter
        st.markdown("#### Simulationsparameter")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            N = st.selectbox("Raster N (Anzeige)", [256, 384, 512, 768], index=0, key="astroN")
        
        with col2:
            t_end = st.number_input("t_end [s]", min_value=1e6, max_value=1e9, value=1e8, step=1e7, key="astro_t_end")
        
        with col3:
            dt = st.number_input("dt [s]", min_value=1e4, max_value=1e6, value=1e5, step=1e4, key="astro_dt")
        
        if st.button("▶️ Simulation starten##astro", use_container_width=True):
            st.info("⏳ Berechne Astro-Simulation...")
            
            # Sammle Himmelskörper-Daten als Objects
            bodies = []
            for body_id in range(num_bodies):
                body_data = st.session_state.astro_bodies[body_id]
                bodies.append({
                    "x": body_data["x"],
                    "y": body_data["y"],
                    "vx": body_data["vx"],
                    "vy": body_data["vy"],
                    "mass": body_data["mass"],
                    "charge": 0.0
                })
            
            # Führe Simulation durch (nur Gravitation)
            trajectory, final_positions, final_velocities = simulate_mechanics(
                bodies, int(t_end), int(dt), interaction_type="Gravitation"
            )
            
            st.success("✅ Astro-Simulation abgeschlossen!")
            
            # Zeige Ergebnisse
            st.markdown("#### Simulationsergebnisse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Orbits**")
                fig_traj = plot_trajectory(trajectory)
                st.pyplot(fig_traj)
            
            with col2:
                st.markdown("**Endpositionen**")
                fig_pos = plot_positions(final_positions, bodies)
                st.pyplot(fig_pos)