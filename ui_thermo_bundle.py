"""
Thermodynamik & Wärmelehre Modul
- Wärmeleitung (1D/2D) mit Echtzeit-Animation
- Kinetische Gastheorie (animierte Teilchen)
- Kreisprozesse (Carnot, Otto, Diesel)
- Zustandsänderungen idealer Gase
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# PHYSICS CORE: Thermodynamics
# ============================================================

# Constants
R_GAS = 8.314  # J/(mol·K)
k_B = 1.380649e-23  # J/K


@dataclass
class GasState:
    """Zustand eines idealen Gases"""
    p: float  # Druck [Pa]
    V: float  # Volumen [m³]
    T: float  # Temperatur [K]
    n: float = 1.0  # Stoffmenge [mol]

    @classmethod
    def from_pV(cls, p: float, V: float, n: float = 1.0):
        T = p * V / (n * R_GAS)
        return cls(p=p, V=V, T=T, n=n)

    @classmethod
    def from_TV(cls, T: float, V: float, n: float = 1.0):
        p = n * R_GAS * T / V
        return cls(p=p, V=V, T=T, n=n)

    @classmethod
    def from_pT(cls, p: float, T: float, n: float = 1.0):
        V = n * R_GAS * T / p
        return cls(p=p, V=V, T=T, n=n)


def isothermal_process(state: GasState, V_end: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Isotherme Zustandsänderung: T = const, pV = const"""
    V = np.linspace(state.V, V_end, n_points)
    p = state.p * state.V / V  # p1*V1 = p2*V2
    return V, p


def isobaric_process(state: GasState, V_end: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Isobare Zustandsänderung: p = const"""
    V = np.linspace(state.V, V_end, n_points)
    p = np.full_like(V, state.p)
    return V, p


def isochoric_process(state: GasState, p_end: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Isochore Zustandsänderung: V = const"""
    p = np.linspace(state.p, p_end, n_points)
    V = np.full_like(p, state.V)
    return V, p


def adiabatic_process(state: GasState, V_end: float, gamma: float = 1.4, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Adiabatische Zustandsänderung: pV^γ = const"""
    V = np.linspace(state.V, V_end, n_points)
    p = state.p * (state.V / V) ** gamma
    return V, p


def carnot_cycle(T_hot: float, T_cold: float, V1: float, V2: float, n: float = 1.0, n_points: int = 50):
    """
    Carnot-Kreisprozess:
    1→2: isotherme Expansion bei T_hot
    2→3: adiabatische Expansion T_hot → T_cold
    3→4: isotherme Kompression bei T_cold
    4→1: adiabatische Kompression T_cold → T_hot
    """
    gamma = 1.4
    
    # Zustand 1: Start (T_hot, V1)
    p1 = n * R_GAS * T_hot / V1
    
    # Zustand 2: nach isothermer Expansion (T_hot, V2)
    p2 = n * R_GAS * T_hot / V2
    
    # Zustand 3: nach adiabatischer Expansion (T_cold, V3)
    # T_hot * V2^(γ-1) = T_cold * V3^(γ-1)
    V3 = V2 * (T_hot / T_cold) ** (1 / (gamma - 1))
    p3 = n * R_GAS * T_cold / V3
    
    # Zustand 4: nach isothermer Kompression (T_cold, V4)
    V4 = V1 * (T_hot / T_cold) ** (1 / (gamma - 1))
    p4 = n * R_GAS * T_cold / V4
    
    # Pfade
    V_12 = np.linspace(V1, V2, n_points)
    p_12 = p1 * V1 / V_12  # isotherm
    
    V_23 = np.linspace(V2, V3, n_points)
    p_23 = p2 * (V2 / V_23) ** gamma  # adiabatisch
    
    V_34 = np.linspace(V3, V4, n_points)
    p_34 = p3 * V3 / V_34  # isotherm
    
    V_41 = np.linspace(V4, V1, n_points)
    p_41 = p4 * (V4 / V_41) ** gamma  # adiabatisch
    
    V_cycle = np.concatenate([V_12, V_23, V_34, V_41])
    p_cycle = np.concatenate([p_12, p_23, p_34, p_41])
    
    # Wirkungsgrad
    eta = 1 - T_cold / T_hot
    
    return V_cycle, p_cycle, eta


def otto_cycle(T1: float, p1: float, compression_ratio: float, Q_in: float, 
               n: float = 1.0, gamma: float = 1.4, n_points: int = 50):
    """
    Otto-Kreisprozess (Benzinmotor):
    1→2: adiabatische Kompression
    2→3: isochore Wärmezufuhr
    3→4: adiabatische Expansion
    4→1: isochore Wärmeabfuhr
    """
    r = compression_ratio
    V1 = n * R_GAS * T1 / p1
    V2 = V1 / r
    
    # 1→2: adiabatisch
    T2 = T1 * r ** (gamma - 1)
    p2 = p1 * r ** gamma
    
    # 2→3: isochor (V = const), Wärmezufuhr
    # Q = n * Cv * (T3 - T2), Cv = R/(γ-1)
    Cv = R_GAS / (gamma - 1)
    T3 = T2 + Q_in / (n * Cv)
    p3 = p2 * T3 / T2
    
    # 3→4: adiabatisch
    T4 = T3 / r ** (gamma - 1)
    p4 = p3 / r ** gamma
    
    # Pfade
    V_12, p_12 = adiabatic_process(GasState(p1, V1, T1, n), V2, gamma, n_points)
    V_23, p_23 = isochoric_process(GasState(p2, V2, T2, n), p3, n_points)
    V_34, p_34 = adiabatic_process(GasState(p3, V2, T3, n), V1, gamma, n_points)
    V_41, p_41 = isochoric_process(GasState(p4, V1, T4, n), p1, n_points)
    
    V_cycle = np.concatenate([V_12, V_23, V_34, V_41])
    p_cycle = np.concatenate([p_12, p_23, p_34, p_41])
    
    # Wirkungsgrad Otto
    eta = 1 - 1 / r ** (gamma - 1)
    
    return V_cycle, p_cycle, eta


# ============================================================
# HEAT CONDUCTION (Wärmeleitung)
# ============================================================

def heat_conduction_1d_step(T: np.ndarray, alpha: float, dx: float, dt: float) -> np.ndarray:
    """Ein Zeitschritt der 1D-Wärmeleitungsgleichung (explizites Euler)"""
    T_new = T.copy()
    # Innere Punkte: ∂T/∂t = α * ∂²T/∂x²
    T_new[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
    return T_new


def heat_conduction_2d_step(T: np.ndarray, alpha: float, dx: float, dt: float) -> np.ndarray:
    """Ein Zeitschritt der 2D-Wärmeleitungsgleichung (explizites Euler)"""
    T_new = T.copy()
    # Innere Punkte: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y²)
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt / dx**2 * (
        T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - 4*T[1:-1, 1:-1]
    )
    return T_new


# ============================================================
# KINETIC GAS THEORY (Gaskinetik)
# ============================================================

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.02
    mass: float = 1.0


def init_particles(n: int, box_size: float = 1.0, T: float = 300.0, mass: float = 1.0) -> List[Particle]:
    """Initialisiere Teilchen mit Maxwell-Boltzmann-Verteilung"""
    particles = []
    # v_rms = sqrt(3 k_B T / m), hier vereinfacht skaliert
    v_scale = np.sqrt(T / 300.0) * 0.3  # relative Geschwindigkeit
    
    for _ in range(n):
        x = np.random.uniform(0.1, box_size - 0.1)
        y = np.random.uniform(0.1, box_size - 0.1)
        # Maxwell-Boltzmann: Geschwindigkeitskomponenten normalverteilt
        vx = np.random.normal(0, v_scale)
        vy = np.random.normal(0, v_scale)
        particles.append(Particle(x, y, vx, vy, radius=0.02, mass=mass))
    return particles


def update_particles(particles: List[Particle], dt: float, box_size: float = 1.0) -> List[Particle]:
    """Update Teilchenpositionen und handle Wandkollisionen"""
    for p in particles:
        # Position update
        p.x += p.vx * dt
        p.y += p.vy * dt
        
        # Wandkollisionen (elastisch)
        if p.x - p.radius < 0:
            p.x = p.radius
            p.vx = -p.vx
        elif p.x + p.radius > box_size:
            p.x = box_size - p.radius
            p.vx = -p.vx
            
        if p.y - p.radius < 0:
            p.y = p.radius
            p.vy = -p.vy
        elif p.y + p.radius > box_size:
            p.y = box_size - p.radius
            p.vy = -p.vy
    
    # Teilchen-Teilchen-Kollisionen (vereinfacht)
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            pi, pj = particles[i], particles[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < pi.radius + pj.radius and dist > 0:
                # Elastischer Stoß
                nx, ny = dx / dist, dy / dist
                dvx = pi.vx - pj.vx
                dvy = pi.vy - pj.vy
                dvn = dvx * nx + dvy * ny
                if dvn > 0:  # Teilchen nähern sich
                    pi.vx -= dvn * nx
                    pi.vy -= dvn * ny
                    pj.vx += dvn * nx
                    pj.vy += dvn * ny
    
    return particles


def compute_pressure(particles: List[Particle], box_size: float = 1.0) -> float:
    """Berechne Druck aus kinetischer Energie (2D)"""
    # p = (2/V) * E_kin für 2D ideales Gas
    E_kin = sum(0.5 * p.mass * (p.vx**2 + p.vy**2) for p in particles)
    V = box_size ** 2
    return 2 * E_kin / V


def compute_temperature(particles: List[Particle]) -> float:
    """Berechne Temperatur aus mittlerer kinetischer Energie"""
    # <E_kin> = (f/2) * k_B * T, f=2 für 2D
    E_kin_mean = np.mean([0.5 * p.mass * (p.vx**2 + p.vy**2) for p in particles])
    return E_kin_mean / k_B


# ============================================================
# STREAMLIT UI
# ============================================================

def render_thermo_tab():
    """Hauptfunktion für den Thermodynamik-Tab"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("🌡️ Thermodynamik & Wärmelehre", "🌡️ Thermodynamics & Heat Transfer"))
    
    # Sub-Tabs
    heat_tab, gas_tab, cycle_tab, kinetic_tab = st.tabs([
        tr("Wärmeleitung", "Heat Conduction"),
        tr("Zustandsänderungen", "State Changes"),
        tr("Kreisprozesse", "Thermodynamic Cycles"),
        tr("Gaskinetik", "Kinetic Gas Theory")
    ])
    
    with heat_tab:
        render_heat_conduction_tab()
    
    with gas_tab:
        render_state_changes_tab()
    
    with cycle_tab:
        render_cycles_tab()
    
    with kinetic_tab:
        render_kinetic_tab()


def render_heat_conduction_tab():
    """Wärmeleitung mit Echtzeit-Animation"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Wärmeleitung (Fourier-Gleichung)", "### Heat Conduction (Fourier Equation)"))
    st.latex(r"\frac{\partial T}{\partial t} = \alpha \nabla^2 T")
    
    dim_choice = st.radio(
        tr("Dimension", "Dimension"),
        ["1D", "2D"],
        horizontal=True,
        key="heat_dim"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider(
            tr("Wärmeleitfähigkeit α [m²/s]", "Thermal diffusivity α [m²/s]"),
            0.01, 1.0, 0.2, 0.01,
            key="heat_alpha"
        )
        n_steps = st.slider(
            tr("Animationsschritte", "Animation steps"),
            20, 200, 80, 10,
            key="heat_steps"
        )
    with col2:
        T_hot = st.slider(
            tr("Heiße Temperatur [°C]", "Hot temperature [°C]"),
            50, 200, 100, 10,
            key="heat_T_hot"
        )
        T_cold = st.slider(
            tr("Kalte Temperatur [°C]", "Cold temperature [°C]"),
            0, 50, 20, 5,
            key="heat_T_cold"
        )
    
    speed = st.slider(
        tr("Animationsgeschwindigkeit", "Animation speed"),
        1, 10, 5,
        key="heat_speed"
    )
    
    if st.button(tr("▶️ Animation starten", "▶️ Start animation"), key="heat_start", use_container_width=True):
        if dim_choice == "1D":
            run_heat_1d_animation(alpha, T_hot, T_cold, n_steps, speed)
        else:
            run_heat_2d_animation(alpha, T_hot, T_cold, n_steps, speed)


def run_heat_1d_animation(alpha: float, T_hot: float, T_cold: float, n_steps: int, speed: int):
    """1D Wärmeleitung Animation"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    N = 100
    dx = 1.0 / N
    dt = 0.4 * dx**2 / alpha  # CFL-Bedingung
    
    # Anfangsbedingung: links heiß, rechts kalt
    T = np.linspace(T_hot, T_cold, N)
    x = np.linspace(0, 1, N)
    
    # Animation mit st.empty()
    chart_placeholder = st.empty()
    info_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for step in range(n_steps):
        # Randbedingungen (Dirichlet)
        T[0] = T_hot
        T[-1] = T_cold
        
        # Update
        T = heat_conduction_1d_step(T, alpha, dx, dt)
        
        # Plot mit Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=T,
            mode='lines',
            fill='tozeroy',
            line=dict(color='red', width=3),
            fillcolor='rgba(255, 100, 100, 0.3)'
        ))
        fig.update_layout(
            title=tr(f"Temperaturverteilung (t = {step * dt:.4f} s)", 
                    f"Temperature distribution (t = {step * dt:.4f} s)"),
            xaxis_title=tr("Position x [m]", "Position x [m]"),
            yaxis_title=tr("Temperatur [°C]", "Temperature [°C]"),
            yaxis=dict(range=[T_cold - 5, T_hot + 5]),
            height=400
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        info_placeholder.info(tr(
            f"Schritt {step + 1}/{n_steps} | Mittlere Temperatur: {np.mean(T):.1f} °C",
            f"Step {step + 1}/{n_steps} | Mean temperature: {np.mean(T):.1f} °C"
        ))
        progress_bar.progress((step + 1) / n_steps)
        
        time.sleep(0.1 / speed)
    
    st.success(tr("✅ Animation abgeschlossen", "✅ Animation completed"))


def run_heat_2d_animation(alpha: float, T_hot: float, T_cold: float, n_steps: int, speed: int):
    """2D Wärmeleitung Animation"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    N = 50
    dx = 1.0 / N
    dt = 0.2 * dx**2 / alpha  # CFL-Bedingung (strenger für 2D)
    
    # Anfangsbedingung: heißer Punkt in der Mitte
    T = np.full((N, N), T_cold, dtype=float)
    center = N // 2
    T[center-3:center+3, center-3:center+3] = T_hot
    
    # Animation
    chart_placeholder = st.empty()
    info_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for step in range(n_steps):
        # Randbedingungen (kalt an allen Rändern)
        T[0, :] = T_cold
        T[-1, :] = T_cold
        T[:, 0] = T_cold
        T[:, -1] = T_cold
        
        # Update
        T = heat_conduction_2d_step(T, alpha, dx, dt)
        
        # Plot
        fig = go.Figure(data=go.Heatmap(
            z=T,
            colorscale='RdYlBu_r',
            zmin=T_cold,
            zmax=T_hot,
            colorbar=dict(title=tr("T [°C]", "T [°C]"))
        ))
        fig.update_layout(
            title=tr(f"2D Wärmeleitung (t = {step * dt:.4f} s)",
                    f"2D Heat conduction (t = {step * dt:.4f} s)"),
            xaxis_title="x",
            yaxis_title="y",
            height=450,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        info_placeholder.info(tr(
            f"Schritt {step + 1}/{n_steps} | Max: {T.max():.1f} °C | Min: {T.min():.1f} °C",
            f"Step {step + 1}/{n_steps} | Max: {T.max():.1f} °C | Min: {T.min():.1f} °C"
        ))
        progress_bar.progress((step + 1) / n_steps)
        
        time.sleep(0.1 / speed)
    
    st.success(tr("✅ Animation abgeschlossen", "✅ Animation completed"))


def render_state_changes_tab():
    """Zustandsänderungen idealer Gase"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Zustandsänderungen idealer Gase", "### State Changes of Ideal Gases"))
    st.latex(r"pV = nRT")
    
    col1, col2 = st.columns(2)
    with col1:
        process_type = st.selectbox(
            tr("Prozesstyp", "Process type"),
            [
                tr("Isotherm (T = const)", "Isothermal (T = const)"),
                tr("Isobar (p = const)", "Isobaric (p = const)"),
                tr("Isochor (V = const)", "Isochoric (V = const)"),
                tr("Adiabatisch (Q = 0)", "Adiabatic (Q = 0)")
            ],
            key="state_process"
        )
        T_start = st.slider(
            tr("Starttemperatur [K]", "Start temperature [K]"),
            200, 600, 300, 10,
            key="state_T"
        )
    with col2:
        p_start = st.slider(
            tr("Startdruck [kPa]", "Start pressure [kPa]"),
            50, 500, 100, 10,
            key="state_p"
        ) * 1000  # in Pa
        
        ratio = st.slider(
            tr("Volumenverhältnis V₂/V₁", "Volume ratio V₂/V₁"),
            0.5, 3.0, 2.0, 0.1,
            key="state_ratio"
        )
    
    if st.button(tr("📊 Prozess berechnen", "📊 Compute process"), key="state_run", use_container_width=True):
        state1 = GasState.from_pT(p_start, T_start)
        V_end = state1.V * ratio
        
        if "Isotherm" in process_type or "Isothermal" in process_type:
            V, p = isothermal_process(state1, V_end)
            title = tr("Isotherme Expansion/Kompression", "Isothermal Expansion/Compression")
            info = tr(f"T = {T_start} K = const", f"T = {T_start} K = const")
        elif "Isobar" in process_type or "Isobaric" in process_type:
            V, p = isobaric_process(state1, V_end)
            title = tr("Isobare Expansion/Kompression", "Isobaric Expansion/Compression")
            info = tr(f"p = {p_start/1000:.0f} kPa = const", f"p = {p_start/1000:.0f} kPa = const")
        elif "Isochor" in process_type or "Isochoric" in process_type:
            p_end = p_start * ratio
            V, p = isochoric_process(state1, p_end)
            title = tr("Isochore Erwärmung/Abkühlung", "Isochoric Heating/Cooling")
            info = tr(f"V = {state1.V*1000:.2f} L = const", f"V = {state1.V*1000:.2f} L = const")
        else:  # Adiabatisch
            V, p = adiabatic_process(state1, V_end, gamma=1.4)
            title = tr("Adiabatische Expansion/Kompression", "Adiabatic Expansion/Compression")
            info = tr("Q = 0 (keine Wärmeübertragung)", "Q = 0 (no heat transfer)")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=V * 1000,  # in Liter
            y=p / 1000,  # in kPa
            mode='lines',
            line=dict(color='blue', width=3),
            name=title
        ))
        fig.add_trace(go.Scatter(
            x=[V[0] * 1000],
            y=[p[0] / 1000],
            mode='markers',
            marker=dict(color='green', size=12),
            name=tr("Start", "Start")
        ))
        fig.add_trace(go.Scatter(
            x=[V[-1] * 1000],
            y=[p[-1] / 1000],
            mode='markers',
            marker=dict(color='red', size=12),
            name=tr("Ende", "End")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=tr("Volumen V [L]", "Volume V [L]"),
            yaxis_title=tr("Druck p [kPa]", "Pressure p [kPa]"),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info(info)
        
        # Arbeit berechnen
        W = -np.trapz(p, V)  # W = -∫p dV
        st.metric(
            tr("Verrichtete Arbeit W", "Work done W"),
            f"{W:.2f} J"
        )


def render_cycles_tab():
    """Kreisprozesse (Carnot, Otto)"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Thermodynamische Kreisprozesse", "### Thermodynamic Cycles"))
    
    cycle_type = st.selectbox(
        tr("Kreisprozess", "Cycle"),
        [tr("Carnot-Prozess", "Carnot Cycle"), tr("Otto-Prozess (Benzin)", "Otto Cycle (Gasoline)")],
        key="cycle_type"
    )
    
    if "Carnot" in cycle_type:
        st.latex(r"\eta_{Carnot} = 1 - \frac{T_{kalt}}{T_{heiß}}")
        
        col1, col2 = st.columns(2)
        with col1:
            T_hot = st.slider(tr("T_heiß [K]", "T_hot [K]"), 400, 800, 600, 10, key="carnot_Th")
            V1 = st.slider(tr("V₁ [L]", "V₁ [L]"), 0.5, 2.0, 1.0, 0.1, key="carnot_V1") / 1000
        with col2:
            T_cold = st.slider(tr("T_kalt [K]", "T_cold [K]"), 250, 400, 300, 10, key="carnot_Tc")
            V2 = st.slider(tr("V₂ [L]", "V₂ [L]"), 1.5, 5.0, 2.5, 0.1, key="carnot_V2") / 1000
        
        if st.button(tr("📊 Carnot-Prozess berechnen", "📊 Compute Carnot Cycle"), key="carnot_run", use_container_width=True):
            V, p, eta = carnot_cycle(T_hot, T_cold, V1, V2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=V * 1000, y=p / 1000,
                mode='lines',
                line=dict(color='purple', width=3),
                fill='toself',
                fillcolor='rgba(128, 0, 128, 0.2)',
                name='Carnot'
            ))
            fig.update_layout(
                title=tr("Carnot-Kreisprozess im pV-Diagramm", "Carnot Cycle in pV-Diagram"),
                xaxis_title=tr("Volumen V [L]", "Volume V [L]"),
                yaxis_title=tr("Druck p [kPa]", "Pressure p [kPa]"),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(tr("Wirkungsgrad η", "Efficiency η"), f"{eta*100:.1f} %")
            with col2:
                st.metric(tr("Theoretisches Maximum", "Theoretical maximum"), "η_Carnot")
    
    else:  # Otto
        st.latex(r"\eta_{Otto} = 1 - \frac{1}{r^{\gamma-1}}")
        
        col1, col2 = st.columns(2)
        with col1:
            compression = st.slider(tr("Verdichtungsverhältnis r", "Compression ratio r"), 6, 14, 10, 1, key="otto_r")
            T1 = st.slider(tr("Ansaugtemperatur [K]", "Intake temperature [K]"), 280, 350, 300, 5, key="otto_T1")
        with col2:
            p1 = st.slider(tr("Ansaugdruck [kPa]", "Intake pressure [kPa]"), 80, 120, 100, 5, key="otto_p1") * 1000
            Q_in = st.slider(tr("Wärmezufuhr [J]", "Heat input [J]"), 500, 3000, 1500, 100, key="otto_Q")
        
        if st.button(tr("📊 Otto-Prozess berechnen", "📊 Compute Otto Cycle"), key="otto_run", use_container_width=True):
            V, p, eta = otto_cycle(T1, p1, compression, Q_in)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=V * 1000, y=p / 1000,
                mode='lines',
                line=dict(color='orange', width=3),
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                name='Otto'
            ))
            fig.update_layout(
                title=tr("Otto-Kreisprozess im pV-Diagramm", "Otto Cycle in pV-Diagram"),
                xaxis_title=tr("Volumen V [L]", "Volume V [L]"),
                yaxis_title=tr("Druck p [kPa]", "Pressure p [kPa]"),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(tr("Wirkungsgrad η", "Efficiency η"), f"{eta*100:.1f} %")
            with col2:
                st.metric(tr("Verdichtung r", "Compression r"), f"{compression}:1")


def render_kinetic_tab():
    """Kinetische Gastheorie mit Animation"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Kinetische Gastheorie", "### Kinetic Gas Theory"))
    st.markdown(tr(
        "Simulation von Gasteilchen in einem Behälter. Druck entsteht durch Wandstöße.",
        "Simulation of gas particles in a container. Pressure arises from wall collisions."
    ))
    
    col1, col2 = st.columns(2)
    with col1:
        n_particles = st.slider(
            tr("Anzahl Teilchen", "Number of particles"),
            10, 100, 30, 5,
            key="kinetic_n"
        )
        n_frames = st.slider(
            tr("Animationsframes", "Animation frames"),
            50, 300, 150, 25,
            key="kinetic_frames"
        )
    with col2:
        temperature = st.slider(
            tr("Temperatur [K]", "Temperature [K]"),
            100, 1000, 300, 50,
            key="kinetic_T"
        )
        speed = st.slider(
            tr("Animationsgeschwindigkeit", "Animation speed"),
            1, 10, 5,
            key="kinetic_speed"
        )
    
    if st.button(tr("▶️ Gaskinetik starten", "▶️ Start gas kinetics"), key="kinetic_start", use_container_width=True):
        run_kinetic_animation(n_particles, n_frames, temperature, speed)


def run_kinetic_animation(n_particles: int, n_frames: int, temperature: float, speed: int):
    """Animierte Gaskinetik-Simulation"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    box_size = 1.0
    dt = 0.05
    
    # Initialisiere Teilchen
    particles = init_particles(n_particles, box_size, temperature)
    
    # Placeholders
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    pressures = []
    temperatures = []
    
    for frame in range(n_frames):
        # Update
        particles = update_particles(particles, dt, box_size)
        
        # Messungen
        p = compute_pressure(particles, box_size)
        pressures.append(p)
        
        # Positionen extrahieren
        x = [p.x for p in particles]
        y = [p.y for p in particles]
        
        # Farbe nach Geschwindigkeit
        speeds = [np.sqrt(p.vx**2 + p.vy**2) for p in particles]
        
        # Plot
        fig = go.Figure()
        
        # Teilchen
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=12,
                color=speeds,
                colorscale='Plasma',
                colorbar=dict(title=tr("v [a.u.]", "v [a.u.]")),
                line=dict(width=1, color='black')
            ),
            name=tr("Teilchen", "Particles")
        ))
        
        # Box
        fig.add_shape(type="rect", x0=0, y0=0, x1=box_size, y1=box_size,
                     line=dict(color="black", width=3))
        
        fig.update_layout(
            title=tr(f"Gaskinetik (Frame {frame + 1}/{n_frames})",
                    f"Gas Kinetics (Frame {frame + 1}/{n_frames})"),
            xaxis=dict(range=[-0.05, box_size + 0.05], title="x"),
            yaxis=dict(range=[-0.05, box_size + 0.05], title="y", scaleanchor="x"),
            height=450,
            showlegend=False
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Metriken
        col1, col2, col3 = metrics_placeholder.columns(3)
        with col1:
            st.metric(tr("Mittlere Geschw.", "Mean velocity"), f"{np.mean(speeds):.3f}")
        with col2:
            st.metric(tr("Druck (rel.)", "Pressure (rel.)"), f"{p:.4f}")
        with col3:
            st.metric(tr("E_kin (rel.)", "E_kin (rel.)"), 
                     f"{sum(0.5*(p.vx**2 + p.vy**2) for p in particles):.3f}")
        
        progress_bar.progress((frame + 1) / n_frames)
        time.sleep(0.05 / speed)
    
    # Finale Auswertung
    st.success(tr("✅ Simulation abgeschlossen", "✅ Simulation completed"))
    
    # Druckverlauf
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        y=pressures,
        mode='lines',
        line=dict(color='green'),
        name=tr("Druck", "Pressure")
    ))
    fig_p.update_layout(
        title=tr("Druckverlauf über Zeit", "Pressure over time"),
        xaxis_title=tr("Frame", "Frame"),
        yaxis_title=tr("Druck [a.u.]", "Pressure [a.u.]"),
        height=300
    )
    st.plotly_chart(fig_p, use_container_width=True)
    
    st.metric(
        tr("Mittlerer Druck", "Mean pressure"),
        f"{np.mean(pressures):.4f} ± {np.std(pressures):.4f}"
    )
