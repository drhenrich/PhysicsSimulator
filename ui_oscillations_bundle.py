"""
Schwingungen & Akustik Modul
============================
- Gekoppelte Oszillatoren und Schwebungen
- Stehende Wellen und Resonanz
- Doppler-Effekt mit bewegten Quellen
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# Physikalische Grundlagen
# =============================================================================

@dataclass
class Oscillator:
    """Harmonischer Oszillator"""
    mass: float = 1.0        # kg
    k: float = 10.0          # N/m (Federkonstante)
    x0: float = 1.0          # m (Anfangsauslenkung)
    v0: float = 0.0          # m/s (Anfangsgeschwindigkeit)
    damping: float = 0.0     # kg/s (Dämpfungskonstante)
    
    @property
    def omega0(self) -> float:
        """Eigenkreisfrequenz ω₀ = √(k/m)"""
        return np.sqrt(self.k / self.mass)
    
    @property
    def f0(self) -> float:
        """Eigenfrequenz f₀ = ω₀/(2π)"""
        return self.omega0 / (2 * np.pi)
    
    @property
    def T0(self) -> float:
        """Schwingungsdauer T₀ = 1/f₀"""
        return 1.0 / self.f0 if self.f0 > 0 else float('inf')
    
    @property
    def gamma(self) -> float:
        """Dämpfungskonstante γ = b/(2m)"""
        return self.damping / (2 * self.mass)
    
    @property
    def omega_d(self) -> float:
        """Gedämpfte Kreisfrequenz ω_d = √(ω₀² - γ²)"""
        diff = self.omega0**2 - self.gamma**2
        return np.sqrt(diff) if diff > 0 else 0.0


def harmonic_oscillator(osc: Oscillator, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Löst die Bewegungsgleichung eines gedämpften harmonischen Oszillators.
    
    m·ẍ + b·ẋ + k·x = 0
    
    Returns:
        x(t), v(t)
    """
    omega0 = osc.omega0
    gamma = osc.gamma
    x0, v0 = osc.x0, osc.v0
    
    if gamma < omega0:  # Unterdämpfung
        omega_d = osc.omega_d
        A = x0
        B = (v0 + gamma * x0) / omega_d if omega_d > 0 else 0
        
        x = np.exp(-gamma * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        v = np.exp(-gamma * t) * (
            -gamma * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t)) +
            omega_d * (-A * np.sin(omega_d * t) + B * np.cos(omega_d * t))
        )
    elif gamma == omega0:  # Kritische Dämpfung
        A = x0
        B = v0 + gamma * x0
        x = np.exp(-gamma * t) * (A + B * t)
        v = np.exp(-gamma * t) * (B - gamma * (A + B * t))
    else:  # Überdämpfung
        beta = np.sqrt(gamma**2 - omega0**2)
        c1 = (x0 * (gamma + beta) + v0) / (2 * beta)
        c2 = (x0 * (beta - gamma) - v0) / (2 * beta)
        
        x = c1 * np.exp((-gamma + beta) * t) + c2 * np.exp((-gamma - beta) * t)
        v = c1 * (-gamma + beta) * np.exp((-gamma + beta) * t) + \
            c2 * (-gamma - beta) * np.exp((-gamma - beta) * t)
    
    return x, v


def coupled_oscillators(m1: float, m2: float, k1: float, k2: float, k_c: float,
                        x1_0: float, x2_0: float, v1_0: float, v2_0: float,
                        t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwei gekoppelte Oszillatoren.
    
    m₁·ẍ₁ = -k₁·x₁ - k_c·(x₁ - x₂)
    m₂·ẍ₂ = -k₂·x₂ - k_c·(x₂ - x₁)
    
    Numerische Lösung mit Runge-Kutta 4.
    """
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    
    # Zustandsvektor: [x1, v1, x2, v2]
    y = np.zeros((len(t), 4))
    y[0] = [x1_0, v1_0, x2_0, v2_0]
    
    def derivatives(state):
        x1, v1, x2, v2 = state
        a1 = (-k1 * x1 - k_c * (x1 - x2)) / m1
        a2 = (-k2 * x2 - k_c * (x2 - x1)) / m2
        return np.array([v1, a1, v2, a2])
    
    # RK4 Integration
    for i in range(1, len(t)):
        k1_rk = derivatives(y[i-1])
        k2_rk = derivatives(y[i-1] + 0.5 * dt * k1_rk)
        k3_rk = derivatives(y[i-1] + 0.5 * dt * k2_rk)
        k4_rk = derivatives(y[i-1] + dt * k3_rk)
        y[i] = y[i-1] + (dt / 6) * (k1_rk + 2*k2_rk + 2*k3_rk + k4_rk)
    
    return y[:, 0], y[:, 2]  # x1(t), x2(t)


def beat_frequency(f1: float, f2: float) -> Tuple[float, float]:
    """
    Schwebungsfrequenz und Trägerfrequenz.
    
    f_beat = |f1 - f2|
    f_carrier = (f1 + f2) / 2
    """
    return abs(f1 - f2), (f1 + f2) / 2


def superposition(A1: float, f1: float, A2: float, f2: float, 
                  t: np.ndarray, phi1: float = 0, phi2: float = 0) -> np.ndarray:
    """
    Überlagerung zweier harmonischer Schwingungen.
    
    y(t) = A₁·sin(2πf₁t + φ₁) + A₂·sin(2πf₂t + φ₂)
    """
    y1 = A1 * np.sin(2 * np.pi * f1 * t + phi1)
    y2 = A2 * np.sin(2 * np.pi * f2 * t + phi2)
    return y1 + y2


def standing_wave(x: np.ndarray, t: float, A: float, k: float, omega: float,
                  n_modes: int = 1) -> np.ndarray:
    """
    Stehende Welle.
    
    y(x,t) = 2A·sin(kx)·cos(ωt)
    
    Für n-ten Modus in einem Rohr/Saite der Länge L:
    k_n = nπ/L, ω_n = n·ω₁
    """
    return 2 * A * np.sin(n_modes * k * x) * np.cos(n_modes * omega * t)


def resonance_amplitude(omega: float, omega0: float, gamma: float, 
                        F0: float, m: float) -> float:
    """
    Amplitude bei erzwungener Schwingung (Resonanzkurve).
    
    A(ω) = F₀/m / √[(ω₀² - ω²)² + (2γω)²]
    """
    denom = np.sqrt((omega0**2 - omega**2)**2 + (2 * gamma * omega)**2)
    return F0 / m / denom if denom > 0 else float('inf')


def resonance_phase(omega: float, omega0: float, gamma: float) -> float:
    """
    Phasenverschiebung bei erzwungener Schwingung.
    
    tan(φ) = 2γω / (ω₀² - ω²)
    """
    return np.arctan2(2 * gamma * omega, omega0**2 - omega**2)


def doppler_frequency(f_source: float, v_source: float, v_observer: float,
                      v_sound: float = 343.0, approaching: bool = True) -> float:
    """
    Doppler-Effekt für Schallwellen.
    
    f' = f · (v_sound ± v_observer) / (v_sound ∓ v_source)
    
    + wenn sich aufeinander zu bewegen
    - wenn sich voneinander entfernen
    """
    if approaching:
        f_observed = f_source * (v_sound + v_observer) / (v_sound - v_source)
    else:
        f_observed = f_source * (v_sound - v_observer) / (v_sound + v_source)
    
    return f_observed


def doppler_wavefronts(x_source: float, y_source: float, v_source: float,
                       f_source: float, v_sound: float, t_max: float,
                       n_fronts: int = 10) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Berechnet Wellenfronten für bewegte Schallquelle.
    
    Returns:
        Liste von (x_circle, y_circle, radius) für jede Wellenfront
    """
    T = 1.0 / f_source  # Periode
    fronts = []
    
    for i in range(n_fronts):
        t_emit = i * T
        if t_emit > t_max:
            break
        
        # Position der Quelle zum Emissionszeitpunkt
        x_emit = x_source + v_source * t_emit
        y_emit = y_source
        
        # Radius der Wellenfront zum aktuellen Zeitpunkt
        radius = v_sound * (t_max - t_emit)
        
        if radius > 0:
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = x_emit + radius * np.cos(theta)
            y_circle = y_emit + radius * np.sin(theta)
            fronts.append((x_circle, y_circle, radius))
    
    return fronts


# =============================================================================
# UI Rendering
# =============================================================================

def render_oscillations_tab():
    """Hauptfunktion für Schwingungen & Akustik Tab"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("🎵 Schwingungen & Akustik", "🎵 Oscillations & Acoustics"))
    
    osc_tab, beat_tab, standing_tab, doppler_tab = st.tabs([
        tr("Oszillatoren", "Oscillators"),
        tr("Schwebungen", "Beats"),
        tr("Stehende Wellen", "Standing Waves"),
        tr("Doppler-Effekt", "Doppler Effect")
    ])
    
    # =========================================================================
    # Tab 1: Oszillatoren
    # =========================================================================
    with osc_tab:
        st.markdown(tr(
            "**Harmonische und gekoppelte Oszillatoren**",
            "**Harmonic and coupled oscillators**"
        ))
        
        mode = st.radio(
            tr("Modus", "Mode"),
            [tr("Einzelner Oszillator", "Single oscillator"),
             tr("Gekoppelte Oszillatoren", "Coupled oscillators")],
            horizontal=True,
            key="osc_mode"
        )
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            if mode == tr("Einzelner Oszillator", "Single oscillator"):
                st.markdown(f"**{tr('Parameter', 'Parameters')}**")
                
                mass = st.number_input(
                    tr("Masse m [kg]", "Mass m [kg]"),
                    min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                    key="osc_mass"
                )
                k = st.number_input(
                    tr("Federkonstante k [N/m]", "Spring constant k [N/m]"),
                    min_value=0.1, max_value=100.0, value=10.0, step=1.0,
                    key="osc_k"
                )
                x0 = st.slider(
                    tr("Anfangsauslenkung x₀ [m]", "Initial displacement x₀ [m]"),
                    min_value=-2.0, max_value=2.0, value=1.0, step=0.1,
                    key="osc_x0"
                )
                v0 = st.slider(
                    tr("Anfangsgeschw. v₀ [m/s]", "Initial velocity v₀ [m/s]"),
                    min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                    key="osc_v0"
                )
                damping = st.slider(
                    tr("Dämpfung b [kg/s]", "Damping b [kg/s]"),
                    min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                    key="osc_damping"
                )
                
                n_periods = st.slider(
                    tr("Anzahl Perioden", "Number of periods"),
                    min_value=1, max_value=20, value=5,
                    key="osc_n_periods"
                )
                
                # Oszillator erstellen
                osc = Oscillator(mass=mass, k=k, x0=x0, v0=v0, damping=damping)
                
                st.markdown("---")
                st.markdown(f"**{tr('Kennwerte', 'Characteristics')}**")
                st.metric("ω₀", f"{osc.omega0:.2f} rad/s")
                st.metric("f₀", f"{osc.f0:.2f} Hz")
                st.metric("T₀", f"{osc.T0:.3f} s")
                if damping > 0:
                    st.metric("γ", f"{osc.gamma:.2f} s⁻¹")
                    if osc.omega_d > 0:
                        st.metric("ω_d", f"{osc.omega_d:.2f} rad/s")
                
            else:  # Gekoppelte Oszillatoren
                st.markdown(f"**{tr('Oszillator 1', 'Oscillator 1')}**")
                m1 = st.number_input("m₁ [kg]", 0.1, 10.0, 1.0, 0.1, key="osc_m1")
                k1 = st.number_input("k₁ [N/m]", 0.1, 100.0, 10.0, 1.0, key="osc_k1")
                x1_0 = st.slider("x₁(0) [m]", -2.0, 2.0, 1.0, 0.1, key="osc_x1_0")
                
                st.markdown(f"**{tr('Oszillator 2', 'Oscillator 2')}**")
                m2 = st.number_input("m₂ [kg]", 0.1, 10.0, 1.0, 0.1, key="osc_m2")
                k2 = st.number_input("k₂ [N/m]", 0.1, 100.0, 10.0, 1.0, key="osc_k2")
                x2_0 = st.slider("x₂(0) [m]", -2.0, 2.0, 0.0, 0.1, key="osc_x2_0")
                
                st.markdown(f"**{tr('Kopplung', 'Coupling')}**")
                k_c = st.slider("k_c [N/m]", 0.0, 50.0, 5.0, 0.5, key="osc_kc")
                
                n_periods = st.slider(
                    tr("Anzahl Perioden", "Number of periods"),
                    min_value=1, max_value=30, value=10,
                    key="osc_n_periods_coupled"
                )
        
        with col_plot:
            if mode == tr("Einzelner Oszillator", "Single oscillator"):
                t_max = n_periods * osc.T0
                t = np.linspace(0, t_max, 1000)
                x, v = harmonic_oscillator(osc, t)
                
                # Energie berechnen
                E_kin = 0.5 * osc.mass * v**2
                E_pot = 0.5 * osc.k * x**2
                E_total = E_kin + E_pot
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        tr("Auslenkung x(t) und Geschwindigkeit v(t)", 
                           "Displacement x(t) and velocity v(t)"),
                        tr("Energie", "Energy")
                    ),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(go.Scatter(
                    x=t, y=x, mode='lines', name='x(t)',
                    line=dict(color='#3182ce', width=2)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=t, y=v, mode='lines', name='v(t)',
                    line=dict(color='#e53e3e', width=2, dash='dash')
                ), row=1, col=1)
                
                # Einhüllende bei Dämpfung
                if damping > 0:
                    envelope = osc.x0 * np.exp(-osc.gamma * t)
                    fig.add_trace(go.Scatter(
                        x=t, y=envelope, mode='lines', 
                        name=tr('Einhüllende', 'Envelope'),
                        line=dict(color='gray', width=1, dash='dot')
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=t, y=-envelope, mode='lines', showlegend=False,
                        line=dict(color='gray', width=1, dash='dot')
                    ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=t, y=E_kin, mode='lines', name='E_kin',
                    line=dict(color='#dd6b20', width=2),
                    fill='tozeroy', fillcolor='rgba(221, 107, 32, 0.3)'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=t, y=E_pot, mode='lines', name='E_pot',
                    line=dict(color='#38a169', width=2),
                    fill='tozeroy', fillcolor='rgba(56, 161, 105, 0.3)'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=t, y=E_total, mode='lines', name='E_total',
                    line=dict(color='#805ad5', width=2, dash='dash')
                ), row=2, col=1)
                
                fig.update_xaxes(title_text="t [s]", row=1, col=1)
                fig.update_xaxes(title_text="t [s]", row=2, col=1)
                fig.update_yaxes(title_text=tr("x [m], v [m/s]", "x [m], v [m/s]"), row=1, col=1)
                fig.update_yaxes(title_text="E [J]", row=2, col=1)
                
                fig.update_layout(height=550, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Gekoppelte Oszillatoren
                # Eigenfrequenzen schätzen
                omega1 = np.sqrt(k1 / m1)
                T1 = 2 * np.pi / omega1
                t_max = n_periods * T1
                t = np.linspace(0, t_max, 2000)
                
                x1, x2 = coupled_oscillators(m1, m2, k1, k2, k_c, x1_0, x2_0, 0, 0, t)
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        tr("Auslenkungen x₁(t) und x₂(t)", "Displacements x₁(t) and x₂(t)"),
                        tr("Phasenraum", "Phase space")
                    ),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(go.Scatter(
                    x=t, y=x1, mode='lines', name='x₁(t)',
                    line=dict(color='#3182ce', width=2)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=t, y=x2, mode='lines', name='x₂(t)',
                    line=dict(color='#e53e3e', width=2)
                ), row=1, col=1)
                
                # Schwerpunkt und Relativbewegung
                x_cm = (m1 * x1 + m2 * x2) / (m1 + m2)
                x_rel = x1 - x2
                
                fig.add_trace(go.Scatter(
                    x=x1, y=x2, mode='lines', name=tr('Trajektorie', 'Trajectory'),
                    line=dict(color='#805ad5', width=1)
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=[x1[0]], y=[x2[0]], mode='markers', name='Start',
                    marker=dict(color='green', size=10)
                ), row=2, col=1)
                
                fig.update_xaxes(title_text="t [s]", row=1, col=1)
                fig.update_xaxes(title_text="x₁ [m]", row=2, col=1)
                fig.update_yaxes(title_text="x [m]", row=1, col=1)
                fig.update_yaxes(title_text="x₂ [m]", scaleanchor="x2", row=2, col=1)
                
                fig.update_layout(height=550)
                st.plotly_chart(fig, use_container_width=True)
                
                # Normalmoden Info
                st.caption(tr(
                    "💡 Bei gleichen Massen und Federn gibt es zwei Normalmoden: "
                    "Gleichtakt (beide in Phase) und Gegentakt (gegenphasig).",
                    "💡 For equal masses and springs, there are two normal modes: "
                    "symmetric (in phase) and antisymmetric (out of phase)."
                ))
    
    # =========================================================================
    # Tab 2: Schwebungen
    # =========================================================================
    with beat_tab:
        st.markdown(tr(
            "**Schwebungen** — Überlagerung zweier Schwingungen mit nahen Frequenzen",
            "**Beats** — Superposition of two oscillations with close frequencies"
        ))
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            st.markdown(f"**{tr('Schwingung 1', 'Oscillation 1')}**")
            A1 = st.slider("A₁", 0.1, 2.0, 1.0, 0.1, key="beat_A1")
            f1 = st.slider("f₁ [Hz]", 1.0, 20.0, 10.0, 0.1, key="beat_f1")
            
            st.markdown(f"**{tr('Schwingung 2', 'Oscillation 2')}**")
            A2 = st.slider("A₂", 0.1, 2.0, 1.0, 0.1, key="beat_A2")
            f2 = st.slider("f₂ [Hz]", 1.0, 20.0, 11.0, 0.1, key="beat_f2")
            
            phi2 = st.slider(
                tr("Phasendifferenz φ₂ [°]", "Phase difference φ₂ [°]"),
                0, 360, 0, 15,
                key="beat_phi"
            )
            
            t_duration = st.slider(
                tr("Dauer [s]", "Duration [s]"),
                0.5, 5.0, 2.0, 0.1,
                key="beat_duration"
            )
            
            st.markdown("---")
            f_beat, f_carrier = beat_frequency(f1, f2)
            st.metric(tr("Schwebungsfrequenz", "Beat frequency"), f"{f_beat:.2f} Hz")
            st.metric(tr("Trägerfrequenz", "Carrier frequency"), f"{f_carrier:.2f} Hz")
            if f_beat > 0:
                st.metric(tr("Schwebungsperiode", "Beat period"), f"{1/f_beat:.3f} s")
        
        with col_plot:
            t = np.linspace(0, t_duration, 2000)
            phi2_rad = np.deg2rad(phi2)
            
            y1 = A1 * np.sin(2 * np.pi * f1 * t)
            y2 = A2 * np.sin(2 * np.pi * f2 * t + phi2_rad)
            y_sum = y1 + y2
            
            # Einhüllende der Schwebung
            if abs(f1 - f2) > 0.01:
                envelope = 2 * min(A1, A2) * np.abs(np.cos(np.pi * (f1 - f2) * t))
            else:
                envelope = np.ones_like(t) * (A1 + A2)
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    tr("Einzelschwingungen", "Individual oscillations"),
                    tr("Überlagerung (Schwebung)", "Superposition (beats)"),
                    tr("Frequenzspektrum", "Frequency spectrum")
                ),
                vertical_spacing=0.1
            )
            
            fig.add_trace(go.Scatter(
                x=t, y=y1, mode='lines', name=f'f₁={f1} Hz',
                line=dict(color='#3182ce', width=1.5)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=t, y=y2, mode='lines', name=f'f₂={f2} Hz',
                line=dict(color='#e53e3e', width=1.5)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=t, y=y_sum, mode='lines', name=tr('Summe', 'Sum'),
                line=dict(color='#805ad5', width=2)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=t, y=envelope, mode='lines', name=tr('Einhüllende', 'Envelope'),
                line=dict(color='#dd6b20', width=1.5, dash='dash')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=t, y=-envelope, mode='lines', showlegend=False,
                line=dict(color='#dd6b20', width=1.5, dash='dash')
            ), row=2, col=1)
            
            # FFT für Spektrum
            n = len(t)
            freq = np.fft.rfftfreq(n, t[1] - t[0])
            fft_sum = np.abs(np.fft.rfft(y_sum)) / n * 2
            
            fig.add_trace(go.Scatter(
                x=freq, y=fft_sum, mode='lines', name='|FFT|',
                line=dict(color='#38a169', width=2),
                fill='tozeroy', fillcolor='rgba(56, 161, 105, 0.3)'
            ), row=3, col=1)
            
            fig.update_xaxes(title_text="t [s]", row=1, col=1)
            fig.update_xaxes(title_text="t [s]", row=2, col=1)
            fig.update_xaxes(title_text="f [Hz]", range=[0, max(f1, f2) * 2], row=3, col=1)
            fig.update_yaxes(title_text="y", row=1, col=1)
            fig.update_yaxes(title_text="y", row=2, col=1)
            fig.update_yaxes(title_text=tr("Amplitude", "Amplitude"), row=3, col=1)
            
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # Tab 3: Stehende Wellen
    # =========================================================================
    with standing_tab:
        st.markdown(tr(
            "**Stehende Wellen und Resonanz**",
            "**Standing waves and resonance**"
        ))
        
        sub_tab1, sub_tab2 = st.tabs([
            tr("Stehende Welle", "Standing wave"),
            tr("Resonanzkurve", "Resonance curve")
        ])
        
        with sub_tab1:
            col_params, col_plot = st.columns([1, 2])
            
            with col_params:
                L = st.slider(
                    tr("Länge L [m]", "Length L [m]"),
                    0.5, 5.0, 2.0, 0.1,
                    key="sw_L"
                )
                
                n_mode = st.selectbox(
                    tr("Harmonische n", "Harmonic n"),
                    [1, 2, 3, 4, 5, 6],
                    key="sw_n"
                )
                
                A_wave = st.slider(
                    tr("Amplitude A [m]", "Amplitude A [m]"),
                    0.1, 1.0, 0.5, 0.05,
                    key="sw_A"
                )
                
                v_wave = st.number_input(
                    tr("Wellengeschw. v [m/s]", "Wave velocity v [m/s]"),
                    10.0, 1000.0, 343.0, 10.0,
                    key="sw_v"
                )
                
                # Animation
                animate = st.checkbox(
                    tr("Animation", "Animation"),
                    value=True,
                    key="sw_animate"
                )
                
                if animate:
                    t_anim = st.slider(
                        tr("Zeit t", "Time t"),
                        0.0, 1.0, 0.0, 0.01,
                        key="sw_t"
                    )
                else:
                    t_anim = 0.0
                
                st.markdown("---")
                
                # Berechne Kennwerte
                lambda_n = 2 * L / n_mode
                f_n = v_wave / lambda_n
                omega_n = 2 * np.pi * f_n
                k_n = 2 * np.pi / lambda_n
                
                st.metric(f"λ_{n_mode}", f"{lambda_n:.3f} m")
                st.metric(f"f_{n_mode}", f"{f_n:.2f} Hz")
            
            with col_plot:
                x = np.linspace(0, L, 500)
                
                # Stehende Welle
                omega = 2 * np.pi * f_n
                k = n_mode * np.pi / L
                
                # Zeitlicher Verlauf
                t_period = 1 / f_n
                t = t_anim * t_period
                
                y = 2 * A_wave * np.sin(k * x) * np.cos(omega * t)
                
                # Knoten und Bäuche
                nodes_x = [i * L / n_mode for i in range(n_mode + 1)]
                antinodes_x = [(i + 0.5) * L / n_mode for i in range(n_mode)]
                
                fig = go.Figure()
                
                # Welle
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines', name=tr('Welle', 'Wave'),
                    line=dict(color='#3182ce', width=3),
                    fill='tozeroy', fillcolor='rgba(49, 130, 206, 0.2)'
                ))
                
                # Einhüllende (max Amplitude)
                y_envelope = 2 * A_wave * np.abs(np.sin(k * x))
                fig.add_trace(go.Scatter(
                    x=x, y=y_envelope, mode='lines', 
                    name=tr('Einhüllende', 'Envelope'),
                    line=dict(color='#e53e3e', width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=-y_envelope, mode='lines', showlegend=False,
                    line=dict(color='#e53e3e', width=1, dash='dash')
                ))
                
                # Knoten markieren
                fig.add_trace(go.Scatter(
                    x=nodes_x, y=[0] * len(nodes_x),
                    mode='markers', name=tr('Knoten', 'Nodes'),
                    marker=dict(color='red', size=12, symbol='x')
                ))
                
                # Bäuche markieren
                antinodes_y = [2 * A_wave * np.sin(k * ax) * np.cos(omega * t) 
                               for ax in antinodes_x]
                fig.add_trace(go.Scatter(
                    x=antinodes_x, y=antinodes_y,
                    mode='markers', name=tr('Bäuche', 'Antinodes'),
                    marker=dict(color='green', size=12, symbol='diamond')
                ))
                
                fig.update_layout(
                    title=tr(f"Stehende Welle - {n_mode}. Harmonische",
                            f"Standing wave - {n_mode}th harmonic"),
                    xaxis_title="x [m]",
                    yaxis_title="y [m]",
                    yaxis=dict(range=[-2.5 * A_wave, 2.5 * A_wave]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Alle Moden anzeigen
                st.markdown(f"**{tr('Alle Harmonischen', 'All harmonics')}**")
                
                fig_modes = go.Figure()
                colors = ['#3182ce', '#e53e3e', '#38a169', '#dd6b20', '#805ad5', '#d53f8c']
                
                for n in range(1, 7):
                    k_n = n * np.pi / L
                    y_n = np.sin(k_n * x)
                    fig_modes.add_trace(go.Scatter(
                        x=x, y=y_n + 2.5 * (6 - n),
                        mode='lines', name=f'n={n}',
                        line=dict(color=colors[n-1], width=2)
                    ))
                
                fig_modes.update_layout(
                    xaxis_title="x [m]",
                    yaxis_title=tr("Modus (versetzt)", "Mode (offset)"),
                    height=350,
                    showlegend=True
                )
                
                st.plotly_chart(fig_modes, use_container_width=True)
        
        with sub_tab2:
            col_params, col_plot = st.columns([1, 2])
            
            with col_params:
                st.markdown(f"**{tr('Oszillator', 'Oscillator')}**")
                
                omega0_res = st.slider(
                    "ω₀ [rad/s]", 1.0, 20.0, 10.0, 0.5,
                    key="res_omega0"
                )
                
                gamma_res = st.slider(
                    "γ [s⁻¹]", 0.1, 5.0, 0.5, 0.1,
                    key="res_gamma"
                )
                
                F0 = st.slider(
                    "F₀ [N]", 0.1, 10.0, 1.0, 0.1,
                    key="res_F0"
                )
                
                m_res = st.slider(
                    "m [kg]", 0.1, 5.0, 1.0, 0.1,
                    key="res_m"
                )
                
                st.markdown("---")
                
                # Qualitätsfaktor
                Q = omega0_res / (2 * gamma_res)
                st.metric(tr("Gütefaktor Q", "Quality factor Q"), f"{Q:.1f}")
                
                # Resonanzfrequenz
                omega_res = np.sqrt(omega0_res**2 - 2 * gamma_res**2)
                if omega_res > 0:
                    st.metric("ω_res", f"{omega_res:.2f} rad/s")
                
                # Max Amplitude
                A_max = F0 / (2 * m_res * gamma_res * omega0_res)
                st.metric("A_max", f"{A_max:.3f} m")
            
            with col_plot:
                omega = np.linspace(0.1, 2.5 * omega0_res, 500)
                
                # Amplitudenkurve
                A_omega = np.array([resonance_amplitude(w, omega0_res, gamma_res, F0, m_res) 
                                   for w in omega])
                
                # Phasenkurve
                phi_omega = np.array([resonance_phase(w, omega0_res, gamma_res) 
                                     for w in omega])
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        tr("Amplitudenresonanz A(ω)", "Amplitude resonance A(ω)"),
                        tr("Phasenverschiebung φ(ω)", "Phase shift φ(ω)")
                    ),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(go.Scatter(
                    x=omega, y=A_omega, mode='lines', name='A(ω)',
                    line=dict(color='#3182ce', width=3),
                    fill='tozeroy', fillcolor='rgba(49, 130, 206, 0.2)'
                ), row=1, col=1)
                
                # Resonanzfrequenz markieren
                fig.add_vline(
                    x=omega0_res, row=1, col=1,
                    line=dict(color='red', dash='dash', width=1),
                    annotation_text="ω₀"
                )
                
                # Halbwertsbreite
                A_half = A_max / np.sqrt(2)
                fig.add_hline(
                    y=A_half, row=1, col=1,
                    line=dict(color='orange', dash='dot', width=1),
                    annotation_text="A_max/√2"
                )
                
                fig.add_trace(go.Scatter(
                    x=omega, y=np.degrees(phi_omega), mode='lines', name='φ(ω)',
                    line=dict(color='#e53e3e', width=3)
                ), row=2, col=1)
                
                fig.add_vline(
                    x=omega0_res, row=2, col=1,
                    line=dict(color='red', dash='dash', width=1)
                )
                
                fig.add_hline(
                    y=-90, row=2, col=1,
                    line=dict(color='gray', dash='dot', width=1),
                    annotation_text="-90°"
                )
                
                fig.update_xaxes(title_text="ω [rad/s]", row=1, col=1)
                fig.update_xaxes(title_text="ω [rad/s]", row=2, col=1)
                fig.update_yaxes(title_text="A [m]", row=1, col=1)
                fig.update_yaxes(title_text="φ [°]", row=2, col=1)
                
                fig.update_layout(height=550, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # Tab 4: Doppler-Effekt
    # =========================================================================
    with doppler_tab:
        st.markdown(tr(
            "**Doppler-Effekt** — Frequenzverschiebung durch Bewegung",
            "**Doppler effect** — Frequency shift due to motion"
        ))
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            st.markdown(f"**{tr('Schallquelle', 'Sound source')}**")
            
            f_source = st.slider(
                tr("Frequenz f₀ [Hz]", "Frequency f₀ [Hz]"),
                100, 1000, 440, 10,
                key="dop_f"
            )
            
            v_source = st.slider(
                tr("Geschwindigkeit v_s [m/s]", "Velocity v_s [m/s]"),
                -150, 150, 50, 5,
                key="dop_vs"
            )
            
            st.markdown(f"**{tr('Beobachter', 'Observer')}**")
            
            v_observer = st.slider(
                tr("Geschwindigkeit v_o [m/s]", "Velocity v_o [m/s]"),
                -50, 50, 0, 5,
                key="dop_vo"
            )
            
            st.markdown(f"**{tr('Medium', 'Medium')}**")
            
            v_sound = st.number_input(
                tr("Schallgeschw. c [m/s]", "Sound speed c [m/s]"),
                100.0, 1500.0, 343.0, 10.0,
                key="dop_c"
            )
            
            st.markdown("---")
            
            # Doppler-Frequenzen berechnen
            if abs(v_source) < v_sound:
                f_approaching = doppler_frequency(f_source, v_source, v_observer, v_sound, True)
                f_receding = doppler_frequency(f_source, v_source, v_observer, v_sound, False)
                
                st.metric(tr("f (nähert sich)", "f (approaching)"), f"{f_approaching:.1f} Hz")
                st.metric(tr("f (entfernt sich)", "f (receding)"), f"{f_receding:.1f} Hz")
                
                # Mach-Zahl
                mach = abs(v_source) / v_sound
                st.metric(tr("Mach-Zahl", "Mach number"), f"{mach:.2f}")
            else:
                st.warning(tr(
                    "⚠️ Überschallgeschwindigkeit! Machkegel entsteht.",
                    "⚠️ Supersonic speed! Mach cone forms."
                ))
                mach = abs(v_source) / v_sound
                theta_mach = np.degrees(np.arcsin(1 / mach))
                st.metric(tr("Mach-Zahl", "Mach number"), f"{mach:.2f}")
                st.metric(tr("Machwinkel", "Mach angle"), f"{theta_mach:.1f}°")
        
        with col_plot:
            # Animationszeit
            t_anim = st.slider(
                tr("Zeit t [s]", "Time t [s]"),
                0.0, 2.0, 1.0, 0.05,
                key="dop_t"
            )
            
            # Wellenfronten berechnen
            x_source_start = -3.0  # Startposition
            x_source = x_source_start + v_source * t_anim
            
            fronts = doppler_wavefronts(
                x_source_start, 0.0, v_source, f_source, v_sound, t_anim, n_fronts=15
            )
            
            fig = go.Figure()
            
            # Wellenfronten zeichnen
            for i, (x_circle, y_circle, radius) in enumerate(fronts):
                opacity = 0.8 - 0.5 * (i / len(fronts))
                fig.add_trace(go.Scatter(
                    x=x_circle, y=y_circle, mode='lines',
                    line=dict(color=f'rgba(49, 130, 206, {opacity})', width=1.5),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Quelle zeichnen
            fig.add_trace(go.Scatter(
                x=[x_source], y=[0],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='circle'),
                text=['🔊'],
                textposition='top center',
                name=tr('Quelle', 'Source')
            ))
            
            # Bewegungspfeil
            if v_source != 0:
                arrow_len = 1.0 if v_source > 0 else -1.0
                fig.add_annotation(
                    x=x_source, y=0,
                    ax=x_source + arrow_len, ay=0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='red'
                )
            
            # Beobachter
            x_obs = 5.0 + v_observer * t_anim
            fig.add_trace(go.Scatter(
                x=[x_obs], y=[0],
                mode='markers+text',
                marker=dict(size=15, color='green', symbol='triangle-up'),
                text=['👂'],
                textposition='top center',
                name=tr('Beobachter', 'Observer')
            ))
            
            # Mach-Kegel bei Überschall
            if abs(v_source) >= v_sound and v_source > 0:
                theta = np.arcsin(v_sound / v_source)
                cone_len = 10
                fig.add_trace(go.Scatter(
                    x=[x_source, x_source - cone_len],
                    y=[0, cone_len * np.tan(theta)],
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash'),
                    name='Mach-Kegel'
                ))
                fig.add_trace(go.Scatter(
                    x=[x_source, x_source - cone_len],
                    y=[0, -cone_len * np.tan(theta)],
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash'),
                    showlegend=False
                ))
            
            fig.update_layout(
                title=tr("Doppler-Effekt Visualisierung", "Doppler Effect Visualization"),
                xaxis_title="x [m]",
                yaxis_title="y [m]",
                xaxis=dict(range=[-8, 10]),
                yaxis=dict(range=[-6, 6], scaleanchor="x", scaleratio=1),
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Frequenz-Zeit-Diagramm
            st.markdown(f"**{tr('Frequenzverlauf für Vorbeifahrt', 'Frequency during pass-by')}**")
            
            t_pass = np.linspace(-2, 2, 500)
            x_source_t = v_source * t_pass
            x_obs_fixed = 5.0
            
            # Abstand Quelle-Beobachter
            r_t = np.sqrt((x_obs_fixed - x_source_t)**2 + 0.5**2)  # 0.5m seitlicher Abstand
            
            # Radialgeschwindigkeit
            v_radial = v_source * (x_source_t - x_obs_fixed) / r_t
            
            # Doppler-Frequenz
            f_doppler = f_source * v_sound / (v_sound + v_radial)
            f_doppler = np.clip(f_doppler, 0, 5 * f_source)
            
            fig_freq = go.Figure()
            
            fig_freq.add_trace(go.Scatter(
                x=t_pass, y=f_doppler,
                mode='lines',
                line=dict(color='#3182ce', width=3),
                name='f(t)'
            ))
            
            fig_freq.add_hline(
                y=f_source,
                line=dict(color='gray', dash='dash'),
                annotation_text=f"f₀ = {f_source} Hz"
            )
            
            fig_freq.update_layout(
                xaxis_title="t [s]",
                yaxis_title="f [Hz]",
                height=300
            )
            
            st.plotly_chart(fig_freq, use_container_width=True)
            
            st.caption(tr(
                "💡 Beim Vorbeiffahren fällt die Frequenz von f_high auf f_low ab "
                "(typischer 'Sireneneffekt').",
                "💡 During pass-by, frequency drops from f_high to f_low "
                "(typical 'siren effect')."
            ))
