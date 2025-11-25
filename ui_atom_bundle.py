"""
Atomphysik-Modul
- Bohr-Modell mit Elektronenübergängen und Spektrallinien
- Photoeffekt-Simulation
- Franck-Hertz-Experiment
- Emissions- und Absorptionsspektren
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ============================================================
# PHYSICS CONSTANTS
# ============================================================

# Fundamental constants
h = 6.62607015e-34      # Planck constant [J·s]
h_eV = 4.135667696e-15  # Planck constant [eV·s]
c = 299792458.0         # Speed of light [m/s]
e = 1.602176634e-19     # Elementary charge [C]
m_e = 9.1093837015e-31  # Electron mass [kg]
k_B = 1.380649e-23      # Boltzmann constant [J/K]
R_inf = 1.097373e7      # Rydberg constant [1/m]
E_H = 13.605693122      # Hydrogen ionization energy [eV]
a_0 = 5.29177210903e-11 # Bohr radius [m]


# ============================================================
# BOHR MODEL
# ============================================================

def bohr_energy(n: int, Z: int = 1) -> float:
    """
    Energie des n-ten Niveaus im Bohr-Modell [eV]
    E_n = -13.6 * Z² / n²
    """
    return -E_H * Z**2 / n**2


def bohr_radius(n: int, Z: int = 1) -> float:
    """
    Bahnradius des n-ten Niveaus [m]
    r_n = a_0 * n² / Z
    """
    return a_0 * n**2 / Z


def transition_wavelength(n_high: int, n_low: int, Z: int = 1) -> float:
    """
    Wellenlänge des emittierten Photons bei Übergang n_high → n_low [nm]
    """
    E_high = bohr_energy(n_high, Z)
    E_low = bohr_energy(n_low, Z)
    delta_E = E_high - E_low  # negativ, da E_high > E_low (weniger negativ)
    
    if delta_E >= 0:
        return float('inf')  # Absorption, kein Photon emittiert
    
    # E = h*c/λ => λ = h*c/|ΔE|
    wavelength_m = h * c / (abs(delta_E) * e)
    return wavelength_m * 1e9  # in nm


def wavelength_to_rgb(wavelength_nm: float) -> Tuple[int, int, int]:
    """
    Konvertiere Wellenlänge zu RGB-Farbe (sichtbares Spektrum)
    """
    if wavelength_nm < 380:
        return (138, 43, 226)  # UV → Violett
    elif wavelength_nm < 440:
        # Violett
        t = (wavelength_nm - 380) / (440 - 380)
        return (int(138 + (75 - 138) * t), int(43 + (0 - 43) * t), int(226 + (130 - 226) * t))
    elif wavelength_nm < 490:
        # Blau
        t = (wavelength_nm - 440) / (490 - 440)
        return (0, int(255 * t), 255)
    elif wavelength_nm < 510:
        # Cyan
        t = (wavelength_nm - 490) / (510 - 490)
        return (0, 255, int(255 * (1 - t)))
    elif wavelength_nm < 580:
        # Grün → Gelb
        t = (wavelength_nm - 510) / (580 - 510)
        return (int(255 * t), 255, 0)
    elif wavelength_nm < 645:
        # Orange
        t = (wavelength_nm - 580) / (645 - 580)
        return (255, int(255 * (1 - t)), 0)
    elif wavelength_nm < 780:
        # Rot
        return (255, 0, 0)
    else:
        return (139, 0, 0)  # IR → Dunkelrot


def get_spectral_series() -> Dict[str, Tuple[int, str]]:
    """Spektralserien für Wasserstoff"""
    return {
        "Lyman": (1, "UV (< 122 nm)"),
        "Balmer": (2, "Sichtbar (383-656 nm)"),
        "Paschen": (3, "Nahes IR (820-1875 nm)"),
        "Brackett": (4, "IR (1458-4051 nm)"),
        "Pfund": (5, "Fernes IR"),
    }


# ============================================================
# PHOTOELECTRIC EFFECT
# ============================================================

@dataclass
class Material:
    name: str
    work_function_eV: float  # Austrittsarbeit [eV]
    color: str


MATERIALS = {
    "Cäsium (Cs)": Material("Cäsium", 1.95, "#FFD700"),
    "Kalium (K)": Material("Kalium", 2.30, "#C0C0C0"),
    "Natrium (Na)": Material("Natrium", 2.75, "#B8860B"),
    "Zink (Zn)": Material("Zink", 4.33, "#708090"),
    "Kupfer (Cu)": Material("Kupfer", 4.65, "#B87333"),
    "Silber (Ag)": Material("Silber", 4.73, "#C0C0C0"),
    "Platin (Pt)": Material("Platin", 5.65, "#E5E4E2"),
}


def photoelectric_kinetic_energy(wavelength_nm: float, work_function_eV: float) -> float:
    """
    Kinetische Energie der Photoelektronen [eV]
    E_kin = h*f - W = h*c/λ - W
    """
    photon_energy_eV = h_eV * c / (wavelength_nm * 1e-9)
    E_kin = photon_energy_eV - work_function_eV
    return max(0.0, E_kin)


def threshold_wavelength(work_function_eV: float) -> float:
    """Grenzwellenlänge für Photoeffekt [nm]"""
    return h_eV * c / work_function_eV * 1e9


def stopping_potential(E_kin_eV: float) -> float:
    """Gegenfeldmethode: Bremsspannung [V]"""
    return E_kin_eV  # U_stop = E_kin / e


# ============================================================
# FRANCK-HERTZ EXPERIMENT
# ============================================================

def franck_hertz_current(U_acc: np.ndarray, U_excitation: float = 4.9, 
                         amplitude: float = 1.0, damping: float = 0.1) -> np.ndarray:
    """
    Simulierte Franck-Hertz-Kurve (Quecksilber)
    U_excitation ≈ 4.9 V für Hg (entspricht 253.7 nm UV-Linie)
    """
    I = np.zeros_like(U_acc)
    
    for i, U in enumerate(U_acc):
        if U <= 0:
            I[i] = 0
            continue
        
        # Anzahl der möglichen Anregungen
        n_excitations = int(U / U_excitation)
        
        # Restenergie nach Anregungen
        U_rest = U - n_excitations * U_excitation
        
        # Strom steigt mit Restenergie, fällt nach jeder Anregung
        # Exponentieller Anstieg zwischen den Maxima
        base_current = amplitude * (1 - np.exp(-U / 2.0))
        
        # Modulationen durch Anregungen
        modulation = 0.0
        for k in range(n_excitations + 1):
            U_local = U - k * U_excitation
            if U_local > 0:
                # Sägezahn-artige Struktur
                phase = (U_local % U_excitation) / U_excitation
                modulation += np.sin(np.pi * phase) * np.exp(-damping * k)
        
        I[i] = base_current * (0.5 + 0.5 * modulation / max(1, n_excitations + 1))
        
        # Zusätzliche Dämpfung bei hohen Spannungen
        I[i] *= np.exp(-damping * U / 20.0)
    
    return I


# ============================================================
# EMISSION SPECTRA (simplified)
# ============================================================

ELEMENT_SPECTRA = {
    "Wasserstoff (H)": {
        "lines": [656.3, 486.1, 434.0, 410.2, 397.0],  # Balmer-Serie [nm]
        "intensities": [1.0, 0.8, 0.5, 0.3, 0.2],
        "color": "#FF6B6B"
    },
    "Helium (He)": {
        "lines": [706.5, 667.8, 587.6, 501.6, 492.2, 471.3, 447.1],
        "intensities": [0.3, 0.5, 1.0, 0.4, 0.6, 0.5, 0.7],
        "color": "#FFE66D"
    },
    "Natrium (Na)": {
        "lines": [589.0, 589.6, 568.8, 498.3, 466.5],  # D-Linien dominant
        "intensities": [1.0, 0.95, 0.2, 0.15, 0.1],
        "color": "#FFA500"
    },
    "Neon (Ne)": {
        "lines": [703.2, 667.8, 640.2, 633.4, 614.3, 585.2],
        "intensities": [0.4, 0.5, 0.8, 1.0, 0.7, 0.6],
        "color": "#FF4500"
    },
    "Quecksilber (Hg)": {
        "lines": [579.1, 546.1, 435.8, 404.7, 365.0],
        "intensities": [0.8, 1.0, 0.9, 0.7, 0.5],
        "color": "#4169E1"
    },
}


# ============================================================
# STREAMLIT UI
# ============================================================

def render_atom_tab():
    """Hauptfunktion für den Atomphysik-Tab"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("⚛️ Atomphysik", "⚛️ Atomic Physics"))
    
    # Sub-Tabs
    bohr_tab, photo_tab, franck_tab, spectra_tab = st.tabs([
        tr("Bohr-Modell", "Bohr Model"),
        tr("Photoeffekt", "Photoelectric Effect"),
        tr("Franck-Hertz", "Franck-Hertz"),
        tr("Spektroskopie", "Spectroscopy")
    ])
    
    with bohr_tab:
        render_bohr_tab()
    
    with photo_tab:
        render_photoeffect_tab()
    
    with franck_tab:
        render_franck_hertz_tab()
    
    with spectra_tab:
        render_spectra_tab()


def render_bohr_tab():
    """Bohr-Modell mit Elektronenübergängen"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Bohrsches Atommodell", "### Bohr Atomic Model"))
    st.latex(r"E_n = -\frac{13.6 \, \text{eV}}{n^2} \cdot Z^2")
    st.latex(r"r_n = \frac{a_0 \cdot n^2}{Z} = \frac{0.529 \, \text{Å} \cdot n^2}{Z}")
    
    col1, col2 = st.columns(2)
    with col1:
        Z = st.selectbox(
            tr("Kernladungszahl Z", "Nuclear charge Z"),
            [1, 2, 3],
            format_func=lambda x: {1: "H (Z=1)", 2: "He⁺ (Z=2)", 3: "Li²⁺ (Z=3)"}[x],
            key="bohr_Z"
        )
        n_max = st.slider(
            tr("Maximale Schale n", "Maximum shell n"),
            3, 7, 5,
            key="bohr_nmax"
        )
    with col2:
        n_initial = st.selectbox(
            tr("Startniveau n_i", "Initial level n_i"),
            list(range(2, n_max + 1)),
            index=min(2, n_max - 2),
            key="bohr_ni"
        )
        n_final = st.selectbox(
            tr("Endniveau n_f", "Final level n_f"),
            list(range(1, n_initial)),
            key="bohr_nf"
        )
    
    animate = st.checkbox(tr("Übergang animieren", "Animate transition"), key="bohr_animate")
    
    if st.button(tr("📊 Atommodell berechnen", "📊 Compute atomic model"), key="bohr_run", use_container_width=True):
        # Energieniveaus berechnen
        energies = {n: bohr_energy(n, Z) for n in range(1, n_max + 1)}
        radii = {n: bohr_radius(n, Z) * 1e10 for n in range(1, n_max + 1)}  # in Ångström
        
        # Übergangs-Wellenlänge
        wl = transition_wavelength(n_initial, n_final, Z)
        delta_E = energies[n_initial] - energies[n_final]
        rgb = wavelength_to_rgb(wl)
        color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        # === Energieniveau-Diagramm ===
        fig_energy = go.Figure()
        
        for n, E in energies.items():
            fig_energy.add_trace(go.Scatter(
                x=[-1, 1],
                y=[E, E],
                mode='lines',
                line=dict(color='blue', width=3),
                name=f"n={n}",
                hoverinfo='text',
                hovertext=f"n={n}: E = {E:.3f} eV"
            ))
            fig_energy.add_annotation(
                x=1.2, y=E,
                text=f"n={n}",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Übergangs-Pfeil
        fig_energy.add_annotation(
            x=0, y=energies[n_initial],
            ax=0, ay=energies[n_final],
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=color_hex
        )
        
        # Photon-Symbol
        fig_energy.add_annotation(
            x=0.5, y=(energies[n_initial] + energies[n_final]) / 2,
            text=f"γ: {wl:.1f} nm",
            showarrow=False,
            font=dict(size=14, color=color_hex),
            bgcolor="white"
        )
        
        fig_energy.update_layout(
            title=tr(f"Energieniveau-Diagramm (Z={Z})", f"Energy Level Diagram (Z={Z})"),
            xaxis=dict(visible=False, range=[-2, 2]),
            yaxis=dict(title=tr("Energie [eV]", "Energy [eV]")),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # === Bahnmodell (kreisförmig) ===
        fig_orbit = go.Figure()
        
        # Kern
        fig_orbit.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=20, color='red'),
            name=tr("Kern", "Nucleus")
        ))
        
        # Bahnen
        theta = np.linspace(0, 2*np.pi, 100)
        for n in range(1, n_max + 1):
            r = radii[n] / radii[n_max]  # Normiert
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            fig_orbit.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='lightblue', width=1, dash='dot'),
                name=f"n={n}",
                hoverinfo='skip'
            ))
        
        # Elektron auf Startbahn
        r_electron = radii[n_initial] / radii[n_max]
        fig_orbit.add_trace(go.Scatter(
            x=[r_electron], y=[0],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name=tr("Elektron", "Electron")
        ))
        
        fig_orbit.update_layout(
            title=tr("Bohr-Bahnen (schematisch)", "Bohr Orbits (schematic)"),
            xaxis=dict(scaleanchor="y", scaleratio=1, visible=False),
            yaxis=dict(visible=False),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_orbit, use_container_width=True)
        
        # Ergebnisse
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(tr("Wellenlänge λ", "Wavelength λ"), f"{wl:.2f} nm")
        with col2:
            st.metric(tr("Photonenenergie", "Photon energy"), f"{abs(delta_E):.3f} eV")
        with col3:
            spectrum_type = "UV" if wl < 380 else ("Sichtbar" if wl < 780 else "IR")
            st.metric(tr("Spektralbereich", "Spectral range"), spectrum_type)
        
        # Farbvisualisierung
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {color_hex}, {color_hex}); 
                    height: 30px; border-radius: 5px; margin: 10px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Animation
        if animate:
            run_bohr_animation(n_initial, n_final, radii, n_max, color_hex, wl)
        
        # Spektralserien-Tabelle
        st.markdown(tr("### Spektralserien", "### Spectral Series"))
        series_data = []
        for series_name, (n_low, description) in get_spectral_series().items():
            wavelengths = []
            for n_high in range(n_low + 1, min(n_low + 5, 8)):
                wl_series = transition_wavelength(n_high, n_low, Z)
                if wl_series < 10000:
                    wavelengths.append(f"{wl_series:.1f}")
            series_data.append({
                tr("Serie", "Series"): series_name,
                tr("Endniveau", "Final level"): f"n = {n_low}",
                tr("Bereich", "Range"): description,
                tr("Wellenlängen [nm]", "Wavelengths [nm]"): ", ".join(wavelengths[:4])
            })
        
        st.dataframe(series_data, use_container_width=True)


def run_bohr_animation(n_initial: int, n_final: int, radii: dict, n_max: int, color_hex: str, wavelength: float):
    """Animation des Elektronenübergangs"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    chart_placeholder = st.empty()
    info_placeholder = st.empty()
    
    n_frames = 60
    r_start = radii[n_initial] / radii[n_max]
    r_end = radii[n_final] / radii[n_max]
    
    for frame in range(n_frames):
        t = frame / (n_frames - 1)
        
        # Elektron spiralt nach innen
        r_current = r_start + (r_end - r_start) * t
        angle = t * 4 * np.pi  # Mehrere Umdrehungen
        
        x_electron = r_current * np.cos(angle)
        y_electron = r_current * np.sin(angle)
        
        fig = go.Figure()
        
        # Kern
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=20, color='red'),
            showlegend=False
        ))
        
        # Bahnen
        theta = np.linspace(0, 2*np.pi, 100)
        for n in range(1, n_max + 1):
            r = radii[n] / radii[n_max]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            line_color = color_hex if n in [n_initial, n_final] else 'lightgray'
            line_width = 2 if n in [n_initial, n_final] else 1
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=line_color, width=line_width),
                showlegend=False
            ))
        
        # Elektron
        fig.add_trace(go.Scatter(
            x=[x_electron], y=[y_electron],
            mode='markers',
            marker=dict(size=15, color='blue'),
            showlegend=False
        ))
        
        # Photon (erscheint ab Mitte der Animation)
        if t > 0.5:
            photon_t = (t - 0.5) * 2
            photon_x = r_end + photon_t * 0.5
            fig.add_trace(go.Scatter(
                x=[photon_x], y=[0],
                mode='markers',
                marker=dict(size=10, color=color_hex, symbol='star'),
                name='γ',
                showlegend=False
            ))
        
        fig.update_layout(
            title=tr(f"Übergang n={n_initial} → n={n_final}", f"Transition n={n_initial} → n={n_final}"),
            xaxis=dict(range=[-1.3, 1.5], scaleanchor="y", visible=False),
            yaxis=dict(range=[-1.3, 1.3], visible=False),
            height=400
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        info_placeholder.info(tr(
            f"Frame {frame + 1}/{n_frames} | Photon: λ = {wavelength:.1f} nm",
            f"Frame {frame + 1}/{n_frames} | Photon: λ = {wavelength:.1f} nm"
        ))
        
        time.sleep(0.05)
    
    st.success(tr("✅ Animation abgeschlossen — Photon emittiert!", "✅ Animation complete — Photon emitted!"))


def render_photoeffect_tab():
    """Photoeffekt-Simulation"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Photoelektrischer Effekt", "### Photoelectric Effect"))
    st.latex(r"E_{kin} = h \cdot f - W = \frac{h \cdot c}{\lambda} - W")
    
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.selectbox(
            tr("Material", "Material"),
            list(MATERIALS.keys()),
            key="photo_material"
        )
        material = MATERIALS[material_name]
        
        st.info(tr(
            f"Austrittsarbeit W = {material.work_function_eV:.2f} eV",
            f"Work function W = {material.work_function_eV:.2f} eV"
        ))
    
    with col2:
        wavelength = st.slider(
            tr("Wellenlänge λ [nm]", "Wavelength λ [nm]"),
            100, 800, 400, 10,
            key="photo_wavelength"
        )
        
        # Farbvisualisierung
        rgb = wavelength_to_rgb(wavelength)
        color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        st.markdown(f"""
        <div style="background: {color_hex}; height: 20px; border-radius: 5px;"></div>
        """, unsafe_allow_html=True)
    
    intensity = st.slider(
        tr("Lichtintensität (relativ)", "Light intensity (relative)"),
        0.1, 2.0, 1.0, 0.1,
        key="photo_intensity"
    )
    
    animate = st.checkbox(tr("Elektronen animieren", "Animate electrons"), key="photo_animate")
    
    if st.button(tr("📊 Photoeffekt berechnen", "📊 Compute photoelectric effect"), key="photo_run", use_container_width=True):
        # Berechnungen
        photon_energy = h_eV * c / (wavelength * 1e-9)
        E_kin = photoelectric_kinetic_energy(wavelength, material.work_function_eV)
        threshold_wl = threshold_wavelength(material.work_function_eV)
        U_stop = stopping_potential(E_kin)
        
        # === E_kin vs λ Diagramm ===
        wavelengths = np.linspace(100, 800, 200)
        E_kin_curve = np.array([photoelectric_kinetic_energy(wl, material.work_function_eV) for wl in wavelengths])
        
        fig = go.Figure()
        
        # E_kin Kurve
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=E_kin_curve,
            mode='lines',
            line=dict(color='blue', width=3),
            name=tr("E_kin(λ)", "E_kin(λ)")
        ))
        
        # Grenzwellenlänge
        fig.add_vline(x=threshold_wl, line_dash="dash", line_color="red",
                     annotation_text=tr(f"λ_grenz = {threshold_wl:.0f} nm", f"λ_threshold = {threshold_wl:.0f} nm"))
        
        # Aktueller Punkt
        fig.add_trace(go.Scatter(
            x=[wavelength],
            y=[E_kin],
            mode='markers',
            marker=dict(size=15, color=color_hex, line=dict(width=2, color='black')),
            name=tr("Aktuell", "Current")
        ))
        
        fig.update_layout(
            title=tr(f"Photoeffekt: {material_name}", f"Photoelectric Effect: {material_name}"),
            xaxis_title=tr("Wellenlänge λ [nm]", "Wavelength λ [nm]"),
            yaxis_title=tr("E_kin [eV]", "E_kin [eV]"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Ergebnisse
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(tr("Photonenenergie", "Photon energy"), f"{photon_energy:.3f} eV")
        with col2:
            st.metric(tr("E_kin (Elektron)", "E_kin (electron)"), f"{E_kin:.3f} eV")
        with col3:
            st.metric(tr("Grenzwellenlänge", "Threshold λ"), f"{threshold_wl:.0f} nm")
        with col4:
            st.metric(tr("Bremsspannung", "Stopping potential"), f"{U_stop:.3f} V")
        
        # Status
        if E_kin > 0:
            st.success(tr(
                f"✅ Photoeffekt tritt auf! Elektronen werden mit E_kin = {E_kin:.3f} eV emittiert.",
                f"✅ Photoelectric effect occurs! Electrons emitted with E_kin = {E_kin:.3f} eV"
            ))
            
            if animate:
                run_photoeffect_animation(E_kin, intensity, color_hex)
        else:
            st.error(tr(
                f"❌ Kein Photoeffekt! λ = {wavelength} nm > λ_grenz = {threshold_wl:.0f} nm",
                f"❌ No photoelectric effect! λ = {wavelength} nm > λ_threshold = {threshold_wl:.0f} nm"
            ))


def run_photoeffect_animation(E_kin: float, intensity: float, light_color: str):
    """Animation der Photoemission"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    chart_placeholder = st.empty()
    progress = st.progress(0)
    
    n_frames = 80
    n_electrons = int(5 * intensity)
    
    # Elektron-Trajektorien vorbereiten
    electrons = []
    for i in range(n_electrons):
        start_frame = np.random.randint(0, n_frames // 2)
        angle = np.random.uniform(-60, 60) * np.pi / 180
        speed = np.sqrt(E_kin) * 0.3
        electrons.append({
            'start': start_frame,
            'angle': angle,
            'speed': speed,
            'x0': np.random.uniform(-0.3, 0.3),
            'y0': 0
        })
    
    for frame in range(n_frames):
        fig = go.Figure()
        
        # Metalloberfläche
        fig.add_shape(type="rect", x0=-1, y0=-0.5, x1=1, y1=0,
                     fillcolor="#708090", line=dict(color="gray"))
        
        # Einfallendes Licht (Photonen)
        for i in range(3):
            photon_y = 1.5 - (frame * 0.05 + i * 0.3) % 2
            if photon_y > 0:
                fig.add_trace(go.Scatter(
                    x=[0.5 - i * 0.3], y=[photon_y],
                    mode='markers',
                    marker=dict(size=8, color=light_color, symbol='diamond'),
                    showlegend=False
                ))
        
        # Emittierte Elektronen
        for e in electrons:
            if frame >= e['start']:
                t = (frame - e['start']) * 0.05
                x = e['x0'] + e['speed'] * np.sin(e['angle']) * t
                y = e['y0'] + e['speed'] * np.cos(e['angle']) * t
                
                if y < 2:
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers',
                        marker=dict(size=8, color='blue'),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=tr("Photoemission", "Photoemission"),
            xaxis=dict(range=[-1.5, 1.5], visible=False),
            yaxis=dict(range=[-0.6, 2], visible=False),
            height=350
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        progress.progress((frame + 1) / n_frames)
        time.sleep(0.03)
    
    st.success(tr("✅ Animation abgeschlossen", "✅ Animation complete"))


def render_franck_hertz_tab():
    """Franck-Hertz-Experiment"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Franck-Hertz-Experiment", "### Franck-Hertz Experiment"))
    st.markdown(tr(
        """
        Nachweis der Quantelung der Atomenergie durch inelastische Elektronenstöße.
        Bei Quecksilber (Hg) beträgt die erste Anregungsenergie **4.9 eV** (entspricht UV-Linie bei 253.7 nm).
        """,
        """
        Demonstration of quantized atomic energy through inelastic electron collisions.
        For mercury (Hg), the first excitation energy is **4.9 eV** (corresponding to UV line at 253.7 nm).
        """
    ))
    
    col1, col2 = st.columns(2)
    with col1:
        element = st.selectbox(
            tr("Gas", "Gas"),
            ["Quecksilber (Hg)", "Neon (Ne)"],
            key="fh_element"
        )
        U_excitation = 4.9 if "Hg" in element else 18.7
        
        U_max = st.slider(
            tr("Maximale Beschleunigungsspannung [V]", "Maximum acceleration voltage [V]"),
            10, 60, 30,
            key="fh_Umax"
        )
    
    with col2:
        damping = st.slider(
            tr("Dämpfung", "Damping"),
            0.05, 0.3, 0.1, 0.01,
            key="fh_damping"
        )
        
        show_theory = st.checkbox(tr("Theoretische Maxima zeigen", "Show theoretical maxima"), value=True, key="fh_theory")
    
    animate = st.checkbox(tr("Messung animieren", "Animate measurement"), key="fh_animate")
    
    if st.button(tr("📊 Franck-Hertz-Kurve berechnen", "📊 Compute Franck-Hertz curve"), key="fh_run", use_container_width=True):
        U = np.linspace(0, U_max, 500)
        I = franck_hertz_current(U, U_excitation, damping=damping)
        
        if animate:
            run_franck_hertz_animation(U, I, U_excitation, show_theory)
        else:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=U, y=I,
                mode='lines',
                line=dict(color='blue', width=2),
                name=tr("Anodenstrom", "Anode current")
            ))
            
            if show_theory:
                for n in range(1, int(U_max / U_excitation) + 1):
                    U_peak = n * U_excitation
                    if U_peak <= U_max:
                        fig.add_vline(x=U_peak, line_dash="dash", line_color="red", opacity=0.5)
                        fig.add_annotation(x=U_peak, y=max(I) * 1.05, text=f"{U_peak:.1f}V", showarrow=False)
            
            fig.update_layout(
                title=tr(f"Franck-Hertz-Kurve ({element})", f"Franck-Hertz Curve ({element})"),
                xaxis_title=tr("Beschleunigungsspannung U [V]", "Acceleration voltage U [V]"),
                yaxis_title=tr("Anodenstrom I [a.u.]", "Anode current I [a.u.]"),
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Ergebnisse
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(tr("Anregungsenergie", "Excitation energy"), f"{U_excitation} eV")
        with col2:
            wavelength = h_eV * c / U_excitation * 1e9
            st.metric(tr("Entspr. Wellenlänge", "Corresp. wavelength"), f"{wavelength:.1f} nm")
        with col3:
            n_peaks = int(U_max / U_excitation)
            st.metric(tr("Anzahl Maxima", "Number of maxima"), n_peaks)


def run_franck_hertz_animation(U: np.ndarray, I: np.ndarray, U_excitation: float, show_theory: bool):
    """Animation der Franck-Hertz-Messung"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    chart_placeholder = st.empty()
    progress = st.progress(0)
    
    n_points = len(U)
    step = max(1, n_points // 100)
    
    for i in range(step, n_points + 1, step):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=U[:i], y=I[:i],
            mode='lines',
            line=dict(color='blue', width=2),
            name=tr("Anodenstrom", "Anode current")
        ))
        
        if show_theory:
            for n in range(1, int(U[i-1] / U_excitation) + 1):
                U_peak = n * U_excitation
                if U_peak <= U[i-1]:
                    fig.add_vline(x=U_peak, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title=tr(f"Franck-Hertz-Messung: U = {U[i-1]:.1f} V", 
                    f"Franck-Hertz Measurement: U = {U[i-1]:.1f} V"),
            xaxis_title=tr("U [V]", "U [V]"),
            yaxis_title=tr("I [a.u.]", "I [a.u.]"),
            xaxis=dict(range=[0, max(U)]),
            yaxis=dict(range=[0, max(I) * 1.1]),
            height=400
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        progress.progress(i / n_points)
        time.sleep(0.02)
    
    st.success(tr("✅ Messung abgeschlossen", "✅ Measurement complete"))


def render_spectra_tab():
    """Emissions- und Absorptionsspektren"""
    import plotly.graph_objects as go
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Atomspektren", "### Atomic Spectra"))
    
    spectrum_type = st.radio(
        tr("Spektrumtyp", "Spectrum type"),
        [tr("Emission", "Emission"), tr("Absorption", "Absorption")],
        horizontal=True,
        key="spectra_type"
    )
    
    element = st.selectbox(
        tr("Element", "Element"),
        list(ELEMENT_SPECTRA.keys()),
        key="spectra_element"
    )
    
    show_continuous = st.checkbox(tr("Kontinuierliches Spektrum zeigen", "Show continuous spectrum"), value=True, key="spectra_cont")
    
    if st.button(tr("📊 Spektrum anzeigen", "📊 Show spectrum"), key="spectra_run", use_container_width=True):
        spec_data = ELEMENT_SPECTRA[element]
        lines = spec_data["lines"]
        intensities = spec_data["intensities"]
        
        fig = go.Figure()
        
        # Kontinuierliches Spektrum im Hintergrund
        if show_continuous:
            wavelengths = np.linspace(380, 780, 400)
            colors = [f"rgb{wavelength_to_rgb(wl)}" for wl in wavelengths]
            
            for i, wl in enumerate(wavelengths[:-1]):
                fig.add_shape(
                    type="rect",
                    x0=wl, x1=wavelengths[i+1],
                    y0=0, y1=1,
                    fillcolor=colors[i],
                    line=dict(width=0),
                    layer="below"
                )
        
        # Spektrallinien
        is_emission = "Emission" in spectrum_type
        
        for wl, intensity in zip(lines, intensities):
            if 380 <= wl <= 780:  # Nur sichtbarer Bereich
                rgb = wavelength_to_rgb(wl)
                color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                
                if is_emission:
                    fig.add_trace(go.Scatter(
                        x=[wl, wl],
                        y=[0, intensity],
                        mode='lines',
                        line=dict(color=color, width=4),
                        name=f"{wl:.1f} nm",
                        hoverinfo='text',
                        hovertext=f"λ = {wl:.1f} nm"
                    ))
                else:
                    # Absorption: schwarze Linien
                    fig.add_shape(
                        type="line",
                        x0=wl, x1=wl,
                        y0=0, y1=1,
                        line=dict(color="black", width=3)
                    )
        
        fig.update_layout(
            title=tr(f"{'Emissions' if is_emission else 'Absorptions'}spektrum: {element}",
                    f"{'Emission' if is_emission else 'Absorption'} Spectrum: {element}"),
            xaxis_title=tr("Wellenlänge λ [nm]", "Wavelength λ [nm]"),
            yaxis_title=tr("Intensität [a.u.]", "Intensity [a.u.]") if is_emission else "",
            xaxis=dict(range=[380, 780]),
            yaxis=dict(range=[0, 1.2], visible=is_emission),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Linientabelle
        st.markdown(tr("### Spektrallinien", "### Spectral Lines"))
        line_data = []
        for wl, intensity in zip(lines, intensities):
            rgb = wavelength_to_rgb(wl)
            color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            spectral_range = "UV" if wl < 380 else ("Sichtbar" if wl < 780 else "IR")
            
            line_data.append({
                tr("Wellenlänge [nm]", "Wavelength [nm]"): f"{wl:.1f}",
                tr("Intensität", "Intensity"): f"{intensity:.2f}",
                tr("Bereich", "Range"): spectral_range,
                tr("Farbe", "Color"): color_hex
            })
        
        st.dataframe(line_data, use_container_width=True)
        
        # Spektralbalken
        st.markdown(tr("### Linienspektrum (visuell)", "### Line Spectrum (visual)"))
        spectrum_html = '<div style="display: flex; height: 50px; background: #111; border-radius: 5px; overflow: hidden;">'
        
        for wl, intensity in sorted(zip(lines, intensities)):
            if 380 <= wl <= 780:
                rgb = wavelength_to_rgb(wl)
                position = (wl - 380) / 400 * 100
                spectrum_html += f'<div style="position: absolute; left: {position}%; width: 3px; height: 50px; background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); opacity: {intensity};"></div>'
        
        spectrum_html += '</div>'
        
        # Vereinfachte Visualisierung
        colors_bar = []
        for wl in sorted(lines):
            if 380 <= wl <= 780:
                rgb = wavelength_to_rgb(wl)
                colors_bar.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
        
        if colors_bar:
            gradient = ", ".join([f"{c} {i*100//(len(colors_bar)-1) if len(colors_bar) > 1 else 50}%" 
                                 for i, c in enumerate(colors_bar)])
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #111 0%, {colors_bar[0]} 5%, 
                        {' '.join([f'{c},' for c in colors_bar])} #111 100%); 
                        height: 40px; border-radius: 5px; margin: 10px 0;"></div>
            """, unsafe_allow_html=True)
