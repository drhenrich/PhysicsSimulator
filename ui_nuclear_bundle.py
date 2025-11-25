"""
Kernphysik & Strahlenschutz Modul
=================================
- Radioaktiver Zerfall und Zerfallsreihen
- Aktivitätskurven (Bateman-Gleichungen)
- Dosimetrie und Abstandsquadratgesetz
- Abschirmungsberechnungen
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Dict

# =============================================================================
# Physikalische Konstanten und Daten
# =============================================================================

# Avogadro-Konstante
N_A = 6.02214076e23  # 1/mol

# Natürliche Zerfallsreihen (vereinfacht)
# Format: (Nuklid, Halbwertszeit [s], Zerfallsart, Tochternuklid)
DECAY_CHAINS = {
    "U-238": [
        ("U-238", 4.468e9 * 365.25 * 24 * 3600, "α", "Th-234"),
        ("Th-234", 24.10 * 24 * 3600, "β⁻", "Pa-234m"),
        ("Pa-234m", 1.17 * 60, "β⁻", "U-234"),
        ("U-234", 2.455e5 * 365.25 * 24 * 3600, "α", "Th-230"),
        ("Th-230", 7.54e4 * 365.25 * 24 * 3600, "α", "Ra-226"),
        ("Ra-226", 1600 * 365.25 * 24 * 3600, "α", "Rn-222"),
        ("Rn-222", 3.8235 * 24 * 3600, "α", "Po-218"),
        ("Po-218", 3.10 * 60, "α", "Pb-214"),
        ("Pb-214", 26.8 * 60, "β⁻", "Bi-214"),
        ("Bi-214", 19.9 * 60, "β⁻", "Po-214"),
        ("Po-214", 164.3e-6, "α", "Pb-210"),
        ("Pb-210", 22.3 * 365.25 * 24 * 3600, "β⁻", "Bi-210"),
        ("Bi-210", 5.013 * 24 * 3600, "β⁻", "Po-210"),
        ("Po-210", 138.376 * 24 * 3600, "α", "Pb-206"),
        ("Pb-206", float('inf'), "stabil", "-"),
    ],
    "Th-232": [
        ("Th-232", 1.405e10 * 365.25 * 24 * 3600, "α", "Ra-228"),
        ("Ra-228", 5.75 * 365.25 * 24 * 3600, "β⁻", "Ac-228"),
        ("Ac-228", 6.15 * 3600, "β⁻", "Th-228"),
        ("Th-228", 1.9116 * 365.25 * 24 * 3600, "α", "Ra-224"),
        ("Ra-224", 3.66 * 24 * 3600, "α", "Rn-220"),
        ("Rn-220", 55.6, "α", "Po-216"),
        ("Po-216", 0.145, "α", "Pb-212"),
        ("Pb-212", 10.64 * 3600, "β⁻", "Bi-212"),
        ("Bi-212", 60.55 * 60, "β⁻/α", "Po-212/Tl-208"),
        ("Po-212", 299e-9, "α", "Pb-208"),
        ("Pb-208", float('inf'), "stabil", "-"),
    ],
    "U-235": [
        ("U-235", 7.04e8 * 365.25 * 24 * 3600, "α", "Th-231"),
        ("Th-231", 25.52 * 3600, "β⁻", "Pa-231"),
        ("Pa-231", 3.276e4 * 365.25 * 24 * 3600, "α", "Ac-227"),
        ("Ac-227", 21.772 * 365.25 * 24 * 3600, "β⁻", "Th-227"),
        ("Th-227", 18.68 * 24 * 3600, "α", "Ra-223"),
        ("Ra-223", 11.43 * 24 * 3600, "α", "Rn-219"),
        ("Rn-219", 3.96, "α", "Po-215"),
        ("Po-215", 1.781e-3, "α", "Pb-211"),
        ("Pb-211", 36.1 * 60, "β⁻", "Bi-211"),
        ("Bi-211", 2.14 * 60, "α", "Tl-207"),
        ("Tl-207", 4.77 * 60, "β⁻", "Pb-207"),
        ("Pb-207", float('inf'), "stabil", "-"),
    ],
}

# Häufig verwendete Radionuklide für Einzelzerfall
COMMON_NUCLIDES = {
    "I-131": {"T_half": 8.0252 * 24 * 3600, "decay": "β⁻", "E_gamma": 0.364, "Gamma": 5.5e-5},
    "Tc-99m": {"T_half": 6.01 * 3600, "decay": "IT", "E_gamma": 0.140, "Gamma": 2.0e-5},
    "Co-60": {"T_half": 5.2714 * 365.25 * 24 * 3600, "decay": "β⁻", "E_gamma": 1.25, "Gamma": 3.5e-4},
    "Cs-137": {"T_half": 30.08 * 365.25 * 24 * 3600, "decay": "β⁻", "E_gamma": 0.662, "Gamma": 8.5e-5},
    "Ra-226": {"T_half": 1600 * 365.25 * 24 * 3600, "decay": "α", "E_gamma": 0.186, "Gamma": 2.2e-4},
    "Am-241": {"T_half": 432.2 * 365.25 * 24 * 3600, "decay": "α", "E_gamma": 0.060, "Gamma": 3.1e-6},
    "F-18": {"T_half": 109.77 * 60, "decay": "β⁺", "E_gamma": 0.511, "Gamma": 1.4e-4},
    "Sr-90": {"T_half": 28.8 * 365.25 * 24 * 3600, "decay": "β⁻", "E_gamma": 0.0, "Gamma": 0.0},
    "P-32": {"T_half": 14.29 * 24 * 3600, "decay": "β⁻", "E_gamma": 0.0, "Gamma": 0.0},
    "C-14": {"T_half": 5730 * 365.25 * 24 * 3600, "decay": "β⁻", "E_gamma": 0.0, "Gamma": 0.0},
}

# Abschirmungsmaterialien: Schwächungskoeffizienten μ [1/cm] bei verschiedenen Energien
# Vereinfacht für typische Gamma-Energien
SHIELDING_MATERIALS = {
    "Blei (Pb)": {
        "rho": 11.34,  # g/cm³
        "mu": {0.1: 59.7, 0.2: 10.15, 0.5: 1.64, 0.662: 1.17, 1.0: 0.776, 1.25: 0.653, 2.0: 0.518},
        "HVL_662": 0.59,  # cm bei Cs-137
        "color": "#4a5568"
    },
    "Beton": {
        "rho": 2.35,
        "mu": {0.1: 4.56, 0.2: 0.60, 0.5: 0.22, 0.662: 0.18, 1.0: 0.15, 1.25: 0.13, 2.0: 0.11},
        "HVL_662": 4.8,
        "color": "#a0aec0"
    },
    "Wasser": {
        "rho": 1.0,
        "mu": {0.1: 0.17, 0.2: 0.14, 0.5: 0.097, 0.662: 0.086, 1.0: 0.071, 1.25: 0.063, 2.0: 0.049},
        "HVL_662": 8.0,
        "color": "#63b3ed"
    },
    "Eisen (Fe)": {
        "rho": 7.87,
        "mu": {0.1: 25.6, 0.2: 2.72, 0.5: 0.65, 0.662: 0.52, 1.0: 0.41, 1.25: 0.36, 2.0: 0.31},
        "HVL_662": 1.6,
        "color": "#718096"
    },
    "Aluminium (Al)": {
        "rho": 2.70,
        "mu": {0.1: 0.44, 0.2: 0.32, 0.5: 0.23, 0.662: 0.20, 1.0: 0.17, 1.25: 0.14, 2.0: 0.12},
        "HVL_662": 3.1,
        "color": "#cbd5e0"
    },
}

# Dosisgrenzwerte (mSv/Jahr)
DOSE_LIMITS = {
    "Bevölkerung": 1.0,
    "Beruflich exponiert (Ganzkörper)": 20.0,
    "Beruflich exponiert (Hände)": 500.0,
    "Beruflich exponiert (Augenlinse)": 20.0,
    "Schwangere (Uterusdosis)": 1.0,
}


# =============================================================================
# Berechnungsfunktionen
# =============================================================================

def decay_constant(T_half: float) -> float:
    """Berechnet Zerfallskonstante λ = ln(2) / T½"""
    if T_half == float('inf'):
        return 0.0
    return np.log(2) / T_half


def activity(A0: float, lambda_: float, t: np.ndarray) -> np.ndarray:
    """Berechnet Aktivität A(t) = A₀ · e^(-λt)"""
    return A0 * np.exp(-lambda_ * t)


def bateman_equation(A0: float, lambdas: List[float], t: np.ndarray) -> np.ndarray:
    """
    Bateman-Gleichung für Zerfallsketten.
    Berechnet die Aktivität des n-ten Glieds einer Zerfallskette.
    
    A_n(t) = A₀ · λ₁·λ₂·...·λₙ · Σᵢ (e^(-λᵢt) / Πⱼ≠ᵢ(λⱼ-λᵢ))
    """
    n = len(lambdas)
    if n == 0:
        return np.zeros_like(t)
    if n == 1:
        return activity(A0, lambdas[0], t)
    
    # Produkt aller Lambda
    lambda_prod = np.prod(lambdas)
    
    result = np.zeros_like(t, dtype=float)
    
    for i in range(n):
        # Produkt der Differenzen
        denom = 1.0
        for j in range(n):
            if i != j:
                diff = lambdas[j] - lambdas[i]
                if abs(diff) < 1e-20:
                    diff = 1e-20  # Vermeidung Division durch 0
                denom *= diff
        
        result += np.exp(-lambdas[i] * t) / denom
    
    return A0 * lambda_prod * result


def solve_decay_chain(chain: List[Tuple], A0: float, t: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Löst Zerfallskette numerisch mit vereinfachtem Bateman-Ansatz.
    """
    results = {}
    n = len(chain)
    
    # Initialisierung
    N = np.zeros((n, len(t)))
    N[0, 0] = A0  # Anfangsaktivität des Mutternuklids
    
    # Zerfallskonstanten
    lambdas = [decay_constant(c[1]) for c in chain]
    
    # Zeitschritt
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    
    # Numerische Integration (Euler-Methode)
    for i in range(1, len(t)):
        for j in range(n):
            decay_out = lambdas[j] * N[j, i-1] * dt if lambdas[j] > 0 else 0
            decay_in = lambdas[j-1] * N[j-1, i-1] * dt if j > 0 and lambdas[j-1] > 0 else 0
            N[j, i] = N[j, i-1] - decay_out + decay_in
            N[j, i] = max(0, N[j, i])  # Keine negativen Werte
    
    # Aktivitäten berechnen
    for j, (nuclide, T_half, _, _) in enumerate(chain):
        lam = lambdas[j]
        A = lam * N[j, :] if lam > 0 else np.zeros_like(t)
        results[nuclide] = A
    
    return results


def dose_rate_point_source(A: float, Gamma: float, r: float) -> float:
    """
    Dosisleistung einer Punktquelle (Abstandsquadratgesetz).
    
    Ḋ = A · Γ / r²
    
    Args:
        A: Aktivität [Bq]
        Gamma: Dosisleistungskonstante [Sv·m²/(Bq·s)]
        r: Abstand [m]
    
    Returns:
        Dosisleistung [Sv/s]
    """
    if r <= 0:
        return float('inf')
    return A * Gamma / (r ** 2)


def shielding_transmission(mu: float, x: float) -> float:
    """
    Transmission durch Abschirmung.
    
    T = e^(-μx)
    
    Args:
        mu: Linearer Schwächungskoeffizient [1/cm]
        x: Materialdicke [cm]
    
    Returns:
        Transmissionsfaktor (0-1)
    """
    return np.exp(-mu * x)


def half_value_layer(mu: float) -> float:
    """Berechnet Halbwertsschichtdicke HVL = ln(2) / μ"""
    if mu <= 0:
        return float('inf')
    return np.log(2) / mu


def tenth_value_layer(mu: float) -> float:
    """Berechnet Zehntelwertsschichtdicke TVL = ln(10) / μ"""
    if mu <= 0:
        return float('inf')
    return np.log(10) / mu


def required_shielding(mu: float, reduction_factor: float) -> float:
    """
    Berechnet erforderliche Abschirmdicke für gewünschte Reduktion.
    
    x = -ln(T) / μ
    """
    if mu <= 0 or reduction_factor <= 0:
        return float('inf')
    return -np.log(reduction_factor) / mu


# =============================================================================
# UI Rendering
# =============================================================================

def render_nuclear_tab():
    """Hauptfunktion für Kernphysik & Strahlenschutz Tab"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("☢️ Kernphysik & Strahlenschutz", "☢️ Nuclear Physics & Radiation Protection"))
    
    decay_tab, chain_tab, dose_tab, shield_tab = st.tabs([
        tr("Radioaktiver Zerfall", "Radioactive Decay"),
        tr("Zerfallsreihen", "Decay Chains"),
        tr("Dosimetrie", "Dosimetry"),
        tr("Abschirmung", "Shielding")
    ])
    
    # =========================================================================
    # Tab 1: Radioaktiver Zerfall
    # =========================================================================
    with decay_tab:
        st.markdown(tr(
            "**Zerfallsgesetz:** A(t) = A₀ · e^(-λt) mit λ = ln(2) / T½",
            "**Decay law:** A(t) = A₀ · e^(-λt) with λ = ln(2) / T½"
        ))
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            st.markdown(f"**{tr('Nuklid wählen', 'Select nuclide')}**")
            
            nuclide = st.selectbox(
                tr("Radionuklid", "Radionuclide"),
                list(COMMON_NUCLIDES.keys()),
                key="decay_nuclide"
            )
            
            nuc_data = COMMON_NUCLIDES[nuclide]
            T_half = nuc_data["T_half"]
            
            # Halbwertszeit anzeigen
            if T_half < 60:
                T_str = f"{T_half:.2f} s"
            elif T_half < 3600:
                T_str = f"{T_half/60:.2f} min"
            elif T_half < 86400:
                T_str = f"{T_half/3600:.2f} h"
            elif T_half < 365.25 * 86400:
                T_str = f"{T_half/86400:.2f} d"
            else:
                T_str = f"{T_half/(365.25*86400):.2f} a"
            
            st.info(f"T½ = {T_str}")
            st.caption(f"{tr('Zerfallsart', 'Decay type')}: {nuc_data['decay']}")
            if nuc_data['E_gamma'] > 0:
                st.caption(f"Eγ = {nuc_data['E_gamma']} MeV")
            
            st.markdown("---")
            
            A0 = st.number_input(
                tr("Anfangsaktivität A₀ [MBq]", "Initial activity A₀ [MBq]"),
                min_value=0.1, max_value=10000.0, value=100.0, step=10.0,
                key="decay_A0"
            ) * 1e6  # Umrechnung in Bq
            
            n_halflives = st.slider(
                tr("Anzahl Halbwertszeiten", "Number of half-lives"),
                min_value=1, max_value=10, value=5,
                key="decay_n_half"
            )
            
            show_log = st.checkbox(
                tr("Logarithmische Skala", "Logarithmic scale"),
                value=False,
                key="decay_log"
            )
        
        with col_plot:
            # Zeitachse
            t_max = n_halflives * T_half
            t = np.linspace(0, t_max, 500)
            
            # Zerfallskonstante
            lam = decay_constant(T_half)
            
            # Aktivität berechnen
            A = activity(A0, lam, t)
            
            # Zeiteinheit für Anzeige
            if T_half < 60:
                t_display = t
                t_unit = "s"
            elif T_half < 3600:
                t_display = t / 60
                t_unit = "min"
            elif T_half < 86400:
                t_display = t / 3600
                t_unit = "h"
            elif T_half < 365.25 * 86400:
                t_display = t / 86400
                t_unit = "d"
            else:
                t_display = t / (365.25 * 86400)
                t_unit = "a"
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=t_display,
                y=A / 1e6,  # MBq
                mode='lines',
                name=nuclide,
                line=dict(color='#e53e3e', width=3),
                fill='tozeroy',
                fillcolor='rgba(229, 62, 62, 0.2)'
            ))
            
            # Halbwertszeit-Markierungen
            for i in range(1, n_halflives + 1):
                t_half_i = i * T_half
                A_half = A0 / (2 ** i)
                
                if T_half < 60:
                    t_mark = t_half_i
                elif T_half < 3600:
                    t_mark = t_half_i / 60
                elif T_half < 86400:
                    t_mark = t_half_i / 3600
                elif T_half < 365.25 * 86400:
                    t_mark = t_half_i / 86400
                else:
                    t_mark = t_half_i / (365.25 * 86400)
                
                fig.add_vline(
                    x=t_mark,
                    line=dict(color='gray', dash='dash', width=1)
                )
                fig.add_annotation(
                    x=t_mark, y=A_half / 1e6,
                    text=f"{i}×T½",
                    showarrow=True,
                    arrowhead=2,
                    ax=30, ay=-20
                )
            
            fig.update_layout(
                title=tr(f"Aktivitätskurve {nuclide}", f"Activity curve {nuclide}"),
                xaxis_title=f"t [{t_unit}]",
                yaxis_title="A [MBq]",
                yaxis_type="log" if show_log else "linear",
                height=450,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Kennwerte
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("λ", f"{lam:.4e} s⁻¹")
            with col2:
                st.metric(tr("A nach 1 T½", "A after 1 T½"), f"{A0/2/1e6:.1f} MBq")
            with col3:
                st.metric(tr("A nach 10 T½", "A after 10 T½"), f"{A0/1024/1e6:.3f} MBq")
    
    # =========================================================================
    # Tab 2: Zerfallsreihen
    # =========================================================================
    with chain_tab:
        st.markdown(tr(
            "**Natürliche Zerfallsreihen** — Bateman-Gleichungen für sequentiellen Zerfall",
            "**Natural decay chains** — Bateman equations for sequential decay"
        ))
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            chain_name = st.selectbox(
                tr("Zerfallsreihe", "Decay chain"),
                list(DECAY_CHAINS.keys()),
                key="chain_select"
            )
            
            chain = DECAY_CHAINS[chain_name]
            
            st.markdown(f"**{tr('Glieder der Reihe', 'Chain members')}**")
            
            # Tabelle der Nuklide
            chain_table = []
            for nuc, T, decay, daughter in chain[:8]:  # Erste 8 zeigen
                if T == float('inf'):
                    T_str = tr("stabil", "stable")
                elif T < 1:
                    T_str = f"{T*1e6:.1f} µs"
                elif T < 60:
                    T_str = f"{T:.2f} s"
                elif T < 3600:
                    T_str = f"{T/60:.1f} min"
                elif T < 86400:
                    T_str = f"{T/3600:.1f} h"
                elif T < 365.25 * 86400:
                    T_str = f"{T/86400:.1f} d"
                else:
                    T_str = f"{T/(365.25*86400):.2e} a"
                
                chain_table.append({
                    tr("Nuklid", "Nuclide"): nuc,
                    "T½": T_str,
                    tr("Zerfall", "Decay"): decay
                })
            
            st.dataframe(chain_table, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            A0_chain = st.number_input(
                tr("A₀ Mutternuklid [kBq]", "A₀ parent nuclide [kBq]"),
                min_value=0.1, max_value=1000.0, value=100.0,
                key="chain_A0"
            ) * 1e3
            
            # Zeitskalierung basierend auf kurzlebigsten relevanten Nuklid
            time_scale = st.selectbox(
                tr("Zeitskala", "Time scale"),
                [tr("Sekunden", "Seconds"), tr("Minuten", "Minutes"), 
                 tr("Stunden", "Hours"), tr("Tage", "Days"), tr("Jahre", "Years")],
                index=3,
                key="chain_time"
            )
            
            n_nuclides = st.slider(
                tr("Anzahl Nuklide anzeigen", "Number of nuclides to show"),
                min_value=2, max_value=min(8, len(chain)), value=min(5, len(chain)),
                key="chain_n_show"
            )
        
        with col_plot:
            # Zeitparameter
            time_factors = {
                tr("Sekunden", "Seconds"): 1.0,
                tr("Minuten", "Minutes"): 60.0,
                tr("Stunden", "Hours"): 3600.0,
                tr("Tage", "Days"): 86400.0,
                tr("Jahre", "Years"): 365.25 * 86400.0
            }
            t_factor = time_factors[time_scale]
            
            # Wähle geeignete Zeitspanne
            relevant_halflives = [c[1] for c in chain[:n_nuclides] if c[1] < float('inf')]
            if relevant_halflives:
                t_max = 10 * max(relevant_halflives)
            else:
                t_max = 1000 * t_factor
            
            t = np.linspace(0, t_max, 1000)
            
            # Zerfallskette lösen
            results = solve_decay_chain(chain[:n_nuclides], A0_chain, t)
            
            # Plot
            fig = go.Figure()
            
            colors = ['#e53e3e', '#dd6b20', '#d69e2e', '#38a169', 
                      '#3182ce', '#805ad5', '#d53f8c', '#718096']
            
            for i, (nuclide, T_half, decay, _) in enumerate(chain[:n_nuclides]):
                if nuclide in results:
                    A_vals = results[nuclide]
                    
                    fig.add_trace(go.Scatter(
                        x=t / t_factor,
                        y=A_vals / 1e3,  # kBq
                        mode='lines',
                        name=f"{nuclide} ({decay})",
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            time_labels = {
                tr("Sekunden", "Seconds"): "s",
                tr("Minuten", "Minutes"): "min",
                tr("Stunden", "Hours"): "h",
                tr("Tage", "Days"): "d",
                tr("Jahre", "Years"): "a"
            }
            
            fig.update_layout(
                title=tr(f"Zerfallsreihe {chain_name}", f"Decay chain {chain_name}"),
                xaxis_title=f"t [{time_labels[time_scale]}]",
                yaxis_title="A [kBq]",
                height=500,
                legend=dict(x=1.02, y=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Erklärung
            st.caption(tr(
                "💡 Bei säkularem Gleichgewicht (T½_Mutter >> T½_Tochter) "
                "erreichen die Tochteraktivitäten die Mutteraktivität.",
                "💡 In secular equilibrium (T½_parent >> T½_daughter), "
                "daughter activities approach parent activity."
            ))
    
    # =========================================================================
    # Tab 3: Dosimetrie
    # =========================================================================
    with dose_tab:
        st.markdown(tr(
            "**Abstandsquadratgesetz:** Ḋ = A · Γ / r²",
            "**Inverse square law:** Ḋ = A · Γ / r²"
        ))
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            st.markdown(f"**{tr('Quelle', 'Source')}**")
            
            dose_nuclide = st.selectbox(
                tr("Radionuklid", "Radionuclide"),
                [n for n in COMMON_NUCLIDES.keys() if COMMON_NUCLIDES[n]["Gamma"] > 0],
                key="dose_nuclide"
            )
            
            Gamma = COMMON_NUCLIDES[dose_nuclide]["Gamma"]
            E_gamma = COMMON_NUCLIDES[dose_nuclide]["E_gamma"]
            
            st.info(f"Γ = {Gamma:.2e} Sv·m²/(Bq·s)")
            st.caption(f"Eγ = {E_gamma} MeV")
            
            A_dose = st.number_input(
                tr("Aktivität [GBq]", "Activity [GBq]"),
                min_value=0.001, max_value=1000.0, value=1.0,
                key="dose_A"
            ) * 1e9  # Bq
            
            st.markdown("---")
            st.markdown(f"**{tr('Abstandsbereich', 'Distance range')}**")
            
            r_min = st.number_input(
                tr("Min. Abstand [m]", "Min. distance [m]"),
                min_value=0.1, max_value=10.0, value=0.5,
                key="dose_r_min"
            )
            
            r_max = st.number_input(
                tr("Max. Abstand [m]", "Max. distance [m]"),
                min_value=1.0, max_value=100.0, value=10.0,
                key="dose_r_max"
            )
            
            exposure_time = st.number_input(
                tr("Aufenthaltszeit [h]", "Exposure time [h]"),
                min_value=0.1, max_value=2000.0, value=1.0,
                key="dose_time"
            )
        
        with col_plot:
            r = np.linspace(r_min, r_max, 200)
            
            # Dosisleistung berechnen [Sv/h]
            D_rate = np.array([dose_rate_point_source(A_dose, Gamma, ri) * 3600 for ri in r])
            
            # Dosis [mSv]
            D_total = D_rate * exposure_time * 1000  # mSv
            
            # Plot
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=(
                                   tr("Dosisleistung Ḋ(r)", "Dose rate Ḋ(r)"),
                                   tr(f"Dosis nach {exposure_time} h", f"Dose after {exposure_time} h")
                               ),
                               vertical_spacing=0.15)
            
            fig.add_trace(go.Scatter(
                x=r, y=D_rate * 1000,  # mSv/h
                mode='lines',
                name=tr("Dosisleistung", "Dose rate"),
                line=dict(color='#e53e3e', width=3),
                fill='tozeroy',
                fillcolor='rgba(229, 62, 62, 0.2)'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=r, y=D_total,
                mode='lines',
                name=tr("Gesamtdosis", "Total dose"),
                line=dict(color='#3182ce', width=3),
                fill='tozeroy',
                fillcolor='rgba(49, 130, 206, 0.2)'
            ), row=2, col=1)
            
            # Grenzwerte einzeichnen
            for limit_name, limit_val in [("1 mSv/a", 1.0), ("20 mSv/a", 20.0)]:
                fig.add_hline(
                    y=limit_val, row=2, col=1,
                    line=dict(color='orange', dash='dash'),
                    annotation_text=limit_name
                )
            
            fig.update_xaxes(title_text="r [m]", row=1, col=1)
            fig.update_xaxes(title_text="r [m]", row=2, col=1)
            fig.update_yaxes(title_text="Ḋ [mSv/h]", type="log", row=1, col=1)
            fig.update_yaxes(title_text="D [mSv]", type="log", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Zusammenfassung
            st.markdown(f"**{tr('Kennwerte', 'Key values')}**")
            col1, col2, col3 = st.columns(3)
            
            D_1m = dose_rate_point_source(A_dose, Gamma, 1.0) * 3600 * 1000  # mSv/h
            with col1:
                st.metric(tr("Ḋ bei 1 m", "Ḋ at 1 m"), f"{D_1m:.2f} mSv/h")
            
            # Abstand für 1 mSv/a bei 2000 h/a
            r_limit = np.sqrt(A_dose * Gamma * 3600 * 2000 / 0.001)
            with col2:
                st.metric(tr("Abstand für <1 mSv/a", "Distance for <1 mSv/a"), f"{r_limit:.1f} m")
            
            with col3:
                D_at_rmin = dose_rate_point_source(A_dose, Gamma, r_min) * 3600 * exposure_time * 1000
                st.metric(f"D @ {r_min} m", f"{D_at_rmin:.2f} mSv")
    
    # =========================================================================
    # Tab 4: Abschirmung
    # =========================================================================
    with shield_tab:
        st.markdown(tr(
            "**Schwächungsgesetz:** I = I₀ · e^(-μx)",
            "**Attenuation law:** I = I₀ · e^(-μx)"
        ))
        
        col_params, col_plot = st.columns([1, 2])
        
        with col_params:
            st.markdown(f"**{tr('Strahlung', 'Radiation')}**")
            
            shield_nuclide = st.selectbox(
                tr("Quelle", "Source"),
                [n for n in COMMON_NUCLIDES.keys() if COMMON_NUCLIDES[n]["E_gamma"] > 0],
                key="shield_nuclide"
            )
            
            E_gamma = COMMON_NUCLIDES[shield_nuclide]["E_gamma"]
            st.info(f"Eγ = {E_gamma} MeV")
            
            st.markdown("---")
            st.markdown(f"**{tr('Material', 'Material')}**")
            
            material = st.selectbox(
                tr("Abschirmmaterial", "Shielding material"),
                list(SHIELDING_MATERIALS.keys()),
                key="shield_material"
            )
            
            mat_data = SHIELDING_MATERIALS[material]
            
            # Interpoliere μ für gegebene Energie
            energies = sorted(mat_data["mu"].keys())
            mus = [mat_data["mu"][e] for e in energies]
            
            if E_gamma <= energies[0]:
                mu = mus[0]
            elif E_gamma >= energies[-1]:
                mu = mus[-1]
            else:
                mu = np.interp(E_gamma, energies, mus)
            
            st.caption(f"ρ = {mat_data['rho']} g/cm³")
            st.caption(f"μ ≈ {mu:.3f} cm⁻¹ @ {E_gamma} MeV")
            
            HVL = half_value_layer(mu)
            TVL = tenth_value_layer(mu)
            st.caption(f"HVL = {HVL:.2f} cm")
            st.caption(f"TVL = {TVL:.2f} cm")
            
            st.markdown("---")
            
            x_max = st.slider(
                tr("Max. Dicke [cm]", "Max. thickness [cm]"),
                min_value=1.0, max_value=50.0, value=10.0,
                key="shield_x_max"
            )
            
            target_reduction = st.selectbox(
                tr("Gewünschte Reduktion", "Target reduction"),
                ["1/2 (50%)", "1/10 (10%)", "1/100 (1%)", "1/1000 (0.1%)"],
                key="shield_target"
            )
        
        with col_plot:
            x = np.linspace(0, x_max, 200)
            
            # Transmission berechnen
            T = np.array([shielding_transmission(mu, xi) for xi in x])
            
            # Plot
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=(
                                   tr("Transmission T(x)", "Transmission T(x)"),
                                   tr("Materialvergleich", "Material comparison")
                               ))
            
            # Linkes Bild: Transmission für gewähltes Material
            fig.add_trace(go.Scatter(
                x=x, y=T * 100,
                mode='lines',
                name=material,
                line=dict(color=mat_data["color"], width=3),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(int(mat_data['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}"
            ), row=1, col=1)
            
            # HVL-Markierungen
            for n in range(1, int(x_max / HVL) + 1):
                x_hvl = n * HVL
                if x_hvl <= x_max:
                    fig.add_vline(
                        x=x_hvl, row=1, col=1,
                        line=dict(color='gray', dash='dot', width=1)
                    )
                    fig.add_annotation(
                        x=x_hvl, y=100 / (2 ** n),
                        text=f"{n}×HVL",
                        showarrow=False,
                        yshift=10,
                        row=1, col=1
                    )
            
            # Rechtes Bild: Vergleich aller Materialien
            for mat_name, mat_info in SHIELDING_MATERIALS.items():
                mat_mus = [mat_info["mu"][e] for e in energies]
                mat_mu = np.interp(E_gamma, energies, mat_mus) if E_gamma > 0 else mat_mus[0]
                
                T_mat = np.array([shielding_transmission(mat_mu, xi) for xi in x])
                
                fig.add_trace(go.Scatter(
                    x=x, y=T_mat * 100,
                    mode='lines',
                    name=mat_name,
                    line=dict(color=mat_info["color"], width=2)
                ), row=1, col=2)
            
            fig.update_xaxes(title_text=tr("Dicke x [cm]", "Thickness x [cm]"), row=1, col=1)
            fig.update_xaxes(title_text=tr("Dicke x [cm]", "Thickness x [cm]"), row=1, col=2)
            fig.update_yaxes(title_text=tr("Transmission [%]", "Transmission [%]"), 
                            type="log", row=1, col=1)
            fig.update_yaxes(title_text=tr("Transmission [%]", "Transmission [%]"), 
                            type="log", row=1, col=2)
            
            fig.update_layout(height=450, legend=dict(x=1.02, y=1))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Erforderliche Dicke berechnen
            reductions = {
                "1/2 (50%)": 0.5,
                "1/10 (10%)": 0.1,
                "1/100 (1%)": 0.01,
                "1/1000 (0.1%)": 0.001
            }
            red_factor = reductions[target_reduction]
            x_required = required_shielding(mu, red_factor)
            
            st.markdown(f"**{tr('Erforderliche Abschirmung', 'Required shielding')}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    tr(f"Für {target_reduction}", f"For {target_reduction}"),
                    f"{x_required:.2f} cm"
                )
            with col2:
                st.metric(
                    tr("Masse pro m²", "Mass per m²"),
                    f"{x_required * mat_data['rho'] * 10:.1f} kg/m²"
                )
            with col3:
                st.metric(
                    tr("Anzahl HVL", "Number of HVL"),
                    f"{x_required / HVL:.1f}"
                )
            
            # Dosisgrenzwerte
            st.markdown("---")
            st.markdown(f"**{tr('Dosisgrenzwerte (StrlSchV)', 'Dose limits (regulation)')}**")
            
            limits_df = [{"Kategorie": k, "Grenzwert": f"{v} mSv/a"} 
                        for k, v in DOSE_LIMITS.items()]
            st.dataframe(limits_df, use_container_width=True, hide_index=True)
