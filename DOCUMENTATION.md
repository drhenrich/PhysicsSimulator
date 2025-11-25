# 📘 Physik-Simulator — Technische Dokumentation

## Inhaltsverzeichnis

1. [Architektur](#architektur)
2. [Module im Detail](#module-im-detail)
3. [Animationssystem](#animationssystem)
4. [Physikalische Modelle](#physikalische-modelle)
5. [Internationalisierung](#internationalisierung)
6. [Erweiterung](#erweiterung)

---

## Architektur

### Übersicht

```
┌─────────────────────────────────────────────────────────┐
│                    physics_sim.py                        │
│                   (Hauptanwendung)                       │
├─────────────────────────────────────────────────────────┤
│  i18n_bundle.py  │  sim_core_bundle.py                  │
│  (Übersetzungen) │  (Kernfunktionen)                    │
├─────────────────────────────────────────────────────────┤
│  UI-Module (ui_*.py)                                    │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ Mechanik │ Thermo   │ Atom     │ Schwing. │         │
│  ├──────────┼──────────┼──────────┼──────────┤         │
│  │ Optik    │ Kernphys │ Med/CT   │ US/MRI   │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────────────────────────────────────────────────────┘
```

### Dateiübersicht

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `physics_sim.py` | ~130 | Hauptanwendung, Tab-Routing |
| `i18n_bundle.py` | ~200 | 120+ Übersetzungspaare |
| `sim_core_bundle.py` | ~400 | Physik-Kernfunktionen |
| `ui_mech_bundle.py` | ~1400 | Mechanik & Himmelsmechanik |
| `ui_thermo_bundle.py` | ~850 | Thermodynamik |
| `ui_atom_bundle.py` | ~1000 | Atomphysik |
| `ui_oscillations_bundle.py` | ~1100 | Schwingungen & Akustik |
| `ui_optics_bundle.py` | ~200 | Optik |
| `ui_nuclear_bundle.py` | ~850 | Kernphysik & Strahlenschutz |
| `ui_med_bundle.py` | ~700 | CT, MRI, Elektrodynamik |
| `ui_ultrasound.py` | ~150 | Ultraschall-UI |
| `ultrasound_sim.py` | ~300 | Ultraschall-Simulation |
| `xray_ct.py` | ~250 | CT-Rekonstruktion |

---

## Module im Detail

### 1. Mechanik (`ui_mech_bundle.py`)

#### Schiefer Wurf
- Parameter: v₀, α, h₀, Luftwiderstand
- Berechnung: Euler-Integration mit optionalem Drag
- Animation: Plotly-Frames (80 Frames)

#### Pendel
- Mathematisches Pendel mit beliebiger Amplitude
- Lösung: Runge-Kutta 4
- Phasenraumdarstellung

#### N-Körper-Simulation
- Gravitationssimulation für 2-10 Körper
- Presets: Orbit, chaotisch, Figure-8
- 3D-Visualisierung mit Plotly

#### Stöße
- 1D: Elastisch/Inelastisch mit Stoßzahl e
- 2D: Stoßparameter, Impulserhaltung
- Billard: Mehrere Kugeln mit Reibung

### 2. Thermodynamik (`ui_thermo_bundle.py`)

#### Wärmeleitung
```python
# 1D Fourier-Gleichung
∂T/∂t = α · ∂²T/∂x²

# Diskretisierung (explizit)
T[i]_new = T[i] + α·dt/dx² · (T[i+1] - 2·T[i] + T[i-1])
```
- CFL-Bedingung: dt ≤ 0.5·dx²/α
- Animierte Heatmap für 2D

#### Kreisprozesse
- Carnot: Isothermen + Adiabaten
- Otto: Isochoren + Adiabaten
- pV-Diagramme mit Flächenintegration

#### Gaskinetik
- Maxwell-Boltzmann-Verteilung
- Partikelanimation mit Wandstößen
- Druckberechnung aus Impulsübertrag

### 3. Atomphysik (`ui_atom_bundle.py`)

#### Bohr-Modell
```python
# Energie der Bahnen
E_n = -13.6 eV / n²

# Photonenwellenlänge
λ = h·c / (E_i - E_f)
```
- Animation des Elektronenübergangs
- Serien: Lyman, Balmer, Paschen

#### Photoeffekt
```python
E_kin = h·f - W_A
```
- Materialauswahl (Cs, Na, Cu, Pt)
- Kennlinien I(U)

#### Franck-Hertz
- Simulation der charakteristischen Kurve
- Animierte Messung

### 4. Schwingungen (`ui_oscillations_bundle.py`)

#### Harmonischer Oszillator
```python
m·ẍ + b·ẋ + k·x = 0

# Lösung (unterdämpft)
x(t) = A·e^(-γt)·cos(ωd·t + φ)
ωd = √(ω₀² - γ²)
```

#### Gekoppelte Oszillatoren
- Normalmoden: Gleichtakt/Gegentakt
- Energieaustausch zwischen Oszillatoren

#### Schwebungen
```python
y(t) = A₁·sin(2πf₁t) + A₂·sin(2πf₂t)
f_beat = |f₁ - f₂|
```
- FFT-Spektrum
- Einhüllende

#### Stehende Wellen
```python
y(x,t) = 2A·sin(kx)·cos(ωt)
```
- Animierte Darstellung
- Knoten und Bäuche markiert
- Harmonische n = 1...6

#### Doppler-Effekt
```python
f' = f₀ · (c ± v_o) / (c ∓ v_s)
```
- Animierte Wellenfronten
- Mach-Kegel bei Überschall

### 5. Kernphysik (`ui_nuclear_bundle.py`)

#### Radioaktiver Zerfall
```python
A(t) = A₀ · e^(-λt)
λ = ln(2) / T½
```
- 10 Radionuklide mit realen Daten
- Logarithmische Darstellung

#### Zerfallsreihen
- Natürliche Reihen: U-238, Th-232, U-235
- Bateman-Gleichungen
- Numerische Lösung (Euler)

#### Dosimetrie
```python
Ḋ = A · Γ / r²
```
- Abstandsgesetz
- Grenzwerte nach StrlSchV

#### Abschirmung
```python
I = I₀ · e^(-μx)
HVL = ln(2) / μ
```
- 5 Materialien (Pb, Beton, H₂O, Fe, Al)
- Energieabhängige μ-Werte

### 6. Medizinphysik

#### CT-Rekonstruktion (`xray_ct.py`)
- Radon-Transformation
- Gefilterte Rückprojektion
- Hounsfield-Skala

#### MRI (`ui_med_bundle.py`)
- Bloch-Gleichungen
- T1/T2-Relaxation
- Spinecho-Sequenz

#### Ultraschall (`ultrasound_sim.py`)
- Beamforming
- B-Mode-Darstellung
- Schallgeschwindigkeit in Gewebe

#### Elektrostatik
- Feldstärke und Potential
- Farbige Heatmaps
- Poisson-Gleichung (Jacobi)

---

## Animationssystem

### Plotly-Frame-Animationen

Alle Animationen wurden auf clientseitige Plotly-Frames umgestellt:

```python
# 1. Simulation vorberechnen
frames = []
for i in range(n_frames):
    # Physik-Update
    state = compute_next_state(state)
    
    # Frame erstellen
    frames.append(go.Frame(
        data=[go.Scatter(x=..., y=...)],
        name=str(i)
    ))

# 2. Figur mit Controls
fig = go.Figure(data=[...], frames=frames)

fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(label="▶️ Play", method="animate",
                 args=[None, {"frame": {"duration": 50}}]),
            dict(label="⏸️ Pause", method="animate",
                 args=[[None], {"mode": "immediate"}]),
            dict(label="🔄 Reset", method="animate",
                 args=[["0"], {"mode": "immediate"}])
        ]
    )]
)
```

### Vorteile

| Server-Animation | Client-Animation |
|------------------|------------------|
| ~5 FPS | bis 60 FPS |
| Netzwerk-Latenz | Lokal |
| Keine Kontrolle | Play/Pause/Reset |
| Blockiert UI | Nicht-blockierend |

---

## Physikalische Modelle

### Numerische Methoden

#### Runge-Kutta 4
```python
def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

#### Jacobi-Iteration
```python
def jacobi_step(phi, rho, dx):
    phi_new = np.zeros_like(phi)
    phi_new[1:-1, 1:-1] = 0.25 * (
        phi[:-2, 1:-1] + phi[2:, 1:-1] +
        phi[1:-1, :-2] + phi[1:-1, 2:] -
        dx**2 * rho[1:-1, 1:-1]
    )
    return phi_new
```

### Konstanten

```python
# Naturkonstanten
c = 299792458       # m/s
h = 6.62607e-34     # J·s
h_eV = 4.13567e-15  # eV·s
k_B = 1.38065e-23   # J/K
e = 1.60218e-19     # C
m_e = 9.10938e-31   # kg
N_A = 6.02214e23    # 1/mol

# Abgeleitete
a_0 = 5.29177e-11   # Bohr-Radius
R_inf = 1.09737e7   # Rydberg-Konstante
```

---

## Internationalisierung

### Übersetzungssystem

```python
# i18n_bundle.py
TRANSLATIONS = {
    "mechanics": {"de": "⚙️ Mechanik", "en": "⚙️ Mechanics"},
    "velocity": {"de": "Geschwindigkeit", "en": "Velocity"},
    # ...
}

def get_text(key: str, lang: str = "de") -> str:
    return TRANSLATIONS.get(key, {}).get(lang, key)
```

### Verwendung in Modulen

```python
lang = st.session_state.get("language", "de")
tr = lambda de, en: de if lang == "de" else en

st.markdown(tr("### Einstellungen", "### Settings"))
```

---

## Erweiterung

### Neues Modul hinzufügen

1. **UI-Datei erstellen**: `ui_newmodule_bundle.py`

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render_newmodule_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Neues Modul", "### New Module"))
    
    # Parameter
    param = st.slider("Parameter", 0.0, 10.0, 5.0)
    
    # Berechnung
    x = np.linspace(0, 10, 100)
    y = np.sin(param * x)
    
    # Visualisierung
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    st.plotly_chart(fig)
```

2. **In physics_sim.py einbinden**:

```python
from ui_newmodule_bundle import render_newmodule_tab

# In der Tab-Liste
tabs = st.tabs([..., tr("Neues Modul", "New Module")])

# Im Tab-Handling
with tabs[N]:
    render_newmodule_tab()
```

3. **Übersetzungen ergänzen** (`i18n_bundle.py`):

```python
TRANSLATIONS.update({
    "new_module": {"de": "Neues Modul", "en": "New Module"},
    # ...
})
```

### Animation hinzufügen

```python
def run_animation(param):
    n_frames = 60
    
    # Vorberechnung
    frames = []
    for i in range(n_frames):
        state = compute_state(i, param)
        frames.append(go.Frame(
            data=[go.Scatter(x=state.x, y=state.y)],
            name=str(i)
        ))
    
    # Figur
    fig = go.Figure(data=[...], frames=frames)
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶️", method="animate",
                     args=[None, {"frame": {"duration": 50}}]),
                dict(label="⏸️", method="animate",
                     args=[[None], {"mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig)
```

---

## Changelog

Siehe [CHANGELOG.md](CHANGELOG.md) für Versionshistorie.

---

<p align="center">
  <b>Physik-Simulator v6.0</b><br>
  <i>Prof. Dr. Dietmar Henrich</i>
</p>
