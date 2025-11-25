# 📘 Physics Simulator — Technical Documentation

## Table of Contents

1. [Architecture](#architecture)
2. [Modules in Detail](#modules-in-detail)
3. [Animation System](#animation-system)
4. [Physical Models](#physical-models)
5. [Internationalization](#internationalization)
6. [Extension](#extension)

---

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                    physics_sim.py                        │
│                  (Main Application)                      │
├─────────────────────────────────────────────────────────┤
│  i18n_bundle.py  │  sim_core_bundle.py                  │
│  (Translations)  │  (Core Functions)                    │
├─────────────────────────────────────────────────────────┤
│  UI Modules (ui_*.py)                                   │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │Mechanics │ Thermo   │ Atomic   │ Oscill.  │         │
│  ├──────────┼──────────┼──────────┼──────────┤         │
│  │ Optics   │ Nuclear  │ Med/CT   │ US/MRI   │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────────────────────────────────────────────────────┘
```

### File Overview

| File | Lines | Description |
|------|-------|-------------|
| `physics_sim.py` | ~130 | Main application, tab routing |
| `i18n_bundle.py` | ~200 | 120+ translation pairs |
| `sim_core_bundle.py` | ~400 | Physics core functions |
| `ui_mech_bundle.py` | ~1400 | Mechanics & celestial mechanics |
| `ui_thermo_bundle.py` | ~850 | Thermodynamics |
| `ui_atom_bundle.py` | ~1000 | Atomic physics |
| `ui_oscillations_bundle.py` | ~1100 | Oscillations & acoustics |
| `ui_optics_bundle.py` | ~200 | Optics |
| `ui_nuclear_bundle.py` | ~850 | Nuclear physics & radiation protection |
| `ui_med_bundle.py` | ~700 | CT, MRI, electrodynamics |
| `ui_ultrasound.py` | ~150 | Ultrasound UI |
| `ultrasound_sim.py` | ~300 | Ultrasound simulation |
| `xray_ct.py` | ~250 | CT reconstruction |

---

## Modules in Detail

### 1. Mechanics (`ui_mech_bundle.py`)

#### Projectile Motion
- Parameters: v₀, α, h₀, air resistance
- Calculation: Euler integration with optional drag
- Animation: Plotly frames (80 frames)

#### Pendulum
- Mathematical pendulum with arbitrary amplitude
- Solution: Runge-Kutta 4
- Phase space representation

#### N-Body Simulation
- Gravitational simulation for 2-10 bodies
- Presets: Orbit, chaotic, Figure-8
- 3D visualization with Plotly

#### Collisions
- 1D: Elastic/inelastic with coefficient of restitution e
- 2D: Impact parameter, momentum conservation
- Billiard: Multiple balls with friction

### 2. Thermodynamics (`ui_thermo_bundle.py`)

#### Heat Conduction
```python
# 1D Fourier equation
∂T/∂t = α · ∂²T/∂x²

# Discretization (explicit)
T[i]_new = T[i] + α·dt/dx² · (T[i+1] - 2·T[i] + T[i-1])
```
- CFL condition: dt ≤ 0.5·dx²/α
- Animated heatmap for 2D

#### Thermodynamic Cycles
- Carnot: Isotherms + adiabats
- Otto: Isochores + adiabats
- pV diagrams with area integration

#### Gas Kinetics
- Maxwell-Boltzmann distribution
- Particle animation with wall collisions
- Pressure calculation from momentum transfer

### 3. Atomic Physics (`ui_atom_bundle.py`)

#### Bohr Model
```python
# Energy levels
E_n = -13.6 eV / n²

# Photon wavelength
λ = h·c / (E_i - E_f)
```
- Animation of electron transition
- Series: Lyman, Balmer, Paschen

#### Photoelectric Effect
```python
E_kin = h·f - W_A
```
- Material selection (Cs, Na, Cu, Pt)
- I(U) characteristics

#### Franck-Hertz
- Simulation of characteristic curve
- Animated measurement

### 4. Oscillations (`ui_oscillations_bundle.py`)

#### Harmonic Oscillator
```python
m·ẍ + b·ẋ + k·x = 0

# Solution (underdamped)
x(t) = A·e^(-γt)·cos(ωd·t + φ)
ωd = √(ω₀² - γ²)
```

#### Coupled Oscillators
- Normal modes: In-phase/anti-phase
- Energy exchange between oscillators

#### Beats
```python
y(t) = A₁·sin(2πf₁t) + A₂·sin(2πf₂t)
f_beat = |f₁ - f₂|
```
- FFT spectrum
- Envelope

#### Standing Waves
```python
y(x,t) = 2A·sin(kx)·cos(ωt)
```
- Animated display
- Nodes and antinodes marked
- Harmonics n = 1...6

#### Doppler Effect
```python
f' = f₀ · (c ± v_o) / (c ∓ v_s)
```
- Animated wavefronts
- Mach cone at supersonic speeds

### 5. Nuclear Physics (`ui_nuclear_bundle.py`)

#### Radioactive Decay
```python
A(t) = A₀ · e^(-λt)
λ = ln(2) / T½
```
- 10 radionuclides with real data
- Logarithmic display

#### Decay Chains
- Natural series: U-238, Th-232, U-235
- Bateman equations
- Numerical solution (Euler)

#### Dosimetry
```python
Ḋ = A · Γ / r²
```
- Inverse square law
- Dose limits per regulations

#### Shielding
```python
I = I₀ · e^(-μx)
HVL = ln(2) / μ
```
- 5 materials (Pb, concrete, H₂O, Fe, Al)
- Energy-dependent μ values

### 6. Medical Physics

#### CT Reconstruction (`xray_ct.py`)
- Radon transform
- Filtered back projection
- Hounsfield scale

#### MRI (`ui_med_bundle.py`)
- Bloch equations
- T1/T2 relaxation
- Spin echo sequence

#### Ultrasound (`ultrasound_sim.py`)
- Beamforming
- B-mode display
- Sound velocity in tissue

#### Electrostatics
- Field strength and potential
- Color heatmaps
- Poisson equation (Jacobi)

---

## Animation System

### Plotly Frame Animations

All animations have been converted to client-side Plotly frames:

```python
# 1. Pre-compute simulation
frames = []
for i in range(n_frames):
    # Physics update
    state = compute_next_state(state)
    
    # Create frame
    frames.append(go.Frame(
        data=[go.Scatter(x=..., y=...)],
        name=str(i)
    ))

# 2. Figure with controls
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

### Advantages

| Server Animation | Client Animation |
|------------------|------------------|
| ~5 FPS | up to 60 FPS |
| Network latency | Local |
| No control | Play/Pause/Reset |
| Blocks UI | Non-blocking |

---

## Physical Models

### Numerical Methods

#### Runge-Kutta 4
```python
def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

#### Jacobi Iteration
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

### Constants

```python
# Physical constants
c = 299792458       # m/s
h = 6.62607e-34     # J·s
h_eV = 4.13567e-15  # eV·s
k_B = 1.38065e-23   # J/K
e = 1.60218e-19     # C
m_e = 9.10938e-31   # kg
N_A = 6.02214e23    # 1/mol

# Derived
a_0 = 5.29177e-11   # Bohr radius
R_inf = 1.09737e7   # Rydberg constant
```

---

## Internationalization

### Translation System

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

### Usage in Modules

```python
lang = st.session_state.get("language", "de")
tr = lambda de, en: de if lang == "de" else en

st.markdown(tr("### Settings", "### Settings"))
```

---

## Extension

### Adding a New Module

1. **Create UI file**: `ui_newmodule_bundle.py`

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render_newmodule_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### New Module", "### New Module"))
    
    # Parameters
    param = st.slider("Parameter", 0.0, 10.0, 5.0)
    
    # Calculation
    x = np.linspace(0, 10, 100)
    y = np.sin(param * x)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    st.plotly_chart(fig)
```

2. **Integrate in physics_sim.py**:

```python
from ui_newmodule_bundle import render_newmodule_tab

# In tab list
tabs = st.tabs([..., tr("New Module", "New Module")])

# In tab handling
with tabs[N]:
    render_newmodule_tab()
```

3. **Add translations** (`i18n_bundle.py`):

```python
TRANSLATIONS.update({
    "new_module": {"de": "Neues Modul", "en": "New Module"},
    # ...
})
```

### Adding Animation

```python
def run_animation(param):
    n_frames = 60
    
    # Pre-computation
    frames = []
    for i in range(n_frames):
        state = compute_state(i, param)
        frames.append(go.Frame(
            data=[go.Scatter(x=state.x, y=state.y)],
            name=str(i)
        ))
    
    # Figure
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

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

<p align="center">
  <b>Physics Simulator v6.0</b><br>
  <i>Prof. Dr. Dietmar Henrich</i>
</p>
