# 📖 Technical Documentation

**Physics Teaching Simulator v5.0**

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Modules in Detail](#2-modules-in-detail)
3. [Physical Foundations](#3-physical-foundations)
4. [API Reference](#4-api-reference)
5. [Extending the System](#5-extending-the-system)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    physics_sim.py                           │
│                  (Main Application)                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Tab 1    │  │ Tab 2    │  │ Tab 3    │  │ Tab 4-8  │   │
│  │Mechanics │  │ Thermo   │  │ Atomic   │  │ ...      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
├───────┼─────────────┼─────────────┼─────────────┼──────────┤
│       ▼             ▼             ▼             ▼          │
│  ui_mech_      ui_thermo_    ui_atom_      ui_*.py        │
│  bundle.py     bundle.py     bundle.py                     │
├─────────────────────────────────────────────────────────────┤
│                    sim_core_bundle.py                       │
│                  (Core Physics Functions)                   │
├─────────────────────────────────────────────────────────────┤
│  i18n_bundle.py          │  ultrasound_sim.py              │
│  (Translations)          │  xray_ct.py                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
User Input (Streamlit Widgets)
        │
        ▼
┌───────────────────┐
│ Parameter         │
│ Validation        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Physics           │
│ Calculation       │
│ (NumPy)           │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Visualization     │
│ (Plotly)          │
└────────┬──────────┘
         │
         ▼
    Browser Display
```

### 1.3 File Overview

| File | Lines | Description |
|------|-------|-------------|
| `physics_sim.py` | ~120 | Main application, tab structure |
| `ui_mech_bundle.py` | ~1400 | Mechanics UI and simulations |
| `ui_thermo_bundle.py` | ~600 | Thermodynamics UI |
| `ui_atom_bundle.py` | ~900 | Atomic physics UI |
| `ui_optics_bundle.py` | ~200 | Optics UI |
| `ui_med_bundle.py` | ~500 | MRI/CT UI |
| `ui_ultrasound.py` | ~150 | Ultrasound UI |
| `ultrasound_sim.py` | ~300 | Ultrasound physics |
| `sim_core_bundle.py` | ~400 | Core physics functions |
| `xray_ct.py` | ~250 | CT physics |
| `i18n_bundle.py` | ~100 | Internationalization |

---

## 2. Modules in Detail

### 2.1 Mechanics (`ui_mech_bundle.py`)

#### Data Classes

```python
@dataclass
class Body2D:
    """2D body for mechanics simulations"""
    name: str
    x: float           # Position x [m]
    y: float           # Position y [m]
    vx: float          # Velocity x [m/s]
    vy: float          # Velocity y [m/s]
    mass: float        # Mass [kg]
    radius: float      # Radius for collisions [m]
    color: str         # Color for visualization
    trail_x: List[float]  # Trajectory x
    trail_y: List[float]  # Trajectory y

@dataclass
class Body3D:
    """3D body for N-body simulations"""
    name: str
    pos: np.ndarray    # Position [x, y, z] in m
    vel: np.ndarray    # Velocity [vx, vy, vz] in m/s
    mass: float        # Mass [kg]
    radius: float      # Radius [m]
    color: str
    charge: float      # Electric charge [C]
    trail: List[np.ndarray]
```

#### Core Functions

```python
def projectile_motion(v0, angle_deg, h0=0, g=9.81, dt=0.01):
    """
    Projectile motion without air resistance.
    
    Args:
        v0: Initial velocity [m/s]
        angle_deg: Launch angle [°]
        h0: Initial height [m]
        g: Gravitational acceleration [m/s²]
        dt: Time step [s]
    
    Returns:
        t, x, y: Arrays with time, x-position, y-position
    """

def simple_pendulum(theta0_deg, L, g=9.81, t_max=10, dt=0.01):
    """
    Mathematical pendulum (non-linear).
    
    Differential equation: θ'' = -(g/L) * sin(θ)
    
    Args:
        theta0_deg: Initial displacement [°]
        L: Pendulum length [m]
    
    Returns:
        t, theta, omega: Time, angle [rad], angular velocity [rad/s]
    """

def inelastic_collision_1d(m1, v1, m2, v2, restitution=0.0):
    """
    1D collision with variable coefficient of restitution.
    
    Args:
        m1, m2: Masses [kg]
        v1, v2: Velocities before collision [m/s]
        restitution: 0 = perfectly inelastic, 1 = elastic
    
    Returns:
        v1_new, v2_new: Velocities after collision [m/s]
    """
```

#### N-Body Simulator

```python
class NBodySimulator:
    """
    N-body simulator with gravitation and collisions.
    
    Methods:
        compute_accelerations(): Calculates forces between all bodies
        step(dt, restitution): One time step (Velocity-Verlet)
        run(t_end, dt, restitution): Complete simulation
    
    Algorithm (Velocity-Verlet):
        1. a_old = compute_accelerations()
        2. pos += vel * dt + 0.5 * a_old * dt²
        3. a_new = compute_accelerations()
        4. vel += 0.5 * (a_old + a_new) * dt
    """
```

#### Solar System Data

```python
SOLAR_SYSTEM_DATA = {
    "Sun":     {"mass": 1.989e30, "distance": 0,        "velocity": 0,     "color": "#FFD700"},
    "Mercury": {"mass": 3.285e23, "distance": 0.387*AU, "velocity": 47870, "color": "#A0522D"},
    "Venus":   {"mass": 4.867e24, "distance": 0.723*AU, "velocity": 35020, "color": "#DEB887"},
    "Earth":   {"mass": 5.972e24, "distance": 1.000*AU, "velocity": 29780, "color": "#4169E1"},
    "Mars":    {"mass": 6.390e23, "distance": 1.524*AU, "velocity": 24130, "color": "#CD5C5C"},
    "Jupiter": {"mass": 1.898e27, "distance": 5.203*AU, "velocity": 13070, "color": "#F4A460"},
    "Saturn":  {"mass": 5.683e26, "distance": 9.537*AU, "velocity": 9690,  "color": "#DAA520"},
}
# AU = 1.495978707e11 m (Astronomical Unit)
```

---

### 2.2 Thermodynamics (`ui_thermo_bundle.py`)

#### Heat Conduction

```python
def heat_conduction_1d_step(T, alpha, dx, dt):
    """
    One time step of the 1D heat equation.
    
    Equation: ∂T/∂t = α * ∂²T/∂x²
    
    Method: Explicit Euler (FTCS)
    
    Stability condition (CFL): dt ≤ 0.5 * dx² / α
    
    Args:
        T: Temperature array [°C or K]
        alpha: Thermal diffusivity [m²/s]
        dx: Spatial step size [m]
        dt: Time step size [s]
    
    Returns:
        T_new: Updated temperature array
    """
    T_new = T.copy()
    r = alpha * dt / dx**2
    T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
    return T_new
```

#### State Changes

```python
def isothermal_process(p1, V1, V2):
    """Isothermal state change: T = const, pV = const"""
    V = np.linspace(V1, V2, 100)
    p = p1 * V1 / V
    W = p1 * V1 * np.log(V2 / V1)  # Work
    return V, p, W

def adiabatic_process(p1, V1, V2, gamma=1.4):
    """Adiabatic state change: Q = 0, pV^γ = const"""
    V = np.linspace(V1, V2, 100)
    p = p1 * (V1 / V)**gamma
    W = (p1*V1 - p[-1]*V2) / (gamma - 1)
    return V, p, W
```

#### Thermodynamic Cycles

```python
def carnot_cycle(T_hot, T_cold, V1, V2, V3):
    """
    Carnot cycle (ideal heat engine process).
    
    Steps:
        1. Isothermal expansion at T_hot
        2. Adiabatic expansion T_hot → T_cold
        3. Isothermal compression at T_cold
        4. Adiabatic compression T_cold → T_hot
    
    Efficiency: η = 1 - T_cold/T_hot (Carnot efficiency)
    """

def otto_cycle(T1, p1, r, gamma=1.4, Q_in=1000):
    """
    Otto cycle (gasoline engine).
    
    Steps:
        1. Isentropic compression (1→2)
        2. Isochoric heat addition (2→3)
        3. Isentropic expansion (3→4)
        4. Isochoric heat rejection (4→1)
    
    Efficiency: η = 1 - 1/r^(γ-1)
    
    Args:
        r: Compression ratio V1/V2
        gamma: Heat capacity ratio (air: 1.4)
        Q_in: Heat input [J]
    """
```

#### Gas Kinetics

```python
@dataclass
class Particle:
    """Gas particle for kinetic simulation"""
    x: float      # Position x [m]
    y: float      # Position y [m]
    vx: float     # Velocity x [m/s]
    vy: float     # Velocity y [m/s]
    mass: float   # Mass [kg]

def init_particles(N, T, box_size, mass=1e-26):
    """
    Initialize N particles with Maxwell-Boltzmann distribution.
    
    RMS velocity: v_rms = sqrt(3 * k_B * T / m)
    """

def compute_pressure(particles, box_size):
    """
    Calculate pressure from kinetic energy.
    
    2D formula: p = (2/V) * E_kin = (N/V) * m * <v²>
    """
```

---

### 2.3 Atomic Physics (`ui_atom_bundle.py`)

#### Physical Constants

```python
h = 6.62607015e-34      # Planck constant [J·s]
c = 299792458.0         # Speed of light [m/s]
e = 1.602176634e-19     # Elementary charge [C]
k_B = 1.380649e-23      # Boltzmann constant [J/K]
R_inf = 1.097373e7      # Rydberg constant [1/m]
E_H = 13.605693122      # Ionization energy H [eV]
a_0 = 5.29177210903e-11 # Bohr radius [m]
```

#### Bohr Model

```python
def bohr_energy(n, Z=1):
    """
    Energy in the n-th level of the Bohr model.
    
    E_n = -13.6 eV * Z² / n²
    
    Args:
        n: Principal quantum number (1, 2, 3, ...)
        Z: Nuclear charge number (1=H, 2=He⁺, 3=Li²⁺)
    
    Returns:
        E: Energy in eV (negative = bound)
    """
    return -E_H * Z**2 / n**2

def bohr_radius(n, Z=1):
    """
    Orbital radius in the n-th level.
    
    r_n = a_0 * n² / Z
    """
    return a_0 * n**2 / Z

def transition_wavelength(n_high, n_low, Z=1):
    """
    Wavelength for transition n_high → n_low.
    
    1/λ = R_inf * Z² * (1/n_low² - 1/n_high²)
    """
    delta_E = abs(bohr_energy(n_high, Z) - bohr_energy(n_low, Z)) * e  # in Joules
    wavelength = h * c / delta_E
    return wavelength * 1e9  # in nm
```

#### Spectral Series

| Series | n_final | Range | Wavelengths (H) |
|--------|---------|-------|-----------------|
| Lyman | 1 | UV | 91.2 - 121.6 nm |
| Balmer | 2 | Visible | 364.6 - 656.3 nm |
| Paschen | 3 | NIR | 820.4 - 1875 nm |
| Brackett | 4 | IR | 1458 - 4051 nm |
| Pfund | 5 | FIR | 2279 - 7460 nm |

#### Photoelectric Effect

```python
WORK_FUNCTIONS = {
    "Cs": 1.95,   # Cesium
    "K":  2.30,   # Potassium
    "Na": 2.75,   # Sodium
    "Zn": 4.33,   # Zinc
    "Cu": 4.65,   # Copper
    "Ag": 4.73,   # Silver
    "Pt": 5.65,   # Platinum
}

def photoelectric_kinetic_energy(wavelength_nm, work_function_eV):
    """
    Kinetic energy of photoelectrons.
    
    Einstein: E_kin = h*f - W = h*c/λ - W
    
    Returns:
        E_kin in eV (0 if λ > λ_threshold)
    """
    E_photon = h * c / (wavelength_nm * 1e-9) / e  # in eV
    E_kin = max(0, E_photon - work_function_eV)
    return E_kin

def threshold_wavelength(work_function_eV):
    """Threshold wavelength: λ_threshold = h*c/W"""
    return h * c / (work_function_eV * e) * 1e9  # in nm
```

#### Franck-Hertz

```python
FRANCK_HERTZ_DATA = {
    "Hg": {"excitation_eV": 4.9, "wavelength_nm": 253.7},  # Mercury
    "Ne": {"excitation_eV": 18.7, "wavelength_nm": 66.2},  # Neon
}

def franck_hertz_current(U, U_excitation, I_max=1.0):
    """
    Simulated current-voltage characteristic.
    
    Maxima at U = n * U_excitation (n = 1, 2, 3, ...)
    """
```

---

### 2.4 Ultrasound (`ultrasound_sim.py`)

#### Data Classes

```python
@dataclass
class Medium:
    c: float = 1540.0              # Speed of sound [m/s]
    rho: float = 1000.0            # Density [kg/m³]
    alpha_db_cm_mhz: float = 0.5   # Attenuation [dB/(MHz·cm)]

@dataclass
class Transducer:
    kind: str = "linear"           # linear | phased
    f0: float = 5e6                # Center frequency [Hz]
    pitch: float = 0.0003          # Element spacing [m]
    N: int = 64                    # Number of elements
    frac_bw: float = 0.6           # Fractional bandwidth
    tx_cycles: float = 2.0         # Transmit cycles

@dataclass
class Beamformer:
    method: str = "das"            # das | capon
    apod: str = "hanning"          # none | hanning | hamming
    tx_focus_z: float = 0.03       # Transmit focus [m]

@dataclass
class Scatterer:
    x: float                       # x-position [m]
    z: float                       # z-position (depth) [m]
    amp: float = 1.0               # Reflectivity
    vx: float = 0.0                # Lateral velocity [m/s]
    vz: float = 0.0                # Axial velocity [m/s]
```

#### Signal Processing

```python
def gaussian_pulse(f0, fs, cycles=2.0, frac_bw=0.6):
    """
    Band-limited transmit pulse with Gaussian envelope.
    
    Args:
        f0: Center frequency [Hz]
        fs: Sampling rate [Hz]
        cycles: Number of cycles at -6 dB
        frac_bw: Fractional bandwidth
    
    Returns:
        s: Pulse array (normalized)
    """

def analytic_signal(x, axis=-1):
    """
    Analytic signal via Hilbert transform.
    
    Implementation: FFT-based (without SciPy)
    
    z(t) = x(t) + j * H{x(t)}
    
    Envelope: |z(t)|
    """

def envelope_db(x, floor_db=-60.0):
    """
    Logarithmic envelope in dB.
    
    dB = 20 * log10(|z|/max|z|)
    """
```

#### Beamforming

```python
def das_beamform(rf, t, fs, x_elems, medium, Ximg, Zimg, apod=None):
    """
    Delay-and-Sum Beamforming for B-mode.
    
    Algorithm:
        1. For each image point (x, z):
        2.   Calculate travel time τ = (r_tx + r_rx) / c
        3.   Interpolate RF signal at τ
        4.   Sum over all channels (with apodization)
        5.   Calculate envelope
    
    Args:
        rf: RF data (ensembles, Nch, Nt)
        t: Time axis [s]
        x_elems: Element positions [m]
        medium: Medium object
        Ximg, Zimg: Image coordinates [m]
        apod: Apodization weights
    
    Returns:
        img: B-mode image (nz, nx)
    """
```

#### Doppler

```python
def doppler_autocorr(rf, lag=1):
    """
    Kasai autocorrelation for Doppler estimation.
    
    R(lag) = Σ y[n] * conj(y[n+lag])
    
    Doppler frequency: f_d = angle(R) / (2π * PRI)
    Velocity:          v = f_d * c / (2 * f0)
    """

def doppler_color_map(rf, t, fs, f0, c, ensembles, pri):
    """
    Color Doppler map (velocity + power).
    
    Returns:
        v: Velocity map [m/s]
        pwr: Power map [dB]
    """
```

#### Phantoms

```python
def carotid_phantom(width=0.006, depth=0.025, flow_vel=0.3, seed=0):
    """
    Carotid phantom with laminar flow.
    
    Contains:
        - Random tissue speckle
        - Cylindrical vessel
        - Parabolic flow profile: v(r) = v_max * (1 - (r/R)²)
    
    Args:
        width: Vessel diameter [m]
        depth: Vessel depth [m]
        flow_vel: Maximum flow velocity [m/s]
    
    Returns:
        List[Scatterer]: Point scatterers with velocities
    """
```

---

## 3. Physical Foundations

### 3.1 Numerical Integration

#### Euler Method (1st Order)
```
y_{n+1} = y_n + h * f(t_n, y_n)
```
- Simple but only 1st order accuracy
- Used for: Heat conduction, simple oscillations

#### Velocity-Verlet (2nd Order, Symplectic)
```
x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt²
a_{n+1} = F(x_{n+1}) / m
v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
```
- Energy-conserving for conservative systems
- Used for: N-body, planetary orbits

#### Leapfrog (2nd Order, Symplectic)
```
v_{n+1/2} = v_{n-1/2} + a_n * dt
x_{n+1} = x_n + v_{n+1/2} * dt
```
- Time-reversal symmetric
- Used for: Coupled oscillators

### 3.2 Stability Conditions

#### Heat Conduction (explicit)
```
CFL condition: dt ≤ dx² / (2α)   (1D)
               dt ≤ dx² / (4α)   (2D)
```

#### Wave Equation
```
CFL condition: dt ≤ dx / c
```

### 3.3 Unit System

| Quantity | SI Unit | Typical Range |
|----------|---------|---------------|
| Length | m | 1e-12 ... 1e12 |
| Time | s | 1e-9 ... 1e9 |
| Mass | kg | 1e-30 ... 1e30 |
| Temperature | K | 0 ... 10000 |
| Energy | J | 1e-30 ... 1e30 |
| Frequency | Hz | 1 ... 1e15 |

---

## 4. API Reference

### 4.1 Main Functions

```python
# Mechanics
render_mechanics_tab()           # Main mechanics tab
render_2d_mechanics_tab()        # 2D simulations
render_3d_nbody_tab()            # N-body
render_celestial_tab()           # Celestial mechanics
render_collisions_tab()          # Collisions

# Thermodynamics
render_thermo_tab()              # Main tab
render_heat_conduction_tab()     # Heat conduction
render_state_changes_tab()       # State changes
render_cycles_tab()              # Thermodynamic cycles
render_kinetic_tab()             # Gas kinetics

# Atomic Physics
render_atom_tab()                # Main tab
render_bohr_tab()                # Bohr model
render_photoeffect_tab()         # Photoelectric effect
render_franck_hertz_tab()        # Franck-Hertz
render_spectra_tab()             # Spectroscopy

# Ultrasound
render_ultrasound_tab()          # Main tab

# Medical Technology
render_mri_bloch_tab()           # MRI
render_xray_ct_tab()             # CT
```

### 4.2 Internationalization

```python
from i18n_bundle import get_text, get_language_name

# Get translation
label = get_text("start_animation", lang)  # "▶️ Start animation" (en)

# Add new translation
TRANSLATIONS["my_key"] = {
    "de": "Mein Text",
    "en": "My Text"
}
```

---

## 5. Extending the System

### 5.1 Adding a New Module

1. **Create physics file** (`my_physics.py`):
```python
def my_simulation(param1, param2):
    """Documentation..."""
    # Calculation
    return results
```

2. **Create UI file** (`ui_my_module.py`):
```python
import streamlit as st
from my_physics import my_simulation

def render_my_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("Mein Modul", "My Module"))
    
    # Parameters
    param1 = st.slider("Parameter 1", 0.0, 1.0, 0.5)
    
    if st.button(tr("Starten", "Start")):
        results = my_simulation(param1, param2)
        # Visualization...
```

3. **Integrate in `physics_sim.py`**:
```python
from ui_my_module import render_my_tab

# Add to tabs_labels
tabs_labels = [..., "My Module"]

# Render tab
with selected_tabs[N]:
    render_my_tab()
```

4. **Add translations in `i18n_bundle.py`**:
```python
TRANSLATIONS["my_module"] = {"de": "Mein Modul", "en": "My Module"}
```

### 5.2 Adding Animations

```python
def run_my_animation(data):
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    # Placeholder for live updates
    chart_placeholder = st.empty()
    progress = st.progress(0)
    
    n_frames = 100
    for frame in range(n_frames):
        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['x'][:frame],
            y=data['y'][:frame],
            mode='lines'
        ))
        fig.update_layout(height=400)
        
        # Update
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        progress.progress((frame + 1) / n_frames)
        time.sleep(0.03)  # ~30 FPS
    
    st.success(tr("✅ Animation abgeschlossen", "✅ Animation complete"))
```

---

## 6. Troubleshooting

### 6.1 Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| Slow animation | Too many frames | Reduce `n_frames` |
| Numerical instability | dt too large | Check CFL condition |
| Memory error | Arrays too large | Reduce resolution |

### 6.2 Performance Optimization

1. **Use NumPy vectorization**:
```python
# Bad
for i in range(N):
    result[i] = a[i] * b[i]

# Good
result = a * b
```

2. **Minimize Plotly updates**:
```python
# Don't update every frame
if frame % 5 == 0:
    chart_placeholder.plotly_chart(fig)
```

3. **Use session state for caching**:
```python
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = run_expensive_simulation()
```

### 6.3 Debug Mode

```python
# In physics_sim.py
DEBUG = True

if DEBUG:
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Language: {st.session_state.language}")
    st.sidebar.write(f"Session keys: {list(st.session_state.keys())}")
```

---

## Appendix

### A. Constants Reference

```python
# Physical constants (CODATA 2018)
c = 299792458           # Speed of light [m/s]
h = 6.62607015e-34      # Planck constant [J·s]
hbar = 1.054571817e-34  # Reduced Planck constant [J·s]
e = 1.602176634e-19     # Elementary charge [C]
m_e = 9.1093837015e-31  # Electron mass [kg]
m_p = 1.67262192369e-27 # Proton mass [kg]
k_B = 1.380649e-23      # Boltzmann constant [J/K]
N_A = 6.02214076e23     # Avogadro constant [1/mol]
R = 8.314462618         # Gas constant [J/(mol·K)]
G = 6.67430e-11         # Gravitational constant [m³/(kg·s²)]
epsilon_0 = 8.8541878e-12  # Electric constant [F/m]
mu_0 = 1.25663706e-6    # Magnetic constant [H/m]

# Astronomical constants
AU = 1.495978707e11     # Astronomical unit [m]
pc = 3.0856776e16       # Parsec [m]
M_sun = 1.98847e30      # Solar mass [kg]
R_sun = 6.9634e8        # Solar radius [m]
M_earth = 5.9722e24     # Earth mass [kg]
R_earth = 6.371e6       # Earth radius [m]
```

### B. References

1. Landau, L.D. & Lifshitz, E.M.: *Mechanics* (Course of Theoretical Physics, Vol. 1)
2. Demtröder, W.: *Experimental Physics 1-4* (Springer)
3. Jensen, J.A.: *Medical Ultrasound Imaging* (Wiley)
4. Haacke, E.M.: *Magnetic Resonance Imaging* (Wiley)

---

*Documentation Version 5.0 | Date: November 2024*
