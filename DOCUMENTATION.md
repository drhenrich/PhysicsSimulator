# 📖 Technische Dokumentation

**Physics Teaching Simulator v5.0**

---

## Inhaltsverzeichnis

1. [Architektur](#1-architektur)
2. [Module im Detail](#2-module-im-detail)
3. [Physikalische Grundlagen](#3-physikalische-grundlagen)
4. [API-Referenz](#4-api-referenz)
5. [Erweiterung des Systems](#5-erweiterung-des-systems)
6. [Fehlerbehebung](#6-fehlerbehebung)

---

## 1. Architektur

### 1.1 Systemübersicht

```
┌─────────────────────────────────────────────────────────────┐
│                    physics_sim.py                           │
│                   (Hauptanwendung)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Tab 1    │  │ Tab 2    │  │ Tab 3    │  │ Tab 4-8  │   │
│  │ Mechanik │  │ Thermo   │  │ Atom     │  │ ...      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
├───────┼─────────────┼─────────────┼─────────────┼──────────┤
│       ▼             ▼             ▼             ▼          │
│  ui_mech_      ui_thermo_    ui_atom_      ui_*.py        │
│  bundle.py     bundle.py     bundle.py                     │
├─────────────────────────────────────────────────────────────┤
│                    sim_core_bundle.py                       │
│                  (Physik-Kernfunktionen)                    │
├─────────────────────────────────────────────────────────────┤
│  i18n_bundle.py          │  ultrasound_sim.py              │
│  (Übersetzungen)         │  xray_ct.py                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Datenfluss

```
Benutzer-Input (Streamlit Widgets)
        │
        ▼
┌───────────────────┐
│ Parameter-        │
│ Validierung       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Physik-           │
│ Berechnung        │
│ (NumPy)           │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Visualisierung    │
│ (Plotly)          │
└────────┬──────────┘
         │
         ▼
    Anzeige im Browser
```

### 1.3 Dateiübersicht

| Datei | Zeilen | Beschreibung |
|-------|--------|--------------|
| `physics_sim.py` | ~120 | Hauptanwendung, Tab-Struktur |
| `ui_mech_bundle.py` | ~1400 | Mechanik-UI und Simulationen |
| `ui_thermo_bundle.py` | ~600 | Thermodynamik-UI |
| `ui_atom_bundle.py` | ~900 | Atomphysik-UI |
| `ui_optics_bundle.py` | ~200 | Optik-UI |
| `ui_med_bundle.py` | ~500 | MRI/CT-UI |
| `ui_ultrasound.py` | ~150 | Ultraschall-UI |
| `ultrasound_sim.py` | ~300 | Ultraschall-Physik |
| `sim_core_bundle.py` | ~400 | Kern-Physikfunktionen |
| `xray_ct.py` | ~250 | CT-Physik |
| `i18n_bundle.py` | ~100 | Internationalisierung |

---

## 2. Module im Detail

### 2.1 Mechanik (`ui_mech_bundle.py`)

#### Datenklassen

```python
@dataclass
class Body2D:
    """2D-Körper für Mechanik-Simulationen"""
    name: str
    x: float           # Position x [m]
    y: float           # Position y [m]
    vx: float          # Geschwindigkeit x [m/s]
    vy: float          # Geschwindigkeit y [m/s]
    mass: float        # Masse [kg]
    radius: float      # Radius für Kollisionen [m]
    color: str         # Farbe für Visualisierung
    trail_x: List[float]  # Trajektorie x
    trail_y: List[float]  # Trajektorie y

@dataclass
class Body3D:
    """3D-Körper für N-Körper-Simulationen"""
    name: str
    pos: np.ndarray    # Position [x, y, z] in m
    vel: np.ndarray    # Geschwindigkeit [vx, vy, vz] in m/s
    mass: float        # Masse [kg]
    radius: float      # Radius [m]
    color: str
    charge: float      # Elektrische Ladung [C]
    trail: List[np.ndarray]
```

#### Kernfunktionen

```python
def projectile_motion(v0, angle_deg, h0=0, g=9.81, dt=0.01):
    """
    Schiefer Wurf ohne Luftwiderstand.
    
    Args:
        v0: Anfangsgeschwindigkeit [m/s]
        angle_deg: Abwurfwinkel [°]
        h0: Anfangshöhe [m]
        g: Erdbeschleunigung [m/s²]
        dt: Zeitschritt [s]
    
    Returns:
        t, x, y: Arrays mit Zeit, x-Position, y-Position
    """

def simple_pendulum(theta0_deg, L, g=9.81, t_max=10, dt=0.01):
    """
    Mathematisches Pendel (nicht-linear).
    
    Differentialgleichung: θ'' = -(g/L) * sin(θ)
    
    Args:
        theta0_deg: Anfangsauslenkung [°]
        L: Pendellänge [m]
    
    Returns:
        t, theta, omega: Zeit, Winkel [rad], Winkelgeschwindigkeit [rad/s]
    """

def inelastic_collision_1d(m1, v1, m2, v2, restitution=0.0):
    """
    1D-Stoß mit variabler Stoßzahl.
    
    Args:
        m1, m2: Massen [kg]
        v1, v2: Geschwindigkeiten vor Stoß [m/s]
        restitution: 0 = vollständig inelastisch, 1 = elastisch
    
    Returns:
        v1_new, v2_new: Geschwindigkeiten nach Stoß [m/s]
    """
```

#### N-Körper-Simulator

```python
class NBodySimulator:
    """
    N-Körper-Simulator mit Gravitation und Kollisionen.
    
    Methoden:
        compute_accelerations(): Berechnet Kräfte zwischen allen Körpern
        step(dt, restitution): Ein Zeitschritt (Velocity-Verlet)
        run(t_end, dt, restitution): Vollständige Simulation
    
    Algorithmus (Velocity-Verlet):
        1. a_old = compute_accelerations()
        2. pos += vel * dt + 0.5 * a_old * dt²
        3. a_new = compute_accelerations()
        4. vel += 0.5 * (a_old + a_new) * dt
    """
```

#### Sonnensystem-Daten

```python
SOLAR_SYSTEM_DATA = {
    "Sonne":   {"mass": 1.989e30, "distance": 0,        "velocity": 0,     "color": "#FFD700"},
    "Merkur":  {"mass": 3.285e23, "distance": 0.387*AU, "velocity": 47870, "color": "#A0522D"},
    "Venus":   {"mass": 4.867e24, "distance": 0.723*AU, "velocity": 35020, "color": "#DEB887"},
    "Erde":    {"mass": 5.972e24, "distance": 1.000*AU, "velocity": 29780, "color": "#4169E1"},
    "Mars":    {"mass": 6.390e23, "distance": 1.524*AU, "velocity": 24130, "color": "#CD5C5C"},
    "Jupiter": {"mass": 1.898e27, "distance": 5.203*AU, "velocity": 13070, "color": "#F4A460"},
    "Saturn":  {"mass": 5.683e26, "distance": 9.537*AU, "velocity": 9690,  "color": "#DAA520"},
}
# AU = 1.495978707e11 m (Astronomische Einheit)
```

---

### 2.2 Thermodynamik (`ui_thermo_bundle.py`)

#### Wärmeleitung

```python
def heat_conduction_1d_step(T, alpha, dx, dt):
    """
    Ein Zeitschritt der 1D-Wärmeleitungsgleichung.
    
    Gleichung: ∂T/∂t = α * ∂²T/∂x²
    
    Methode: Explizites Euler-Verfahren (FTCS)
    
    Stabilitätsbedingung (CFL): dt ≤ 0.5 * dx² / α
    
    Args:
        T: Temperaturarray [°C oder K]
        alpha: Wärmeleitfähigkeit [m²/s]
        dx: Ortsschrittweite [m]
        dt: Zeitschrittweite [s]
    
    Returns:
        T_new: Aktualisiertes Temperaturarray
    """
    T_new = T.copy()
    r = alpha * dt / dx**2
    T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
    return T_new
```

#### Zustandsänderungen

```python
def isothermal_process(p1, V1, V2):
    """Isotherme Zustandsänderung: T = const, pV = const"""
    V = np.linspace(V1, V2, 100)
    p = p1 * V1 / V
    W = p1 * V1 * np.log(V2 / V1)  # Arbeit
    return V, p, W

def adiabatic_process(p1, V1, V2, gamma=1.4):
    """Adiabatische Zustandsänderung: Q = 0, pV^γ = const"""
    V = np.linspace(V1, V2, 100)
    p = p1 * (V1 / V)**gamma
    W = (p1*V1 - p[-1]*V2) / (gamma - 1)
    return V, p, W
```

#### Kreisprozesse

```python
def carnot_cycle(T_hot, T_cold, V1, V2, V3):
    """
    Carnot-Kreisprozess (idealer Wärmekraftprozess).
    
    Schritte:
        1. Isotherme Expansion bei T_hot
        2. Adiabatische Expansion T_hot → T_cold
        3. Isotherme Kompression bei T_cold
        4. Adiabatische Kompression T_cold → T_hot
    
    Wirkungsgrad: η = 1 - T_cold/T_hot (Carnot-Wirkungsgrad)
    """

def otto_cycle(T1, p1, r, gamma=1.4, Q_in=1000):
    """
    Otto-Kreisprozess (Benzinmotor).
    
    Schritte:
        1. Isentrope Kompression (1→2)
        2. Isochore Wärmezufuhr (2→3)
        3. Isentrope Expansion (3→4)
        4. Isochore Wärmeabfuhr (4→1)
    
    Wirkungsgrad: η = 1 - 1/r^(γ-1)
    
    Args:
        r: Verdichtungsverhältnis V1/V2
        gamma: Adiabatenexponent (Luft: 1.4)
        Q_in: Zugeführte Wärme [J]
    """
```

#### Gaskinetik

```python
@dataclass
class Particle:
    """Gasteilchen für kinetische Simulation"""
    x: float      # Position x [m]
    y: float      # Position y [m]
    vx: float     # Geschwindigkeit x [m/s]
    vy: float     # Geschwindigkeit y [m/s]
    mass: float   # Masse [kg]

def init_particles(N, T, box_size, mass=1e-26):
    """
    Initialisiere N Teilchen mit Maxwell-Boltzmann-Verteilung.
    
    Mittlere Geschwindigkeit: v_rms = sqrt(3 * k_B * T / m)
    """

def compute_pressure(particles, box_size):
    """
    Berechne Druck aus kinetischer Energie.
    
    2D-Formel: p = (2/V) * E_kin = (N/V) * m * <v²>
    """
```

---

### 2.3 Atomphysik (`ui_atom_bundle.py`)

#### Physikalische Konstanten

```python
h = 6.62607015e-34      # Planck-Konstante [J·s]
c = 299792458.0         # Lichtgeschwindigkeit [m/s]
e = 1.602176634e-19     # Elementarladung [C]
k_B = 1.380649e-23      # Boltzmann-Konstante [J/K]
R_inf = 1.097373e7      # Rydberg-Konstante [1/m]
E_H = 13.605693122      # Ionisierungsenergie H [eV]
a_0 = 5.29177210903e-11 # Bohr-Radius [m]
```

#### Bohr-Modell

```python
def bohr_energy(n, Z=1):
    """
    Energie im n-ten Niveau des Bohr-Modells.
    
    E_n = -13.6 eV * Z² / n²
    
    Args:
        n: Hauptquantenzahl (1, 2, 3, ...)
        Z: Kernladungszahl (1=H, 2=He⁺, 3=Li²⁺)
    
    Returns:
        E: Energie in eV (negativ = gebunden)
    """
    return -E_H * Z**2 / n**2

def bohr_radius(n, Z=1):
    """
    Bahnradius im n-ten Niveau.
    
    r_n = a_0 * n² / Z
    """
    return a_0 * n**2 / Z

def transition_wavelength(n_high, n_low, Z=1):
    """
    Wellenlänge beim Übergang n_high → n_low.
    
    1/λ = R_inf * Z² * (1/n_low² - 1/n_high²)
    """
    delta_E = abs(bohr_energy(n_high, Z) - bohr_energy(n_low, Z)) * e  # in Joule
    wavelength = h * c / delta_E
    return wavelength * 1e9  # in nm
```

#### Spektralserien

| Serie | n_final | Bereich | Wellenlängen (H) |
|-------|---------|---------|------------------|
| Lyman | 1 | UV | 91.2 - 121.6 nm |
| Balmer | 2 | Sichtbar | 364.6 - 656.3 nm |
| Paschen | 3 | NIR | 820.4 - 1875 nm |
| Brackett | 4 | IR | 1458 - 4051 nm |
| Pfund | 5 | FIR | 2279 - 7460 nm |

#### Photoeffekt

```python
WORK_FUNCTIONS = {
    "Cs": 1.95,   # Cäsium
    "K":  2.30,   # Kalium
    "Na": 2.75,   # Natrium
    "Zn": 4.33,   # Zink
    "Cu": 4.65,   # Kupfer
    "Ag": 4.73,   # Silber
    "Pt": 5.65,   # Platin
}

def photoelectric_kinetic_energy(wavelength_nm, work_function_eV):
    """
    Kinetische Energie der Photoelektronen.
    
    Einstein: E_kin = h*f - W = h*c/λ - W
    
    Returns:
        E_kin in eV (0 wenn λ > λ_grenz)
    """
    E_photon = h * c / (wavelength_nm * 1e-9) / e  # in eV
    E_kin = max(0, E_photon - work_function_eV)
    return E_kin

def threshold_wavelength(work_function_eV):
    """Grenzwellenlänge: λ_grenz = h*c/W"""
    return h * c / (work_function_eV * e) * 1e9  # in nm
```

#### Franck-Hertz

```python
FRANCK_HERTZ_DATA = {
    "Hg": {"excitation_eV": 4.9, "wavelength_nm": 253.7},  # Quecksilber
    "Ne": {"excitation_eV": 18.7, "wavelength_nm": 66.2},  # Neon
}

def franck_hertz_current(U, U_excitation, I_max=1.0):
    """
    Simulierte Strom-Spannungs-Kennlinie.
    
    Maxima bei U = n * U_excitation (n = 1, 2, 3, ...)
    """
```

---

### 2.4 Ultraschall (`ultrasound_sim.py`)

#### Datenklassen

```python
@dataclass
class Medium:
    c: float = 1540.0              # Schallgeschwindigkeit [m/s]
    rho: float = 1000.0            # Dichte [kg/m³]
    alpha_db_cm_mhz: float = 0.5   # Dämpfung [dB/(MHz·cm)]

@dataclass
class Transducer:
    kind: str = "linear"           # linear | phased
    f0: float = 5e6                # Zentralfrequenz [Hz]
    pitch: float = 0.0003          # Elementabstand [m]
    N: int = 64                    # Anzahl Elemente
    frac_bw: float = 0.6           # Fraktionale Bandbreite
    tx_cycles: float = 2.0         # Sendezyklen

@dataclass
class Beamformer:
    method: str = "das"            # das | capon
    apod: str = "hanning"          # none | hanning | hamming
    tx_focus_z: float = 0.03       # Sendefokus [m]

@dataclass
class Scatterer:
    x: float                       # x-Position [m]
    z: float                       # z-Position (Tiefe) [m]
    amp: float = 1.0               # Reflektivität
    vx: float = 0.0                # Laterale Geschw. [m/s]
    vz: float = 0.0                # Axiale Geschw. [m/s]
```

#### Signalverarbeitung

```python
def gaussian_pulse(f0, fs, cycles=2.0, frac_bw=0.6):
    """
    Bandlimitierter Sendepuls mit Gauss-Hüllkurve.
    
    Args:
        f0: Zentralfrequenz [Hz]
        fs: Abtastrate [Hz]
        cycles: Anzahl Zyklen bei -6 dB
        frac_bw: Fraktionale Bandbreite
    
    Returns:
        s: Puls-Array (normalisiert)
    """

def analytic_signal(x, axis=-1):
    """
    Analytisches Signal via Hilbert-Transformation.
    
    Implementierung: FFT-basiert (ohne SciPy)
    
    z(t) = x(t) + j * H{x(t)}
    
    Hüllkurve: |z(t)|
    """

def envelope_db(x, floor_db=-60.0):
    """
    Logarithmische Hüllkurve in dB.
    
    dB = 20 * log10(|z|/max|z|)
    """
```

#### Beamforming

```python
def das_beamform(rf, t, fs, x_elems, medium, Ximg, Zimg, apod=None):
    """
    Delay-and-Sum Beamforming für B-Mode.
    
    Algorithmus:
        1. Für jeden Bildpunkt (x, z):
        2.   Berechne Laufzeit τ = (r_tx + r_rx) / c
        3.   Interpoliere RF-Signal bei τ
        4.   Summiere über alle Kanäle (mit Apodisation)
        5.   Berechne Hüllkurve
    
    Args:
        rf: RF-Daten (ensembles, Nch, Nt)
        t: Zeitachse [s]
        x_elems: Element-Positionen [m]
        medium: Medium-Objekt
        Ximg, Zimg: Bildkoordinaten [m]
        apod: Apodisationsgewichte
    
    Returns:
        img: B-Mode Bild (nz, nx)
    """
```

#### Doppler

```python
def doppler_autocorr(rf, lag=1):
    """
    Kasai-Autocorrelation für Doppler-Schätzung.
    
    R(lag) = Σ y[n] * conj(y[n+lag])
    
    Doppler-Frequenz: f_d = angle(R) / (2π * PRI)
    Geschwindigkeit:  v = f_d * c / (2 * f0)
    """

def doppler_color_map(rf, t, fs, f0, c, ensembles, pri):
    """
    Farbdoppler-Karte (Geschwindigkeit + Leistung).
    
    Returns:
        v: Geschwindigkeitskarte [m/s]
        pwr: Leistungskarte [dB]
    """
```

#### Phantome

```python
def carotid_phantom(width=0.006, depth=0.025, flow_vel=0.3, seed=0):
    """
    Carotis-Phantom mit laminarer Strömung.
    
    Enthält:
        - Zufälliges Gewebe-Speckle
        - Zylindrisches Gefäß
        - Parabolisches Strömungsprofil: v(r) = v_max * (1 - (r/R)²)
    
    Args:
        width: Gefäßdurchmesser [m]
        depth: Gefäßtiefe [m]
        flow_vel: Maximale Strömungsgeschwindigkeit [m/s]
    
    Returns:
        List[Scatterer]: Punktstreuer mit Geschwindigkeiten
    """
```

---

## 3. Physikalische Grundlagen

### 3.1 Numerische Integration

#### Euler-Verfahren (1. Ordnung)
```
y_{n+1} = y_n + h * f(t_n, y_n)
```
- Einfach, aber nur 1. Ordnung Genauigkeit
- Verwendet für: Wärmeleitung, einfache Schwingungen

#### Velocity-Verlet (2. Ordnung, symplektisch)
```
x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt²
a_{n+1} = F(x_{n+1}) / m
v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
```
- Energieerhaltend für konservative Systeme
- Verwendet für: N-Körper, Planetenbahnen

#### Leapfrog (2. Ordnung, symplektisch)
```
v_{n+1/2} = v_{n-1/2} + a_n * dt
x_{n+1} = x_n + v_{n+1/2} * dt
```
- Zeitumkehr-symmetrisch
- Verwendet für: Gekoppelte Oszillatoren

### 3.2 Stabilitätsbedingungen

#### Wärmeleitung (explizit)
```
CFL-Bedingung: dt ≤ dx² / (2α)   (1D)
               dt ≤ dx² / (4α)   (2D)
```

#### Wellengleichung
```
CFL-Bedingung: dt ≤ dx / c
```

### 3.3 Einheitensystem

| Größe | SI-Einheit | Typischer Bereich |
|-------|------------|-------------------|
| Länge | m | 1e-12 ... 1e12 |
| Zeit | s | 1e-9 ... 1e9 |
| Masse | kg | 1e-30 ... 1e30 |
| Temperatur | K | 0 ... 10000 |
| Energie | J | 1e-30 ... 1e30 |
| Frequenz | Hz | 1 ... 1e15 |

---

## 4. API-Referenz

### 4.1 Hauptfunktionen

```python
# Mechanik
render_mechanics_tab()           # Haupttab Mechanik
render_2d_mechanics_tab()        # 2D-Simulationen
render_3d_nbody_tab()            # N-Körper
render_celestial_tab()           # Himmelsmechanik
render_collisions_tab()          # Stöße

# Thermodynamik
render_thermo_tab()              # Haupttab
render_heat_conduction_tab()     # Wärmeleitung
render_state_changes_tab()       # Zustandsänderungen
render_cycles_tab()              # Kreisprozesse
render_kinetic_tab()             # Gaskinetik

# Atomphysik
render_atom_tab()                # Haupttab
render_bohr_tab()                # Bohr-Modell
render_photoeffect_tab()         # Photoeffekt
render_franck_hertz_tab()        # Franck-Hertz
render_spectra_tab()             # Spektroskopie

# Ultraschall
render_ultrasound_tab()          # Haupttab

# Medizintechnik
render_mri_bloch_tab()           # MRI
render_xray_ct_tab()             # CT
```

### 4.2 Internationalisierung

```python
from i18n_bundle import get_text, get_language_name

# Übersetzung abrufen
label = get_text("start_animation", lang)  # "▶️ Animation starten" (de)

# Neue Übersetzung hinzufügen
TRANSLATIONS["my_key"] = {
    "de": "Mein Text",
    "en": "My Text"
}
```

---

## 5. Erweiterung des Systems

### 5.1 Neues Modul hinzufügen

1. **Physik-Datei erstellen** (`my_physics.py`):
```python
def my_simulation(param1, param2):
    """Dokumentation..."""
    # Berechnung
    return results
```

2. **UI-Datei erstellen** (`ui_my_module.py`):
```python
import streamlit as st
from my_physics import my_simulation

def render_my_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("Mein Modul", "My Module"))
    
    # Parameter
    param1 = st.slider("Parameter 1", 0.0, 1.0, 0.5)
    
    if st.button(tr("Starten", "Start")):
        results = my_simulation(param1, param2)
        # Visualisierung...
```

3. **In `physics_sim.py` einbinden**:
```python
from ui_my_module import render_my_tab

# In tabs_labels hinzufügen
tabs_labels = [..., "Mein Modul"]

# Tab rendern
with selected_tabs[N]:
    render_my_tab()
```

4. **Übersetzungen in `i18n_bundle.py`**:
```python
TRANSLATIONS["my_module"] = {"de": "Mein Modul", "en": "My Module"}
```

### 5.2 Animation hinzufügen

```python
def run_my_animation(data):
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    # Placeholder für Live-Updates
    chart_placeholder = st.empty()
    progress = st.progress(0)
    
    n_frames = 100
    for frame in range(n_frames):
        # Plotly-Figure erstellen
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

## 6. Fehlerbehebung

### 6.1 Häufige Probleme

| Problem | Ursache | Lösung |
|---------|---------|--------|
| `ModuleNotFoundError` | Fehlende Abhängigkeit | `pip install -r requirements.txt` |
| Langsame Animation | Zu viele Frames | `n_frames` reduzieren |
| Numerische Instabilität | dt zu groß | CFL-Bedingung prüfen |
| Speicherfehler | Zu große Arrays | Auflösung reduzieren |

### 6.2 Performance-Optimierung

1. **NumPy-Vektorisierung nutzen**:
```python
# Schlecht
for i in range(N):
    result[i] = a[i] * b[i]

# Gut
result = a * b
```

2. **Plotly-Updates minimieren**:
```python
# Nicht jedes Frame updaten
if frame % 5 == 0:
    chart_placeholder.plotly_chart(fig)
```

3. **Session State für Caching**:
```python
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = run_expensive_simulation()
```

### 6.3 Debug-Modus

```python
# In physics_sim.py
DEBUG = True

if DEBUG:
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Language: {st.session_state.language}")
    st.sidebar.write(f"Session keys: {list(st.session_state.keys())}")
```

---

## Anhang

### A. Konstanten-Referenz

```python
# Naturkonstanten (CODATA 2018)
c = 299792458           # Lichtgeschwindigkeit [m/s]
h = 6.62607015e-34      # Planck-Konstante [J·s]
hbar = 1.054571817e-34  # Reduzierte Planck-Konstante [J·s]
e = 1.602176634e-19     # Elementarladung [C]
m_e = 9.1093837015e-31  # Elektronenmasse [kg]
m_p = 1.67262192369e-27 # Protonenmasse [kg]
k_B = 1.380649e-23      # Boltzmann-Konstante [J/K]
N_A = 6.02214076e23     # Avogadro-Konstante [1/mol]
R = 8.314462618         # Gaskonstante [J/(mol·K)]
G = 6.67430e-11         # Gravitationskonstante [m³/(kg·s²)]
epsilon_0 = 8.8541878e-12  # Elektrische Feldkonstante [F/m]
mu_0 = 1.25663706e-6    # Magnetische Feldkonstante [H/m]

# Astronomische Konstanten
AU = 1.495978707e11     # Astronomische Einheit [m]
pc = 3.0856776e16       # Parsec [m]
M_sun = 1.98847e30      # Sonnenmasse [kg]
R_sun = 6.9634e8        # Sonnenradius [m]
M_earth = 5.9722e24     # Erdmasse [kg]
R_earth = 6.371e6       # Erdradius [m]
```

### B. Literatur

1. Landau, L.D. & Lifshitz, E.M.: *Mechanik* (Lehrbuch der Theoretischen Physik, Band 1)
2. Demtröder, W.: *Experimentalphysik 1-4* (Springer)
3. Jensen, J.A.: *Medical Ultrasound Imaging* (Wiley)
4. Haacke, E.M.: *Magnetic Resonance Imaging* (Wiley)

---

*Dokumentation Version 5.0 | Stand: November 2024*
