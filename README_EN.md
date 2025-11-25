# 🔬 Physics Teaching Simulator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-5.0-purple.svg)](https://github.com)

**Interactive Physics Simulations for University Education**

A comprehensive, bilingual (German/English) simulation tool for physics and medical technology, developed with Python and Streamlit. Ideal for lectures, exercises, and self-study.

![Physics Simulator Banner](https://via.placeholder.com/800x200/667eea/ffffff?text=Physics+Teaching+Simulator)

---

## 📋 Table of Contents

- [Features](#-features)
- [Modules](#-modules)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Screenshots](#-screenshots)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## ✨ Features

### General
- 🌍 **Bilingual** — Full support for German and English
- 🎬 **Real-time Animations** — Interactive visualizations with Plotly
- 📊 **Physically Accurate** — Based on established numerical methods
- 📱 **Responsive Design** — Works on desktop and tablet
- 🎓 **Education-Oriented** — Formulas, explanations, and presets for typical teaching scenarios

### Didactic
- Pre-configured scenarios for common physics experiments
- Real-time parameter variation
- Energy conservation and momentum diagrams
- Export of simulation data

---

## 📚 Modules

### 1. 🚀 Mechanics & Celestial Mechanics

| Simulation | Description | Animation |
|------------|-------------|-----------|
| **Projectile Motion** | With/without air resistance | ✅ |
| **Simple Pendulum** | Non-linear oscillation, phase space | ✅ |
| **Coupled Pendulums** | Energy exchange, beating | — |
| **Spring Oscillator** | Damped harmonic oscillator | — |
| **Inclined Plane** | With friction, energy balance | — |
| **3D N-Body** | Gravitation, collisions, Velocity-Verlet | ✅ |
| **Solar System** | Real planetary data, Kepler orbits | — |
| **Lagrange Points** | L1-L5 calculation and visualization | — |
| **1D/2D Collisions** | Elastic/inelastic, restitution coefficient | ✅ |
| **Billiards** | Multiple balls, wall reflection | ✅ |
| **Newton's Cradle** | Momentum transfer | ✅ |

**Physics Highlights:**
- Velocity-Verlet integration for energy conservation
- Figure-8 solution of the three-body problem
- Maxwell-Boltzmann velocity distribution

---

### 2. 🌡️ Thermodynamics

| Simulation | Description | Animation |
|------------|-------------|-----------|
| **1D Heat Conduction** | Explicit Euler method | ✅ |
| **2D Heat Conduction** | Heatmap visualization | ✅ |
| **State Changes** | Isothermal, isobaric, isochoric, adiabatic | — |
| **Carnot Cycle** | Ideal thermodynamic cycle, efficiency | — |
| **Otto Cycle** | Gasoline engine simulation | — |
| **Gas Kinetics** | Particle simulation in 2D box | ✅ |

**Equations:**
```
Heat conduction:  ∂T/∂t = α ∇²T
Carnot:           η = 1 - T_cold/T_hot
Otto:             η = 1 - 1/r^(γ-1)
Ideal gas:        pV = nRT
```

---

### 3. ⚛️ Atomic Physics

| Simulation | Description | Animation |
|------------|-------------|-----------|
| **Bohr Model** | H, He⁺, Li²⁺ (Z=1-3), n=1-7 | ✅ |
| **Photoelectric Effect** | 7 materials, E_kin vs. λ | ✅ |
| **Franck-Hertz** | Hg (4.9 eV), Ne (18.7 eV) | ✅ |
| **Spectroscopy** | Emission/absorption spectra | — |

**Spectral Series:**
- Lyman (UV): n → 1
- Balmer (visible): n → 2  
- Paschen (NIR): n → 3

**Materials (Work Function):**
| Material | W [eV] |
|----------|--------|
| Cesium | 1.95 |
| Potassium | 2.30 |
| Sodium | 2.75 |
| Zinc | 4.33 |
| Copper | 4.65 |
| Silver | 4.73 |
| Platinum | 5.65 |

---

### 4. 🔬 Optics

| Simulation | Description |
|------------|-------------|
| **Geometric Optics** | Lenses, mirrors, refraction |
| **Ray-Tracing** | Ray path through optical systems |
| **Wave Optics** | Interference, diffraction |

---

### 5. 🩻 X-Ray & CT

| Simulation | Description |
|------------|-------------|
| **X-Ray Spectrum** | Bremsstrahlung, characteristic lines |
| **CT Reconstruction** | Radon transform, back-projection |
| **Hounsfield Scale** | Tissue contrasts |

---

### 6. 🧲 MRI & Bloch Equations

| Simulation | Description |
|------------|-------------|
| **Bloch Equations** | Magnetization dynamics M(t) |
| **T1/T2 Relaxation** | Spin-lattice, spin-spin |
| **FID Signal** | Free Induction Decay |
| **Sequences** | Spin-echo, gradient-echo |

**Bloch Equations:**
```
dMx/dt = γ(M × B)_x - Mx/T2
dMy/dt = γ(M × B)_y - My/T2  
dMz/dt = γ(M × B)_z - (Mz - M0)/T1
```

---

### 7. 🔊 Ultrasound (NEW!)

| Simulation | Description |
|------------|-------------|
| **B-Mode Imaging** | Delay-and-Sum Beamforming |
| **Point Scatter PSF** | Point Spread Function |
| **Carotid Phantom** | Vessel with laminar flow |
| **Color Doppler** | Kasai autocorrelation |

**Features:**
- Linear array with 16-128 elements
- Apodization (Hanning, Hamming)
- Frequency-dependent attenuation
- RF data export (NPZ)

**Parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| Frequency f₀ | 1-20 MHz | 7 MHz |
| Elements N | 16-128 | 64 |
| Pitch | 0.1-1.0 mm | 0.3 mm |
| Attenuation α | 0-2 dB/(MHz·cm) | 0.5 |
| Sampling rate fs | 10-100 MHz | 40 MHz |

---

### 8. ⚡ Electrodynamics

| Simulation | Description |
|------------|-------------|
| **E-Field** | Point charges, field lines |
| **B-Field** | Currents, coils |
| **EM Waves** | Propagation, polarization |

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python Package Manager)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/physics-simulator.git
cd physics-simulator
```

### Step 2: Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Start Application
```bash
streamlit run physics_sim.py
```

The application opens automatically at `http://localhost:8501`

---

## ⚡ Quick Start

```bash
# One-liner for quick start
pip install streamlit numpy plotly && streamlit run physics_sim.py
```

### Docker (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "physics_sim.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t physics-sim .
docker run -p 8501:8501 physics-sim
```

---

## 📸 Screenshots

### Mechanics — Newton's Cradle
![Newton's Cradle](https://via.placeholder.com/600x300/4169E1/ffffff?text=Newton%27s+Cradle+Animation)

### Thermodynamics — Gas Kinetics
![Gas Kinetics](https://via.placeholder.com/600x300/FF6B6B/ffffff?text=Gas+Kinetics+Simulation)

### Atomic Physics — Bohr Model
![Bohr Model](https://via.placeholder.com/600x300/9B59B6/ffffff?text=Bohr+Model+Animation)

### Ultrasound — B-Mode
![Ultrasound B-Mode](https://via.placeholder.com/600x300/1ABC9C/ffffff?text=Ultrasound+B-Mode+Image)

---

## 🔧 Technical Details

### Numerical Methods

| Method | Application |
|--------|-------------|
| Velocity-Verlet | N-body, planetary orbits |
| Explicit Euler | Heat conduction, pendulum |
| Leapfrog | Coupled oscillators |
| RK4 | Bloch equations |
| DAS Beamforming | Ultrasound imaging |

### Libraries

| Package | Version | Usage |
|---------|---------|-------|
| `streamlit` | ≥1.28.0 | Web interface |
| `numpy` | ≥1.24.0 | Numerics |
| `plotly` | ≥5.18.0 | Visualization |
| `matplotlib` | ≥3.7.0 | Additional plots |

### Performance

- Typical frame time: 20-50 ms
- Recommended browser: Chrome, Firefox
- RAM usage: ~200-500 MB

---

## 📁 Project Structure

```
physics-simulator/
│
├── physics_sim.py          # Main application (entry point)
├── requirements.txt        # Python dependencies
├── README.md              # German documentation
├── README_EN.md           # English documentation (this file)
├── DOCUMENTATION.md       # Technical documentation (German)
├── DOCUMENTATION_EN.md    # Technical documentation (English)
├── CHANGELOG.md           # Version history
├── LICENSE                # MIT License
│
├── i18n_bundle.py         # Internationalization (DE/EN)
├── sim_core_bundle.py     # Core physics functions
│
├── ui_mech_bundle.py      # Mechanics UI
├── ui_thermo_bundle.py    # Thermodynamics UI
├── ui_atom_bundle.py      # Atomic physics UI
├── ui_optics_bundle.py    # Optics UI
├── ui_med_bundle.py       # MRI/Bloch UI
├── ui_ultrasound.py       # Ultrasound UI
│
├── ultrasound_sim.py      # Ultrasound physics
└── xray_ct.py             # CT physics
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create feature branch** (`git checkout -b feature/NewSimulation`)
3. **Commit changes** (`git commit -m 'Add: New Simulation'`)
4. **Push branch** (`git push origin feature/NewSimulation`)
5. **Open Pull Request**

### Coding Guidelines
- PEP 8 for Python code
- Docstrings for all functions
- Bilingual UI texts in `i18n_bundle.py`
- Tests for physical calculations

### Ideas for Extensions
- [ ] PET/SPECT simulation
- [ ] Quantum mechanics (wave functions)
- [ ] Acoustics (room acoustics, resonance)
- [ ] Quiz mode with evaluation
- [ ] CSV/JSON export

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2024 Prof. Dr. Dietmar Henrich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👤 Author

**Prof. Dr. Dietmar Henrich**  
Professor of Medical Technology  
Focus: Physics, Medical Imaging, Educational Software

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the excellent framework
- [Plotly](https://plotly.com) for interactive visualizations
- [NumPy](https://numpy.org) for numerical computations

---

<p align="center">
  <b>⭐ If you like this project, give it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ for Physics Education
</p>
