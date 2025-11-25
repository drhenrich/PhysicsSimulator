# 🔬 Physics Simulator for Education

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-6.0-purple.svg)]()

A comprehensive, interactive simulation platform for physics and medical technology education. Built with Python and Streamlit, the simulator offers **10 specialized modules** with over **50 interactive visualizations**.

## ✨ Key Features

- 🎓 **Educationally designed** — Optimized for lectures and labs
- 🌍 **Bilingual** — Full DE/EN support
- 🎬 **Smooth animations** — Client-side Plotly frame animations
- 📱 **Responsive** — Works on desktop and tablet
- 🔧 **Modular** — Easy to extend

## 📚 Modules

| Module | Description | Simulations |
|--------|-------------|-------------|
| ⚙️ **Mechanics** | Classical mechanics & celestial mechanics | Projectile, pendulum, N-body, collisions, billiard |
| 🌡️ **Thermodynamics** | Heat transfer & gas theory | Heat conduction 1D/2D, thermodynamic cycles, gas kinetics |
| ⚛️ **Atomic Physics** | Quantum phenomena | Bohr model, photoelectric effect, Franck-Hertz, spectra |
| 🎵 **Oscillations** | Oscillations & acoustics | Oscillators, beats, standing waves, Doppler effect |
| 🔭 **Optics** | Geometric optics | Lenses, mirrors, ray tracing |
| ☢️ **Nuclear Physics** | Radioactivity & radiation protection | Decay, decay chains, dosimetry, shielding |
| 🩻 **X-ray/CT** | Medical imaging | Absorption, CT reconstruction, Hounsfield units |
| 🧲 **MRI & Bloch** | Nuclear magnetic resonance | Bloch equations, T1/T2 relaxation, spin echo |
| 🔊 **Ultrasound** | Sonography | Wave propagation, beamforming, B-mode |
| ⚡ **Electrodynamics** | Electrostatics | Field lines, potentials, Poisson equation |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YourUsername/physics-simulator.git
cd physics-simulator

# Install dependencies
pip install -r requirements.txt

# Start simulator
streamlit run physics_sim.py
```

### Requirements

- Python 3.9+
- Streamlit ≥ 1.28.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- Plotly ≥ 5.18.0

## 🎬 Animations

All animations use **Plotly frame technology** for smooth, client-side playback:

```
▶️ Play   — Start animation
⏸️ Pause  — Pause animation
🔄 Reset  — Return to beginning
```

**Animated Simulations:**
- Projectile motion & pendulum
- Elastic/inelastic collisions
- Billiard & Newton's cradle
- Heat conduction (1D/2D)
- Gas kinetics (Maxwell-Boltzmann)
- Electron transitions (Bohr)
- Photoemission
- Standing waves
- Doppler effect

## 📖 Documentation

Detailed documentation: [DOCUMENTATION_EN.md](DOCUMENTATION_EN.md)

### Project Structure

```
physics-simulator/
├── physics_sim.py          # Main application
├── i18n_bundle.py          # Translations
├── sim_core_bundle.py      # Core functions
├── ui_mech_bundle.py       # Mechanics module
├── ui_thermo_bundle.py     # Thermodynamics module
├── ui_atom_bundle.py       # Atomic physics module
├── ui_oscillations_bundle.py # Oscillations module
├── ui_optics_bundle.py     # Optics module
├── ui_nuclear_bundle.py    # Nuclear physics module
├── ui_med_bundle.py        # Medical physics module
├── ui_ultrasound.py        # Ultrasound UI
├── ultrasound_sim.py       # Ultrasound simulation
├── xray_ct.py              # CT reconstruction
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
├── CHANGELOG.md
└── LICENSE
```

## 🔬 Physical Foundations

The simulator implements the following physical models:

### Mechanics
- Newton's equations of motion
- Runge-Kutta 4 integration
- Conservation of momentum and energy
- Law of gravitation (N-body)

### Thermodynamics
- Fourier heat equation
- Ideal gas law
- Carnot and Otto cycles
- Maxwell-Boltzmann distribution

### Atomic Physics
- Bohr atomic model
- Einstein's photoelectric equation
- Franck-Hertz experiment
- Emission/absorption spectra

### Nuclear Physics
- Decay law: A(t) = A₀·e^(-λt)
- Bateman equations (decay chains)
- Inverse square law: Ḋ = A·Γ/r²
- Shielding: I = I₀·e^(-μx)

### Oscillations
- Damped harmonic oscillator
- Coupled oscillators
- Doppler effect: f' = f·(c±v_o)/(c∓v_s)

## 🎯 Use Cases

- **Lectures** — Live demonstrations of physical phenomena
- **Labs** — Virtual experiments and data analysis
- **Self-study** — Interactive learning with parameter variation
- **Exam preparation** — Visualization of complex relationships

## 🤝 Contributing

Contributions are welcome!

1. Create a fork
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push branch (`git push origin feature/NewFeature`)
5. Create pull request

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Prof. Dr. Dietmar Henrich**  
Medical Technology & Physics

---

<p align="center">
  <i>Developed for education. Inspired by physics.</i>
</p>
