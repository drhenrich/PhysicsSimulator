# 🔬 Physics Teaching Simulator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-5.0-purple.svg)](https://github.com)

**Interaktive Physiksimulationen für die Hochschulausbildung** | **Interactive Physics Simulations for University Education**

Ein umfassendes, zweisprachiges (DE/EN) Simulationstool für Physik und Medizintechnik, entwickelt mit Python und Streamlit. Ideal für Vorlesungen, Übungen und Selbststudium.

![Physics Simulator Banner](https://via.placeholder.com/800x200/667eea/ffffff?text=Physics+Teaching+Simulator)

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Module](#-module)
- [Installation](#-installation)
- [Schnellstart](#-schnellstart)
- [Screenshots](#-screenshots)
- [Technische Details](#-technische-details)
- [Projektstruktur](#-projektstruktur)
- [Mitwirken](#-mitwirken)
- [Lizenz](#-lizenz)
- [Autor](#-autor)

---

## ✨ Features

### Allgemein
- 🌍 **Zweisprachig** — Vollständige Unterstützung für Deutsch und Englisch
- 🎬 **Echtzeit-Animationen** — Interaktive Visualisierungen mit Plotly
- 📊 **Physikalisch korrekt** — Basierend auf etablierten numerischen Methoden
- 📱 **Responsive Design** — Funktioniert auf Desktop und Tablet
- 🎓 **Bildungsorientiert** — Formeln, Erklärungen und Presets für typische Lehrszenarien

### Didaktisch
- Vorbereitete Szenarien für gängige Physik-Experimente
- Parametervariation in Echtzeit
- Energieerhaltungs- und Impulsdiagramme
- Export von Simulationsdaten

---

## 📚 Module

### 1. 🚀 Mechanik & Himmelsmechanik

| Simulation | Beschreibung | Animation |
|------------|--------------|-----------|
| **Schiefer Wurf** | Mit/ohne Luftwiderstand | ✅ |
| **Einfaches Pendel** | Nicht-lineare Schwingung, Phasenraum | ✅ |
| **Gekoppelte Pendel** | Energieaustausch, Schwebung | — |
| **Federschwingung** | Gedämpfter harmonischer Oszillator | — |
| **Schiefe Ebene** | Mit Reibung, Energiebilanz | — |
| **3D N-Körper** | Gravitation, Kollisionen, Velocity-Verlet | ✅ |
| **Sonnensystem** | Echte Planetendaten, Kepler-Bahnen | — |
| **Lagrange-Punkte** | L1-L5 Berechnung und Visualisierung | — |
| **1D/2D Stöße** | Elastisch/inelastisch, Stoßzahl | ✅ |
| **Billard** | Mehrere Kugeln, Wandreflexion | ✅ |
| **Newton-Wiege** | Impulsübertragung | ✅ |

**Physik-Highlights:**
- Velocity-Verlet Integration für Energieerhaltung
- Figure-8 Lösung des Dreikörperproblems
- Maxwell-Boltzmann Geschwindigkeitsverteilung

---

### 2. 🌡️ Thermodynamik

| Simulation | Beschreibung | Animation |
|------------|--------------|-----------|
| **1D Wärmeleitung** | Explizites Euler-Verfahren | ✅ |
| **2D Wärmeleitung** | Heatmap-Visualisierung | ✅ |
| **Zustandsänderungen** | Isotherm, isobar, isochor, adiabatisch | — |
| **Carnot-Prozess** | Idealer Kreisprozess, Wirkungsgrad | — |
| **Otto-Prozess** | Benzinmotor-Simulation | — |
| **Gaskinetik** | Teilchensimulation in 2D-Box | ✅ |

**Formeln:**
```
Wärmeleitung:    ∂T/∂t = α ∇²T
Carnot:          η = 1 - T_kalt/T_heiß
Otto:            η = 1 - 1/r^(γ-1)
Ideales Gas:     pV = nRT
```

---

### 3. ⚛️ Atomphysik

| Simulation | Beschreibung | Animation |
|------------|--------------|-----------|
| **Bohr-Modell** | H, He⁺, Li²⁺ (Z=1-3), n=1-7 | ✅ |
| **Photoeffekt** | 7 Materialien, E_kin vs. λ | ✅ |
| **Franck-Hertz** | Hg (4.9 eV), Ne (18.7 eV) | ✅ |
| **Spektroskopie** | Emissions-/Absorptionsspektren | — |

**Spektralserien:**
- Lyman (UV): n → 1
- Balmer (sichtbar): n → 2  
- Paschen (IR): n → 3

**Materialien (Austrittsarbeit):**
| Material | W [eV] |
|----------|--------|
| Cäsium | 1.95 |
| Kalium | 2.30 |
| Natrium | 2.75 |
| Zink | 4.33 |
| Kupfer | 4.65 |
| Silber | 4.73 |
| Platin | 5.65 |

---

### 4. 🔬 Optik

| Simulation | Beschreibung |
|------------|--------------|
| **Geometrische Optik** | Linsen, Spiegel, Brechung |
| **Ray-Tracing** | Strahlengang durch optische Systeme |
| **Wellenoptik** | Interferenz, Beugung |

---

### 5. 🩻 Röntgen & CT

| Simulation | Beschreibung |
|------------|--------------|
| **Röntgenspektrum** | Bremsstrahlung, charakteristische Linien |
| **CT-Rekonstruktion** | Radon-Transformation, Rückprojektion |
| **Hounsfield-Skala** | Gewebekontraste |

---

### 6. 🧲 MRI & Bloch-Gleichungen

| Simulation | Beschreibung |
|------------|--------------|
| **Bloch-Gleichungen** | Magnetisierungsdynamik M(t) |
| **T1/T2-Relaxation** | Spin-Gitter, Spin-Spin |
| **FID-Signal** | Free Induction Decay |
| **Sequenzen** | Spin-Echo, Gradienten-Echo |

**Bloch-Gleichungen:**
```
dMx/dt = γ(M × B)_x - Mx/T2
dMy/dt = γ(M × B)_y - My/T2  
dMz/dt = γ(M × B)_z - (Mz - M0)/T1
```

---

### 7. 🔊 Ultraschall (NEU!)

| Simulation | Beschreibung |
|------------|--------------|
| **B-Mode Bildgebung** | Delay-and-Sum Beamforming |
| **Punktstreuer-PSF** | Point Spread Function |
| **Carotis-Phantom** | Gefäß mit laminarer Strömung |
| **Farbdoppler** | Kasai-Autocorrelation |

**Features:**
- Lineararray mit 16-128 Elementen
- Apodisation (Hanning, Hamming)
- Frequenzabhängige Dämpfung
- RF-Daten Export (NPZ)

**Parameter:**
| Parameter | Bereich | Default |
|-----------|---------|---------|
| Frequenz f₀ | 1-20 MHz | 7 MHz |
| Elemente N | 16-128 | 64 |
| Pitch | 0.1-1.0 mm | 0.3 mm |
| Dämpfung α | 0-2 dB/(MHz·cm) | 0.5 |
| Abtastrate fs | 10-100 MHz | 40 MHz |

---

### 8. ⚡ Elektrodynamik

| Simulation | Beschreibung |
|------------|--------------|
| **E-Feld** | Punktladungen, Feldlinien |
| **B-Feld** | Ströme, Spulen |
| **EM-Wellen** | Ausbreitung, Polarisation |

---

## 🚀 Installation

### Voraussetzungen
- Python 3.9 oder höher
- pip (Python Package Manager)

### Schritt 1: Repository klonen
```bash
git clone https://github.com/yourusername/physics-simulator.git
cd physics-simulator
```

### Schritt 2: Virtuelle Umgebung erstellen (empfohlen)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

### Schritt 3: Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### Schritt 4: Anwendung starten
```bash
streamlit run physics_sim.py
```

Die Anwendung öffnet sich automatisch unter `http://localhost:8501`

---

## ⚡ Schnellstart

```bash
# Einzeiler für schnellen Start
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

### Mechanik — Newton-Wiege
![Newton's Cradle](https://via.placeholder.com/600x300/4169E1/ffffff?text=Newton%27s+Cradle+Animation)

### Thermodynamik — Gaskinetik
![Gas Kinetics](https://via.placeholder.com/600x300/FF6B6B/ffffff?text=Gas+Kinetics+Simulation)

### Atomphysik — Bohr-Modell
![Bohr Model](https://via.placeholder.com/600x300/9B59B6/ffffff?text=Bohr+Model+Animation)

### Ultraschall — B-Mode
![Ultrasound B-Mode](https://via.placeholder.com/600x300/1ABC9C/ffffff?text=Ultrasound+B-Mode+Image)

---

## 🔧 Technische Details

### Numerische Methoden

| Methode | Anwendung |
|---------|-----------|
| Velocity-Verlet | N-Körper, Planetenbahnen |
| Euler explizit | Wärmeleitung, Pendel |
| Leapfrog | Gekoppelte Oszillatoren |
| RK4 | Bloch-Gleichungen |
| DAS Beamforming | Ultraschall-Bildgebung |

### Bibliotheken

| Paket | Version | Verwendung |
|-------|---------|------------|
| `streamlit` | ≥1.28.0 | Web-Interface |
| `numpy` | ≥1.24.0 | Numerik |
| `plotly` | ≥5.18.0 | Visualisierung |
| `matplotlib` | ≥3.7.0 | Zusätzliche Plots |

### Performance

- Typische Framezeit: 20-50 ms
- Empfohlener Browser: Chrome, Firefox
- RAM-Verbrauch: ~200-500 MB

---

## 📁 Projektstruktur

```
physics-simulator/
│
├── physics_sim.py          # Hauptanwendung (Entry Point)
├── requirements.txt        # Python-Abhängigkeiten
├── README.md              # Diese Datei
├── DOCUMENTATION.md       # Technische Dokumentation
├── LICENSE                # MIT-Lizenz
│
├── i18n_bundle.py         # Internationalisierung (DE/EN)
├── sim_core_bundle.py     # Physik-Kernfunktionen
│
├── ui_mech_bundle.py      # Mechanik-UI
├── ui_thermo_bundle.py    # Thermodynamik-UI
├── ui_atom_bundle.py      # Atomphysik-UI
├── ui_optics_bundle.py    # Optik-UI
├── ui_med_bundle.py       # MRI/Bloch-UI
├── ui_ultrasound.py       # Ultraschall-UI
│
├── ultrasound_sim.py      # Ultraschall-Physik
└── xray_ct.py             # CT-Physik
```

---

## 🤝 Mitwirken

Beiträge sind willkommen! So können Sie helfen:

1. **Fork** des Repositories erstellen
2. **Feature-Branch** anlegen (`git checkout -b feature/NeueSimulation`)
3. **Änderungen committen** (`git commit -m 'Add: Neue Simulation'`)
4. **Branch pushen** (`git push origin feature/NeueSimulation`)
5. **Pull Request** öffnen

### Coding Guidelines
- PEP 8 für Python-Code
- Docstrings für alle Funktionen
- Zweisprachige UI-Texte in `i18n_bundle.py`
- Tests für physikalische Berechnungen

### Ideen für Erweiterungen
- [ ] PET/SPECT Simulation
- [ ] Quantenmechanik (Wellenfunktionen)
- [ ] Akustik (Raumakustik, Resonanz)
- [ ] Quiz-Modus mit Auswertung
- [ ] CSV/JSON Export

---

## 📄 Lizenz

Dieses Projekt ist unter der **MIT-Lizenz** lizenziert. Siehe [LICENSE](LICENSE) für Details.

```
MIT License

Copyright (c) 2024 Prof. Dr. Dietmar Henrich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👤 Autor

**Prof. Dr. Dietmar Henrich**  
Professor für Medizintechnik  
Schwerpunkt: Physik, Medizinische Bildgebung, Educational Software

---

## 🙏 Danksagungen

- [Streamlit](https://streamlit.io) für das hervorragende Framework
- [Plotly](https://plotly.com) für interaktive Visualisierungen
- [NumPy](https://numpy.org) für numerische Berechnungen

---

<p align="center">
  <b>⭐ Wenn Ihnen dieses Projekt gefällt, geben Sie ihm einen Stern! ⭐</b>
</p>

<p align="center">
  Made with ❤️ for Physics Education
</p>
