# 🔬 Physik-Simulator für die Lehre

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-6.0-purple.svg)]()

Eine umfassende, interaktive Simulationsplattform für den Physik- und Medizintechnik-Unterricht. Entwickelt mit Python und Streamlit, bietet der Simulator **10 Fachmodule** mit über **50 interaktiven Visualisierungen**.

## ✨ Hauptmerkmale

- 🎓 **Didaktisch konzipiert** — Optimiert für Vorlesungen und Praktika
- 🌍 **Zweisprachig** — Vollständige DE/EN Unterstützung
- 🎬 **Flüssige Animationen** — Client-seitige Plotly-Frame-Animationen
- 📱 **Responsive** — Funktioniert auf Desktop und Tablet
- 🔧 **Modular** — Einfach erweiterbar

## 📚 Module

| Modul | Beschreibung | Simulationen |
|-------|--------------|--------------|
| ⚙️ **Mechanik** | Klassische Mechanik & Himmelsmechanik | Wurf, Pendel, N-Körper, Stöße, Billard |
| 🌡️ **Thermodynamik** | Wärmelehre & Gastheorie | Wärmeleitung 1D/2D, Kreisprozesse, Gaskinetik |
| ⚛️ **Atomphysik** | Quantenphänomene | Bohr-Modell, Photoeffekt, Franck-Hertz, Spektren |
| 🎵 **Schwingungen** | Oszillationen & Akustik | Oszillatoren, Schwebungen, Stehende Wellen, Doppler |
| 🔭 **Optik** | Geometrische Optik | Linsen, Spiegel, Strahlengänge |
| ☢️ **Kernphysik** | Radioaktivität & Strahlenschutz | Zerfall, Zerfallsreihen, Dosimetrie, Abschirmung |
| 🩻 **Röntgen/CT** | Medizinische Bildgebung | Absorption, CT-Rekonstruktion, Hounsfield |
| 🧲 **MRI & Bloch** | Kernspinresonanz | Bloch-Gleichungen, T1/T2-Relaxation, Spinecho |
| 🔊 **Ultraschall** | Sonographie | Wellenausbreitung, Beamforming, B-Mode |
| ⚡ **Elektrodynamik** | Elektrostatik | Feldlinien, Potentiale, Poisson-Gleichung |

## 🚀 Schnellstart

### Installation

```bash
# Repository klonen
git clone https://github.com/IhrUsername/physics-simulator.git
cd physics-simulator

# Abhängigkeiten installieren
pip install -r requirements.txt

# Simulator starten
streamlit run physics_sim.py
```

### Anforderungen

- Python 3.9+
- Streamlit ≥ 1.28.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- Plotly ≥ 5.18.0

## 🎬 Animationen

Alle Animationen nutzen **Plotly-Frame-Technologie** für flüssige, client-seitige Wiedergabe:

```
▶️ Play   — Animation starten
⏸️ Pause  — Animation anhalten
🔄 Reset  — Zurück zum Anfang
```

**Animierte Simulationen:**
- Schiefer Wurf & Pendel
- Elastische/Inelastische Stöße
- Billard & Newton-Wiege
- Wärmeleitung (1D/2D)
- Gaskinetik (Maxwell-Boltzmann)
- Elektronenübergänge (Bohr)
- Photoemission
- Stehende Wellen
- Doppler-Effekt

## 📖 Dokumentation

Ausführliche Dokumentation: [DOCUMENTATION.md](DOCUMENTATION.md)

### Projektstruktur

```
physics-simulator/
├── physics_sim.py          # Hauptanwendung
├── i18n_bundle.py          # Übersetzungen
├── sim_core_bundle.py      # Kernfunktionen
├── ui_mech_bundle.py       # Mechanik-Modul
├── ui_thermo_bundle.py     # Thermodynamik-Modul
├── ui_atom_bundle.py       # Atomphysik-Modul
├── ui_oscillations_bundle.py # Schwingungen-Modul
├── ui_optics_bundle.py     # Optik-Modul
├── ui_nuclear_bundle.py    # Kernphysik-Modul
├── ui_med_bundle.py        # Medizinphysik-Modul
├── ui_ultrasound.py        # Ultraschall-UI
├── ultrasound_sim.py       # Ultraschall-Simulation
├── xray_ct.py              # CT-Rekonstruktion
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
├── CHANGELOG.md
└── LICENSE
```

## 🔬 Physikalische Grundlagen

Der Simulator implementiert folgende physikalische Modelle:

### Mechanik
- Newton'sche Bewegungsgleichungen
- Runge-Kutta 4 Integration
- Impuls- und Energieerhaltung
- Gravitationsgesetz (N-Körper)

### Thermodynamik
- Fourier'sche Wärmeleitungsgleichung
- Ideale Gasgleichung
- Carnot- und Otto-Kreisprozesse
- Maxwell-Boltzmann-Verteilung

### Atomphysik
- Bohr'sches Atommodell
- Einstein'sche Photoeffekt-Gleichung
- Franck-Hertz-Experiment
- Emissions-/Absorptionsspektren

### Kernphysik
- Zerfallsgesetz: A(t) = A₀·e^(-λt)
- Bateman-Gleichungen (Zerfallsketten)
- Abstandsgesetz: Ḋ = A·Γ/r²
- Abschirmung: I = I₀·e^(-μx)

### Schwingungen
- Gedämpfter harmonischer Oszillator
- Gekoppelte Oszillatoren
- Doppler-Effekt: f' = f·(c±v_o)/(c∓v_s)

## 🎯 Einsatzszenarien

- **Vorlesungen** — Live-Demonstrationen physikalischer Phänomene
- **Praktika** — Virtuelle Experimente und Datenanalyse
- **Selbststudium** — Interaktives Lernen mit Parametervariation
- **Prüfungsvorbereitung** — Visualisierung komplexer Zusammenhänge

## 🤝 Beitragen

Beiträge sind willkommen!

1. Fork erstellen
2. Feature-Branch anlegen (`git checkout -b feature/NeuesFunktion`)
3. Änderungen committen (`git commit -m 'Neue Funktion hinzugefügt'`)
4. Branch pushen (`git push origin feature/NeuesFunktion`)
5. Pull Request erstellen

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert — siehe [LICENSE](LICENSE) für Details.

## 👤 Autor

**Prof. Dr. Dietmar Henrich**  
Medizintechnik & Physik

---

<p align="center">
  <i>Entwickelt für die Lehre. Inspiriert von der Physik.</i>
</p>
