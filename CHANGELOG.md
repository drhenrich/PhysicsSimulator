# Changelog

Alle wichtigen Änderungen an diesem Projekt werden in dieser Datei dokumentiert.

Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/),
und dieses Projekt folgt [Semantic Versioning](https://semver.org/lang/de/).

---

## [5.0.0] - 2024-11-25

### Hinzugefügt
- **Ultraschall-Modul** (`ultrasound_sim.py`, `ui_ultrasound.py`)
  - B-Mode Bildgebung mit Delay-and-Sum Beamforming
  - Punktstreuer-PSF Analyse
  - Carotis-Phantom mit laminarer Strömung
  - Farbdoppler mit Kasai-Autocorrelation
  - Lineararray-Simulation (16-128 Elemente)
  - Apodisation (Hanning, Hamming, None)
  - RF-Daten Export als NPZ
- Vollständige **GitHub-Dokumentation**
  - README.md mit Feature-Übersicht
  - DOCUMENTATION.md mit technischen Details
  - LICENSE (MIT)
  - CHANGELOG.md

### Geändert
- Hauptanwendung auf 8 Tabs erweitert
- i18n_bundle.py um Ultraschall-Übersetzungen ergänzt
- Versionsnummer auf 5.0 aktualisiert

---

## [4.0.0] - 2024-11-25

### Hinzugefügt
- **Mechanik-Modul komplett neu erstellt** (`ui_mech_bundle.py`)
  - 2D-Mechanik: Schiefer Wurf, Pendel, Federschwingung, Schiefe Ebene
  - 3D N-Körper-Simulation mit Velocity-Verlet Integration
  - Himmelsmechanik: Sonnensystem, Kepler-Bahnen, Lagrange-Punkte
  - Kollisionen: 1D/2D Stöße, Billard, Newton-Wiege
  - Figure-8 Lösung des Dreikörperproblems
  - Echtzeit-Animationen für alle Hauptsimulationen

### Geändert
- Body2D und Body3D Datenklassen mit Trail-Unterstützung
- NBodySimulator mit Kollisionserkennung und -auflösung
- Sonnensystem-Daten mit echten Planetenparametern

---

## [3.0.0] - 2024-11-24

### Hinzugefügt
- **Atomphysik-Modul** (`ui_atom_bundle.py`)
  - Bohr-Modell für H, He⁺, Li²⁺ (Z=1-3)
  - Animierte Elektronenübergänge
  - Photoeffekt mit 7 Materialien
  - Franck-Hertz Experiment (Hg, Ne)
  - Emissions- und Absorptionsspektren
  - Spektralserien (Lyman, Balmer, Paschen, Brackett, Pfund)

### Geändert
- Tab-Struktur auf 7 Tabs erweitert
- Übersetzungen für Atomphysik ergänzt

---

## [2.0.0] - 2024-11-24

### Hinzugefügt
- **Thermodynamik-Modul** (`ui_thermo_bundle.py`)
  - 1D/2D Wärmeleitung mit explizitem Euler-Verfahren
  - Zustandsänderungen (isotherm, isobar, isochor, adiabatisch)
  - Carnot- und Otto-Kreisprozesse
  - Gaskinetik mit Maxwell-Boltzmann-Verteilung
  - Echtzeit-Animationen für Wärmeleitung und Teilchenbewegung

### Geändert
- Modulare Architektur eingeführt
- UI von Physik-Code getrennt

---

## [1.0.0] - 2024-11-24

### Hinzugefügt
- Initiale Version des Physics Teaching Simulator
- **Mechanik-Modul** (Grundversion)
  - Einfache Mehrkörpersimulation
  - Presets für typische Szenarien
- **Optik-Modul** (`ui_optics_bundle.py`)
  - Geometrische Optik
  - Ray-Tracing
- **Medizintechnik-Module**
  - MRI & Bloch-Gleichungen (`ui_med_bundle.py`)
  - Röntgen & CT (`xray_ct.py`)
- **Elektrodynamik-Modul**
  - E-Feld und B-Feld Visualisierung
- **Internationalisierung** (`i18n_bundle.py`)
  - Deutsch und Englisch
- **Kern-Physikfunktionen** (`sim_core_bundle.py`)
  - Simulator-Klasse
  - Presets
  - Plotting-Funktionen

---

## Versionsschema

- **MAJOR**: Inkompatible API-Änderungen oder neue Hauptmodule
- **MINOR**: Neue Funktionen, abwärtskompatibel
- **PATCH**: Bugfixes, kleine Verbesserungen

---

## Geplante Features

- [ ] PET/SPECT Simulation
- [ ] Quantenmechanik (Wellenfunktionen, Tunneleffekt)
- [ ] Akustik (Raumakustik, Resonanz, Chladni-Figuren)
- [ ] Quiz-Modus mit automatischer Auswertung
- [ ] CSV/JSON Export für alle Simulationen
- [ ] Drag-and-Drop Objekteditor
- [ ] Dark Mode
- [ ] Mobile-optimierte Ansicht
