# Changelog

Alle wesentlichen Änderungen an diesem Projekt werden hier dokumentiert.

## [6.0] - 2024-11

### Hinzugefügt
- **Schwingungen & Akustik Modul** (~1100 Zeilen)
  - Gedämpfter harmonischer Oszillator
  - Gekoppelte Oszillatoren mit Normalmoden
  - Schwebungen mit FFT-Spektrum
  - Stehende Wellen (animiert)
  - Doppler-Effekt mit Wellenfronten-Animation
  - Resonanzkurven

- **Kernphysik & Strahlenschutz Modul** (~850 Zeilen)
  - Radioaktiver Zerfall (10 Nuklide)
  - Natürliche Zerfallsreihen (U-238, Th-232, U-235)
  - Dosimetrie mit Abstandsgesetz
  - Abschirmungsberechnung (5 Materialien)
  - Dosisgrenzwerte nach StrlSchV

### Geändert
- **Alle Animationen auf Plotly-Frames umgestellt**
  - Client-seitige Wiedergabe (bis 60 FPS)
  - Einheitliche Play/Pause/Reset-Steuerung
  - Keine Server-Roundtrips mehr

- **Elektrostatik-Modul überarbeitet**
  - Farbige Felddarstellung (wie Lehrbuch)
  - Potential mit RdBu_r-Farbskala
  - Feldstärke mit Viridis-Farbskala
  - Korrigierter Feldlinien-Algorithmus

### Behoben
- ValueError in Elektrostatik (zip()-Funktion für Feldlinien)
- Beamforming-Fehler in Ultraschall-Modul

## [5.0] - 2024-10

### Hinzugefügt
- Ultraschall-Modul
- GitHub-Dokumentation (README, DOCUMENTATION)
- MIT-Lizenz

## [4.0] - 2024-09

### Hinzugefügt
- Mechanik-Modul (~1400 Zeilen)
  - 3D N-Körper-Simulation
  - Stoßsimulationen (1D/2D)
  - Billard-Simulation
  - Newton-Wiege

## [3.0] - 2024-08

### Hinzugefügt
- Atomphysik-Modul
- Optik-Modul
- MRI & Bloch-Gleichungen

## [2.0] - 2024-07

### Hinzugefügt
- Thermodynamik-Modul
- CT-Rekonstruktion

## [1.0] - 2024-06

### Erstveröffentlichung
- Grundstruktur mit Streamlit
- Internationalisierung (DE/EN)
- Basis-Mechanik
