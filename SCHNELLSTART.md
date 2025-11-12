# 🚀 Schnellstart-Anleitung / Quick Start Guide

## Start der Anwendung

```bash
streamlit run teaching_physics_simulator_enhanced.py
```

---

## 🌐 Sprachauswahl / Language Selection

**Position:** Links oben in der Sidebar (first element in sidebar, top left)

```
┌────────────────────────────┐
│ 🌐 Language / Sprache      │
│   ├─ Deutsch 🇩🇪           │
│   └─ English 🇬🇧           │
└────────────────────────────┘
```

**Die gesamte Benutzeroberfläche wechselt sofort die Sprache!**
**The entire user interface switches language immediately!**

---

## 📦 Presets verwenden / Using Presets

### 1. Vordefinierte Presets laden / Load predefined presets

**Deutsch:**
1. Sidebar öffnen
2. Unter "Voreinstellungen" Preset auswählen
3. Button "📥 Preset laden" klicken
4. Daten erscheinen im Objekt-Editor

**English:**
1. Open sidebar
2. Under "Presets" select preset
3. Click button "📥 Load preset"
4. Data appears in object editor

### 2. Eigenes Preset speichern / Save custom preset

**Deutsch:**
1. Objekte im Editor konfigurieren
2. Zum Abschnitt "💾 Eigenes Preset speichern" scrollen
3. Namen eingeben
4. "💾 Speichern" klicken

**English:**
1. Configure objects in editor
2. Scroll to section "💾 Save custom preset"
3. Enter name
4. Click "💾 Save"

### 3. Preset importieren / Import preset

**Deutsch:**
1. Zum Abschnitt "📥 Preset importieren" scrollen
2. JSON-Datei hochladen
3. Preset wird automatisch geladen

**English:**
1. Scroll to section "📥 Import preset"
2. Upload JSON file
3. Preset loads automatically

### 4. Preset exportieren / Export preset

**Deutsch:**
1. Preset speichern (siehe oben)
2. In "📚 Gespeicherte eigene Presets" expandieren
3. Button "💾" beim gewünschten Preset klicken
4. JSON-Datei wird heruntergeladen

**English:**
1. Save preset (see above)
2. Expand "📚 Saved custom presets"
3. Click "💾" button for desired preset
4. JSON file downloads

---

## 🎯 Verfügbare Presets / Available Presets

### Mechanik / Mechanics
- ✅ **Geladenes Paar** / *Charged Pair*
- ✅ **Drei Ladungen** / *Three Charges*
- ✅ **Elastischer Stoß** / *Elastic Collision*
- ✅ **Inelastischer Stoß** / *Inelastic Collision*
- ✅ **Federsystem** / *Spring System*
- ✅ **Planetensystem** / *Planetary System*

### Optik / Optics
- ✅ **Einzelne Linse** / *Single Lens*
- ✅ **Zwei-Linsen-System** / *Two-Lens System*
- ✅ **Teleskop** / *Telescope*
- ✅ **Mikroskop** / *Microscope*

---

## 📊 Workflow

### Deutsch:
1. **Sprache wählen** → Deutsch 🇩🇪
2. **Preset laden** → z.B. "Elastischer Stoß"
3. **Parameter anpassen** (optional)
4. **Simulation starten** → Tab "▶️ Simulation"
5. **Ergebnisse analysieren** → Diagramme, Tabellen
6. **Daten exportieren** → Tab "💾 Export"

### English:
1. **Choose language** → English 🇬🇧
2. **Load preset** → e.g. "Elastic Collision"
3. **Adjust parameters** (optional)
4. **Run simulation** → Tab "▶️ Simulation"
5. **Analyze results** → Charts, tables
6. **Export data** → Tab "💾 Export"

---

## ⚙️ Simulationseinstellungen / Simulation Settings

### Wichtige Parameter / Important Parameters

| Deutsch | English | Bereich / Range |
|---------|---------|-----------------|
| Restitutionskoeffizient | Restitution coefficient | 0.0 - 1.0 |
| Luftwiderstand | Air resistance | 0.0 - 10.0 |
| Magnetfeld Bz | Magnetic field Bz | -1.0 - 1.0 T |
| Zeitschritt dt | Time step dt | 0.0001 - 0.1 s |
| Endzeit t_end | End time t_end | 0.1 - 100 s |

---

## 🔬 Optik / Optics

### Lichtquellen / Light Sources

**Punktquelle / Point Source:**
- Strahlen in alle Richtungen / Rays in all directions
- Anzahl wählbar / Number selectable

**Parallelbündel / Parallel Beam:**
- Parallel zur opt. Achse / Parallel to optical axis
- Für Teleskope / For telescopes

### Berechnungen / Calculations

**Linsengleichung / Lens Equation:**
```
1/f = 1/g + 1/b
```

- f = Brennweite / Focal length
- g = Gegenstandsweite / Object distance
- b = Bildweite / Image distance

---

## 💾 Datenexport / Data Export

### CSV (Mechanik / Mechanics)
```csv
time,Obj0_x,Obj0_y,Obj0_z,...
0.000,1.000,0.000,0.000,...
0.001,1.001,0.001,0.000,...
```

**Verwendbar in / Usable in:**
- Excel, LibreOffice Calc
- Python (pandas)
- Matlab, Octave
- Origin, Igor Pro

### JSON (Presets)
```json
{
  "name": "My Experiment",
  "bodies": [...],
  "connections": [...]
}
```

**Verwendbar für / Usable for:**
- Teilen von Experimenten / Sharing experiments
- Reproduzierbarkeit / Reproducibility
- Vorlagen / Templates

---

## 🆘 Hilfe / Help

### Problem: Sprachauswahl nicht sichtbar
**Lösung:** Sidebar mit Pfeil oben links öffnen
**Solution:** Open sidebar with arrow in top left

### Problem: Preset lädt nicht
**Lösung:** Sicherstellen, dass "(Keine)" / "(None)" nicht gewählt ist
**Solution:** Ensure "(None)" is not selected

### Problem: Simulation friert ein
**Lösung:** Zeitschritt vergrößern oder Endzeit reduzieren
**Solution:** Increase time step or reduce end time

### Problem: Optik-Modul fehlt
**Lösung:** optics_module.py ins gleiche Verzeichnis kopieren
**Solution:** Copy optics_module.py to same directory

---

## 📚 Weitere Informationen / More Information

Siehe / See: **VOLLSTÄNDIGE_DOKUMENTATION.md**

---

**Viel Erfolg! / Good luck! 🚀**
