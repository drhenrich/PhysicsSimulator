# Physics Teaching Simulator - Enhanced Version 2.0

## 🔬 Mehrsprachiger Physik-Simulator für die Lehre

Interaktive Simulation für **Mechanik**, **Elektrodynamik** und **Optik** mit vollständiger Deutsch/Englisch-Unterstützung und erweitertem Preset-System.

---

## 🚀 Quick Start

### 1. Installation der Abhängigkeiten

```bash
pip install streamlit numpy pandas plotly matplotlib
```

### 2. Dateien platzieren

Stellen Sie sicher, dass beide Dateien im **selben Verzeichnis** sind:
- `teaching_physics_simulator_enhanced.py` (Hauptprogramm)
- `optics_module.py` (Optik-Modul)

### 3. Starten

```bash
cd /pfad/zu/ihrem/verzeichnis
streamlit run teaching_physics_simulator_enhanced.py
```

Die Anwendung öffnet sich automatisch im Browser unter `http://localhost:8501`

---

## ✨ Neue Features Version 2.0

### 🌐 Mehrsprachigkeit
- **Deutsch 🇩🇪** und **Englisch 🇬🇧** vollständig
- Sprachauswahl links oben in der Sidebar
- Sofortiger Wechsel ohne Neustart
- Über 100 übersetzte UI-Elemente

### 📦 Erweitertes Preset-System
- **6 Mechanik-Presets** vorinstalliert
- **4 Optik-Presets** vorinstalliert
- **Eigene Presets speichern** und wiederverwenden
- **JSON Import/Export** zum Teilen
- **Preset-Verwaltung** (Löschen, Umbenennen)

### ⚙️ Alle Original-Features
- Mechanik (Newton, Kollisionen, Federn)
- Elektrodynamik (Coulomb, Lorentz)
- Optik (Linsen, Spiegel, Strahlengang)
- Datenexport (CSV, JSON)

---

## 🎯 Verwendung

### Schritt 1: Sprache wählen
1. Öffnen Sie die Sidebar (Pfeil ≡ oben links)
2. Ganz oben: Dropdown "🌐 Language / Sprache"
3. Wählen Sie **Deutsch 🇩🇪** oder **English 🇬🇧**

### Schritt 2: Preset laden
1. In Sidebar unter "Voreinstellungen" / "Presets"
2. Preset aus Dropdown wählen (z.B. "Elastischer Stoß")
3. Button "📥 Preset laden" / "📥 Load preset" klicken
4. Daten erscheinen im Objekt-Editor

### Schritt 3: Simulation starten
1. Wechseln Sie zum Tab "▶️ Simulation"
2. Klicken Sie "▶️ Simulation starten" / "▶️ Run simulation"
3. Warten Sie auf Berechnung
4. Analysieren Sie Ergebnisse in Diagrammen

### Schritt 4: Daten exportieren (optional)
1. Tab "💾 Export"
2. CSV für Zahlendaten oder JSON für Presets
3. Download-Button klicken

---

## 📦 Verfügbare Presets

### Mechanik
| Preset | Beschreibung |
|--------|--------------|
| Geladenes Paar | Zwei entgegengesetzt geladene Teilchen |
| Drei Ladungen | Elektrostatische Konfiguration |
| Elastischer Stoß | Perfekt elastische Kollision (e=1.0) |
| Inelastischer Stoß | Energie-Verlust bei Kollision (e<1.0) |
| Federsystem | Harmonischer Oszillator |
| Planetensystem | Keplersche Bahnbewegung |

### Optik
| Preset | Beschreibung |
|--------|--------------|
| Einzelne Linse | Bildkonstruktion mit Brennpunkten |
| Zwei-Linsen-System | Kombinierte optische Wirkung |
| Teleskop | Objektiv + Okular |
| Mikroskop | Starke Vergrößerung |

---

## 💾 Eigene Presets

### Speichern
1. Konfigurieren Sie Objekte im Editor
2. Scrollen Sie zu "💾 Eigenes Preset speichern"
3. Geben Sie einen Namen ein
4. Klicken Sie "💾 Speichern"

### Exportieren
1. Nach dem Speichern erscheint das Preset in der Liste
2. Expandieren Sie "📚 Gespeicherte eigene Presets"
3. Klicken Sie "💾" beim gewünschten Preset
4. JSON-Datei wird heruntergeladen

### Importieren
1. Scrollen Sie zu "📥 Preset importieren"
2. Laden Sie eine JSON-Datei hoch
3. Preset wird automatisch verfügbar

---

## 🛠️ Fehlerbehebung

### Problem: "Anwendungsfehler" beim Start
**Ursache:** Alte Version oder falsche Datei  
**Lösung:** 
- Stellen Sie sicher, dass Sie die neueste `teaching_physics_simulator_enhanced.py` verwenden
- Löschen Sie alte Versionen aus dem Papierkorb
- Starten Sie neu

### Problem: Sprachauswahl nicht sichtbar
**Ursache:** Sidebar nicht geöffnet  
**Lösung:** Klicken Sie auf ≡ (Hamburger-Menü) oben links

### Problem: Optik-Modul nicht verfügbar
**Ursache:** `optics_module.py` fehlt  
**Lösung:** Kopieren Sie `optics_module.py` ins gleiche Verzeichnis

### Problem: Preset lädt nicht
**Ursache:** "(Keine)" / "(None)" ausgewählt  
**Lösung:** Wählen Sie ein tatsächliches Preset aus dem Dropdown

### Problem: Simulation friert ein
**Ursache:** Zu viele Zeitschritte  
**Lösung:** 
- Vergrößern Sie den Zeitschritt `dt`
- Reduzieren Sie die Endzeit `t_end`
- Verwenden Sie weniger Objekte (< 10)

---

## 📋 Systemanforderungen

### Minimum
- **Python:** 3.8 oder höher
- **RAM:** 2 GB
- **Browser:** Chrome, Firefox, Safari (aktuell)
- **Internet:** Für CDN-Ressourcen

### Empfohlen
- **Python:** 3.10+
- **RAM:** 4 GB+
- **CPU:** Multi-Core für große Simulationen

---

## 📚 Dokumentation

Ausführliche Dokumentation finden Sie in:
- **VOLLSTÄNDIGE_DOKUMENTATION.md** - Alle Features im Detail
- **SCHNELLSTART.md** - Quick Reference (DE/EN)
- **CHANGELOG.md** - Änderungshistorie

---

## 🎓 Didaktischer Einsatz

### Zielgruppen
- Schüler (Oberstufe Physik)
- Studenten (Bachelor Physik, Ingenieurwesen)
- Lehrer (Demonstrationen)
- Interessierte (Selbststudium)

### Einsatzszenarien
- **Präsenzunterricht:** Live-Demonstrationen
- **Online-Lehre:** Screen-Sharing
- **Hausaufgaben:** Eigenständige Experimente
- **Projekte:** Forschung und Dokumentation

### Lernziele
- Newtonsche Mechanik verstehen
- Erhaltungssätze verifizieren
- Elektromagnetismus visualisieren
- Optische Abbildung konstruieren

---

## 🔧 Technische Details

### Architektur
```
teaching_physics_simulator_enhanced.py
├─ Übersetzungssystem (DE/EN)
├─ Physik-Engine (RK4, Velocity-Verlet)
├─ Preset-System (Load/Save/Import/Export)
├─ Visualisierung (Plotly, Matplotlib)
└─ Streamlit UI (Mehrsprachig)
```

### Numerische Methoden
- **Zeitintegration:** Velocity-Verlet (symplektisch)
- **Kollisionen:** Impulssatz + Restitution
- **Kräfte:** Newton, Coulomb, Lorentz
- **Optik:** Strahlenoptik (geometrisch)

---

## 📄 Lizenz

Frei verwendbar für Bildung und Forschung.

**Autor:** Prof. Dr.rer.nat. Dietmar Henrich
**Version:** 2.0 (Mehrsprachig + Presets)  
**Datum:** 12. November 2025

---

## 🆘 Support

Bei Problemen oder Fragen:
1. Prüfen Sie die Dokumentation
2. Schauen Sie in CHANGELOG.md nach bekannten Problemen
3. Kontaktieren Sie den Autor

---

## 🎉 Viel Erfolg beim Experimentieren!

**Happy Simulating! 🚀🔬**

---

**Hinweis:** Diese Anwendung ist ein Lehrmittel und dient der Illustration physikalischer Konzepte. Für präzise wissenschaftliche Berechnungen verwenden Sie spezialisierte Software.
