# Physics Teaching Simulator - Vollständige Dokumentation
## Version 2.0 mit Mehrsprachigkeit und erweiterten Preset-Funktionen

---

## 📋 Inhaltsverzeichnis

1. [Neue Features](#neue-features)
2. [Sprachauswahl](#sprachauswahl)
3. [Preset-System](#preset-system)
4. [Mechanik-Simulationen](#mechanik-simulationen)
5. [Elektrodynamik](#elektrodynamik)
6. [Optik](#optik)
7. [Technische Details](#technische-details)
8. [Verwendung](#verwendung)

---

## 🆕 Neue Features

### 1. Mehrsprachige Benutzeroberfläche
- **Deutsch** und **Englisch** vollständig unterstützt
- Sofortiger Sprachwechsel ohne Neustart
- Über 100 übersetzte UI-Elemente
- Fachterminologie wissenschaftlich korrekt

### 2. Erweiterte Preset-Funktionalität
- **Vordefinierte Presets** für häufige Szenarien
- **Eigene Presets speichern** für Experimente
- **JSON Import/Export** zum Teilen von Konfigurationen
- **Persistente Speicherung** über Session State

### 3. Alle Original-Features erhalten
- Mechanik (Gravitation, Kollisionen, Verbindungen)
- Elektrodynamik (geladene Teilchen, Magnetfeld)
- Optik (Linsen, Spiegel, Strahlengang)
- Datenexport (CSV, JSON)

---

## 🌐 Sprachauswahl

### Position und Bedienung

Die Sprachauswahl befindet sich **links oben in der Sidebar**, als erstes Element vor allen anderen Einstellungen.

```python
# Dropdown-Menü mit Flaggen-Symbolen
🌐 Language / Sprache
   ├─ Deutsch 🇩🇪
   └─ English 🇬🇧
```

### Features
- **Bidirektionale Beschriftung**: Immer beide Sprachen sichtbar
- **Flaggen-Symbole**: Visuell intuitiv (🇩🇪 🇬🇧)
- **Session-Persistenz**: Gewählte Sprache bleibt erhalten
- **Sofortige Aktualisierung**: Gesamte UI wird sofort übersetzt

### Übersetztes Vokabular

#### Hauptnavigation
| Deutsch | English |
|---------|---------|
| Physics Teaching Simulator - Erweiterte Version | Physics Teaching Simulator - Enhanced Version |
| Objekt-Editor | Object Editor |
| Simulation | Simulation |
| Optik | Optics |
| Export | Export |

#### Simulationseinstellungen
| Deutsch | English |
|---------|---------|
| Voreinstellungen | Presets |
| Physik-Parameter | Physics Parameters |
| Restitutionskoeffizient | Restitution coefficient |
| Luftwiderstand | Air resistance |
| Relativistische Korrektur | Relativistic correction |
| Magnetfeld | Magnetic Field |

#### Optik-Begriffe
| Deutsch | English |
|---------|---------|
| Brennweite | Focal length |
| Brechkraft | Optical power |
| Gegenstandsweite | Object distance |
| Bildweite | Image distance |
| Vergrößerung | Magnification |
| Brennpunkte | Focal points |
| Konstruktionsstrahlen | Construction rays |

---

## 📦 Preset-System

### Vordefinierte Presets

Die Anwendung enthält folgende vorgefertigte Szenarien:

#### Mechanik
1. **Geladenes Paar** / *Charged Pair*
   - Zwei entgegengesetzt geladene Teilchen
   - Demonstriert Coulomb-Kraft

2. **Drei Ladungen** / *Three Charges*
   - Konfiguration: +3µC, -1µC, -3µC
   - Zeigt elektrisches Feld

3. **Elastischer Stoß** / *Elastic Collision*
   - Restitutionskoeffizient e = 1.0
   - Impuls- und Energieerhaltung

4. **Inelastischer Stoß** / *Inelastic Collision*
   - Restitutionskoeffizient e < 1.0
   - Energieverlust durch Verformung

5. **Federsystem** / *Spring System*
   - Zwei Massen mit elastischer Verbindung
   - Harmonische Schwingung

6. **Planetensystem** / *Planetary System*
   - Sonne mit ein oder zwei Planeten
   - Keplersche Gesetze

#### Optik
1. **Einzelne Linse** / *Single Lens*
   - Eine Sammellinse
   - Bildkonstruktion

2. **Zwei-Linsen-System** / *Two-Lens System*
   - Kombinierte Linsen
   - Vergrößerung und Brennweiten

3. **Teleskop** / *Telescope*
   - Objektiv + Okular
   - Parallele Strahlen

4. **Mikroskop** / *Microscope*
   - Kurze Brennweiten
   - Starke Vergrößerung

### Preset laden

**Schritte:**
1. Sidebar öffnen
2. Unter "Voreinstellungen" / "Presets" Dropdown-Menü öffnen
3. Gewünschtes Preset auswählen
4. Button "📥 Preset laden" / "📥 Load preset" klicken
5. Daten werden in Objekt-Editor geladen

**Code-Struktur:**
```python
PRESETS = {
    'Elastischer Stoß': scenario_elastic_collision,
    'Federsystem': scenario_spring_system,
    'Planetensystem': scenario_planetary_scaled,
    # ...
}
```

### Eigenes Preset speichern

Experimentelle Konfigurationen können gespeichert werden:

**Schritte:**
1. Objekte im Editor konfigurieren
2. Verbindungen definieren (optional)
3. Unter "💾 Eigenes Preset speichern" Namen eingeben
4. Button "💾 Speichern" / "💾 Save" klicken
5. Preset erscheint in Liste und Dropdown

**Features:**
- Wird in Session State gespeichert
- Als JSON exportierbar
- Wiederverwendbar in aktueller Sitzung

### Preset Import/Export

#### Export
Gespeicherte Presets können als JSON-Datei exportiert werden:

1. Preset speichern (siehe oben)
2. In "📚 Gespeicherte eigene Presets" expandieren
3. Bei gewünschtem Preset Button "💾" klicken
4. JSON-Datei wird heruntergeladen

**JSON-Format:**
```json
{
  "name": "Mein Experiment",
  "bodies": [
    {
      "name": "Körper A",
      "pos": [1.0, 0.0, 0.0],
      "vel": [0.0, 1.0, 0.0],
      "mass": 2.0,
      "charge": 0.0,
      "t0": 0.0,
      "dt": 0.001,
      "t_end": 5.0,
      "color": "red"
    }
  ],
  "connections": [
    {
      "i": 0,
      "j": 1,
      "typ": "elastic",
      "strength": 10.0,
      "rest_length": 2.0
    }
  ]
}
```

#### Import
JSON-Presets von anderen Nutzern importieren:

1. Unter "📥 Preset importieren" / "📥 Import preset"
2. JSON-Datei hochladen
3. Preset wird automatisch geladen und zur Liste hinzugefügt
4. Bei Namenskonflikt wird automatisch umbenannt

**Verwendung:**
- Teilen von Experimenten
- Vorlagen für Lehre
- Reproduzierbare Simulationen

---

## ⚙️ Mechanik-Simulationen

### Objekt-Editor

Vollständige Kontrolle über alle Parameter:

**Bearbeitbare Felder:**
- **Name**: Bezeichnung des Objekts
- **Position**: x, y, z (m)
- **Geschwindigkeit**: vx, vy, vz (m/s)
- **Masse**: (kg)
- **Ladung**: (C)
- **Zeitparameter**: t0, dt, t_end (s)
- **Farbe**: Darstellung in Plots

**Verbindungen:**
Format: `i-j:typ:stärke`
- **i, j**: Objekt-Indizes (0-basiert)
- **typ**: `elastic` oder `rigid`
- **stärke**: Federkonstante (N/m)

Beispiel:
```
0-1:elastic:10.0
1-2:rigid:1e8
```

### Physik-Parameter

**Kollisionen:**
- Restitutionskoeffizient: 0.0 (inelastisch) bis 1.0 (elastisch)
- Automatische Kollisionserkennung
- Impuls- und Energieerhaltung

**Reibung:**
- Linear: F = -c·v
- Quadratisch: F = -c·|v|·v
- Einstellbar über Koeffizient

**Relativistische Effekte:**
- Gamma-Faktor bei hohen Geschwindigkeiten
- Korrekte Energie-Impuls-Relation
- Optional aktivierbar

### Koordinatensysteme

Mehrere Darstellungen parallel:

1. **Kartesisch** (x, y, z)
   - Standard-Koordinaten
   - Intuitive Darstellung

2. **Impuls** (px, py, pz)
   - Phasenraum-Darstellung
   - Zeigt Erhaltungsgrößen

3. **Energie** (E_kin, E_pot, E_tot)
   - Energieerhaltung sichtbar
   - Zeitlicher Verlauf

4. **Schwerpunkt-relativ**
   - COM-System
   - Eliminiert Schwerpunktsbewegung

---

## ⚡ Elektrodynamik

### Geladene Teilchen

**Coulomb-Kraft:**
```
F = k_e · q1·q2 / r²
```
- k_e = 8.99×10⁹ N·m²/C²
- Automatische Berechnung

**Magnetfeld:**
- Eingabe als Vektor (Bx, By, Bz) in Tesla
- Lorentz-Kraft: F = q·(v × B)
- Kreisbewegung bei homogenem Feld

### Visualisierung

**Potentialfeld:**
- Äquipotentiallinien
- Feldlinien (Stromlinien)
- Farbkodierung

**Feldstärke:**
- Vektorfeld-Darstellung
- Pfeile zeigen Richtung
- Länge entspricht Stärke

---

## 🔬 Optik

### Optische Elemente

**Linsen:**
- Brennweite f (positiv = konvex, negativ = konkav)
- Brechkraft D = 1/f (in Dioptrien)
- Durchmesser (Apertur)

**Spiegel:**
- Ebene oder gekrümmt
- Neigungswinkel
- Höhe

**Schirme:**
- Bildebene
- Höhe einstellbar

**Blenden:**
- Durchmesser
- Begrenzt Strahlenbündel

### Lichtquellen

**Punktquelle:**
- Strahlen in alle Richtungen
- Anzahl wählbar
- Winkelverteilung

**Parallelbündel:**
- Parallel zur optischen Achse
- Simuliert unendlich ferne Quelle
- Für Teleskop-Optik

### Berechnungen

**Linsengleichung:**
```
1/f = 1/g + 1/b
```
- f: Brennweite
- g: Gegenstandsweite
- b: Bildweite

**Vergrößerung:**
```
V = b/g = B/G
```
- B: Bildgröße
- G: Gegenstandsgröße

**Bildtyp:**
- **Reell**: V < 0 (umgekehrt)
- **Virtuell**: V > 0 (aufrecht)

### Strahlengang

**Hauptstrahlen:**
1. Parallelstrahl → durch Brennpunkt
2. Brennpunktstrahl → parallel
3. Mittelpunktstrahl → ungebrochen

**Konstruktion:**
- Automatische Strahlenverfolg ung
- Reflexion und Brechung
- Intensitätsverluste

---

## 💾 Datenexport

### CSV-Export

**Mechanik-Daten:**
```csv
time,Obj0_x,Obj0_y,Obj0_z,Obj1_x,...
0.000,1.000,0.000,0.000,-1.000,...
0.001,1.001,0.001,0.000,-0.999,...
```

**Verwendung:**
- Excel/LibreOffice Calc
- Python (pandas)
- Matlab/Octave
- Eigene Analysen

### JSON-Export

**Preset-Format:**
Siehe [Preset Import/Export](#preset-importexport)

**Optik-System:**
```json
{
  "elements": [
    {
      "type": "Lens",
      "position": 0.0,
      "focal_length": 0.2,
      "diameter": 0.1
    }
  ],
  "sources": [...]
}
```

---

## 🔧 Technische Details

### Architektur

```
teaching_physics_simulator_enhanced.py
├─ TRANSLATIONS Dictionary (Zeilen ~15-266)
├─ get_translation() Funktion
├─ Physikalische Konstanten
├─ Datenklassen (Body, Connection, CollisionEvent)
├─ Simulator-Klasse
│  ├─ Kräfteberechnung
│  ├─ Zeitintegration (RK4, Velocity-Verlet)
│  ├─ Kollisionserkennung
│  └─ Erhaltungsgrößen
├─ Preset-Funktionen (scenario_xxx)
├─ PRESETS Dictionary
├─ Koordinaten-Transformationen
├─ Visualisierung (Plotly, Matplotlib)
├─ Export-Funktionen (CSV, JSON)
└─ Streamlit UI
   ├─ Sprachauswahl (Sidebar)
   ├─ Preset-Verwaltung
   ├─ Objekt-Editor
   ├─ Simulation
   ├─ Optik
   └─ Export
```

### Übersetzungssystem

**Implementation:**
```python
# 1. Dictionary mit allen Sprachen
TRANSLATIONS = {
    'de': {'key': 'Deutscher Text', ...},
    'en': {'key': 'English text', ...}
}

# 2. Getter-Funktion
def get_translation(key: str, lang: str = 'de') -> str:
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][key]
    elif key in TRANSLATIONS['de']:
        return TRANSLATIONS['de'][key]  # Fallback
    else:
        return key  # Notfall

# 3. Verwendung in UI
lang = st.selectbox('🌐 Language / Sprache', ['de', 'en'])
t = lambda key: get_translation(key, lang)
st.title(t('title'))  # Übersetzter Titel
```

**Vorteile:**
- Zentrale Verwaltung
- Einfache Erweiterung
- Fallback-Mechanismus
- Keine API-Aufrufe
- Zero Performance Impact

### Numerische Methoden

**Zeitintegration:**
1. **Velocity-Verlet** (Standard)
   - Symplektisch
   - 2. Ordnung
   - Energieerhaltung gut

2. **Runge-Kutta 4** (Alternative)
   - Explizit
   - 4. Ordnung
   - Vielseitig

**Kollisionen:**
- Elastisch/inelastisch wählbar
- Impulssatz exakt
- Separation nach Kollision
- Event-Logging

**Verbindungen:**
- Federkräfte (Hooke)
- Constraint-Enforcement (SHAKE)
- Iterative Korrektur

### Performance

**Optimierungen:**
- NumPy-Vektorisierung
- Adaptive Zeitschritte
- Session State Caching
- Lazy Loading

**Limits:**
- Max. 10 Objekte empfohlen
- Max. 800.000 Zeitschritte
- Abhängig von Hardware

---

## 🚀 Verwendung

### Installation

**Voraussetzungen:**
```bash
pip install streamlit numpy pandas plotly matplotlib
```

**Optional (für Optik):**
```bash
# optics_module.py im gleichen Verzeichnis
```

### Start

```bash
streamlit run teaching_physics_simulator_enhanced.py
```

**Browser öffnet automatisch:**
```
http://localhost:8501
```

### Workflow

1. **Sprache wählen**
   - Sidebar öffnen
   - Deutsch oder Englisch

2. **Szenario laden**
   - Preset aus Dropdown
   - "Preset laden" klicken
   - Oder eigene Konfiguration

3. **Parameter anpassen**
   - Objekt-Editor verwenden
   - Physik-Parameter einstellen
   - Verbindungen definieren

4. **Simulation starten**
   - Tab "Simulation"
   - Button "Simulation starten"
   - Warten...

5. **Ergebnisse analysieren**
   - 3D-Visualisierung
   - Energie-/Impulsdiagramme
   - Kollisions-Tabelle

6. **Daten exportieren**
   - Tab "Export"
   - CSV für Zahlendaten
   - JSON für Preset

### Beispiel-Session

```python
# 1. Elastischer Stoß laden
# Preset: "Elastischer Stoß"

# 2. Parameter ändern
# Masse Körper 1: 1.0 → 2.0 kg
# Geschwindigkeit: 2.0 → 3.0 m/s

# 3. Simulation (3 Sekunden)
# Restitution: 1.0 (elastisch)
# Ohne Reibung

# 4. Ergebnis
# → Energieerhaltung perfekt
# → Impuls erhalten
# → 1 Kollision bei t ≈ 1.0s

# 5. Als "Stoß 3-2" speichern
# 6. JSON exportieren
```

---

## 📊 Didaktische Anwendungen

### Mechanik
- **Impulserhaltung** bei Stößen demonstrieren
- **Energieerhaltung** mit/ohne Reibung
- **Schwingungen** (harmonisch, gedämpft)
- **Chaos** (Doppelpendel)
- **Keplersche Gesetze** (Planetensystem)

### Elektrodynamik
- **Coulomb-Kraft** zwischen Ladungen
- **Elektrisches Feld** visualisieren
- **Lorentz-Kraft** im Magnetfeld
- **Zyklotronbewegung**

### Optik
- **Linsengleichung** experimentell
- **Bildkonstruktion** mit Hauptstrahlen
- **Teleskop/Mikroskop** Aufbau verstehen
- **Brechung und Reflexion**

### Vorteile
- ✅ Interaktiv und experimentell
- ✅ Sofortige visuelle Rückmeldung
- ✅ Parameterstudien einfach
- ✅ Reproduzierbare Ergebnisse
- ✅ Export für weitere Analyse
- ✅ Mehrsprachig für internationale Lehre

---

## 🌍 Internationalisierung

### Aktuell unterstützt
- 🇩🇪 **Deutsch** (Muttersprache der Wissenschaft ;-)
- 🇬🇧 **Englisch** (Lingua franca)

### Erweiterbar auf
- 🇫🇷 Französisch
- 🇪🇸 Spanisch
- 🇮🇹 Italienisch
- 🇯🇵 Japanisch
- ...

**Vorgehen:**
1. TRANSLATIONS Dictionary erweitern
2. Neue Sprache in selectbox
3. Fertig!

---

## 📝 Lizenz & Credits

**Autor:** Dr. Heinrich (Professor für Medizintechnik)
**Version:** 2.0 (mit Mehrsprachigkeit)
**Datum:** 12. November 2025

**Verwendete Bibliotheken:**
- Streamlit (UI-Framework)
- NumPy (Numerik)
- Plotly (Interaktive Plots)
- Matplotlib (Statische Plots)
- Pandas (Datentabellen)

**Verwendung:**
Frei verwendbar für Lehre und Forschung.

---

## 🐛 Bekannte Einschränkungen

1. **Browser-Abhängigkeit**
   - Läuft nur in modernen Browsern
   - JavaScript muss aktiviert sein

2. **Performance**
   - Große Simulationen (>10 Objekte) langsam
   - Lange Zeiträume (>10000 Steps) dauern

3. **Optik**
   - Nur geometrische Optik
   - Keine Wellenoptik
   - Dünne-Linsen-Näherung

4. **Numerik**
   - Keine Quantenmechanik
   - Relativität nur näherungsweise
   - Softening für Singularitäten

---

## 🔮 Geplante Erweiterungen

- [ ] Weitere Sprachen (FR, ES, IT)
- [ ] 3D-Optik mit Z-Komponente
- [ ] Wellenoptik (Interferenz, Beugung)
- [ ] Quantenmechanik-Modul
- [ ] Cloud-Speicherung von Presets
- [ ] Kollaborative Simulationen
- [ ] Mobile App-Version
- [ ] VR/AR-Integration

---

**Ende der Dokumentation**

Bei Fragen oder Problemen: siehe Kommentare im Code oder kontaktieren Sie den Autor.

*Viel Erfolg beim Experimentieren! 🚀🔬*
