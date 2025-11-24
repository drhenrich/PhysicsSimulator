
# ---- ADDED: table and GUI helpers to fix table update/editing race conditions ----
from PyQt5 import QtWidgets, QtCore, QtGui
def ensure_table_commit_and_resize(table: QtWidgets.QTableWidget, new_rowcount: int, default_value="0"):
    """Safely update table row count while committing any open editor and avoiding signal loops."""
    # commit and close any editor
    if table.state() == QtWidgets.QAbstractItemView.EditingState:
        editor = table.focusWidget()
        if editor:
            try:
                table.commitData(editor)
            except Exception:
                pass
            try:
                table.closeEditor(editor, QtWidgets.QAbstractItemDelegate.NoHint)
            except Exception:
                pass
    # use a simple updating flag on table to avoid reacting to programmatic changes
    if not hasattr(table, "_updating_table"):
        table._updating_table = False
    table._updating_table = True
    try:
        table.blockSignals(True)
        current = table.rowCount()
        table.setRowCount(new_rowcount)
        if new_rowcount > current:
            for r in range(current, new_rowcount):
                for c in range(table.columnCount()):
                    if table.item(r, c) is None:
                        table.setItem(r, c, QtWidgets.QTableWidgetItem(str(default_value)))
    finally:
        table.blockSignals(False)
        table._updating_table = False

# Helper decorator to ignore callbacks triggered by programmatic changes
def ignore_programmatic(fn):
    def wrapper(*args, **kwargs):
        # assume first arg is self and that self.table exists, or inspect args for a QTableWidget
        self = args[0] if args else None
        table = getattr(self, 'table', None)
        if table is None and len(args) > 1 and isinstance(args[1], int):
            table = None
        if table is not None and getattr(table, "_updating_table", False):
            return
        return fn(*args, **kwargs)
    return wrapper
# ---- end added helpers ----

# teaching_physics_simulator_enhanced.py
# Erweiterte Version mit mehreren Koordinatensystemen und verbesserter Visualisierung
# Start: streamlit run teaching_physics_simulator_enhanced.py

import math, time, io, csv
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import json

# Optik-Modul importieren
try:
    import sys
    import os
    # Füge aktuelles Verzeichnis zum Pfad hinzu
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from optics_module import (
        OpticalSystem, Lens, Mirror, Screen, Aperture, LightSource,
        LightRay, OpticalElement, OPTICS_PRESETS, plot_optical_system
    )
    HAS_OPTICS = True
except Exception as e:
    HAS_OPTICS = False
    print(f"Optik-Modul konnte nicht geladen werden: {e}")

# Optional libraries
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ============================================================
# ÜBERSETZUNGEN / TRANSLATIONS
# ============================================================

TRANSLATIONS = {
    'de': {
        # Hauptmenü
        'title': '🔬 Physics Teaching Simulator - Erweiterte Version',
        'subtitle': 'Interaktive Simulation für Gravitation, Elektrodynamik, Stöße, Verbindungen und Optik',
        'language': 'Sprache',
        
        # Sidebar
        'configuration': '⚙️ Konfiguration',
        'presets': 'Voreinstellungen',
        'choose_preset': 'Wähle ein Preset',
        'none': '(Keine)',
        'load_preset': '📥 Preset laden',
        'physics_parameters': 'Physik-Parameter',
        'restitution_coeff': 'Restitutionskoeffizient (Stoß)',
        'restitution_help': '1.0 = elastisch, 0.0 = vollständig inelastisch',
        'air_resistance': 'Luftwiderstand',
        'quadratic': 'Quadratisch',
        'relativistic_correction': 'Relativistische Korrektur',
        'relativistic_help': 'Berücksichtigt relativistische Effekte',
        'magnetic_field': 'Magnetfeld',
        'b_field_input': 'B-Feld (Bx,By,Bz) [T]',
        'visualization': 'Visualisierung',
        'coordinate_systems': 'Koordinatensysteme',
        'coord_help': 'Wähle mehrere Darstellungen',
        'show_conservation': 'Erhaltungsgrößen anzeigen',
        'show_collision_analysis': 'Stoßanalyse anzeigen',
        'show_electrostatic': 'Elektrostatisches Potential',
        
        # Tabs
        'tab_object_editor': '📊 Objekt-Editor',
        'tab_simulation': '▶️ Simulation',
        'tab_optics': '🔬 Optik',
        'tab_export': '💾 Export',
        
        # Objekt-Editor
        'object_configuration': 'Objekt-Konfiguration',
        'num_objects': 'Anzahl Objekte',
        'edit_parameters': '**Bearbeite die Parameter direkt in der Tabelle:**',
        'connections_between': 'Verbindungen zwischen Objekten',
        'connections_format': 'Format: `i-j:typ:stärke`',
        'connections': 'Verbindungen',
        'connections_example': 'Beispiel: 0-1:elastic:10.0',
        'save_own_preset': '💾 Eigenes Preset speichern',
        'preset_name': 'Preset-Name',
        'preset_example': 'z.B. Mein Experiment',
        'save': '💾 Speichern',
        'saved_presets': '📚 Gespeicherte eigene Presets',
        'import_preset': '📥 Preset importieren',
        'upload_json': 'JSON-Datei hochladen',
        'upload_help': 'Lade ein exportiertes Preset',
        
        # Simulation
        'start_simulation': 'Simulation starten',
        'run_simulation': '▶️ Simulation starten',
        'reset': '🗑️ Zurücksetzen',
        
        # Optik
        'optics_header': '🔬 Optik-Simulation',
        'optics_unavailable': 'Optik-Modul nicht verfügbar',
        'optics_install': 'Installiere optics_module.py im gleichen Verzeichnis',
        'optical_setup': 'Optisches System',
        'select_preset': 'Preset auswählen',
        'single_lens': 'Einzelne Linse',
        'two_lens': 'Zwei-Linsen-System',
        'telescope': 'Teleskop',
        'microscope': 'Mikroskop',
        'light_source_config': 'Lichtquellen-Konfiguration',
        'source_type': 'Quellentyp',
        'point_source': 'Punktquelle',
        'parallel_beam': 'Paralleles Bündel',
        'num_rays': 'Anzahl Strahlen',
        'source_position': 'Quellposition x (m)',
        'source_height': 'Quellhöhe y (m)',
        'run_optics': '🔬 Strahlen verfolgen',
        'show_focal': 'Brennpunkte anzeigen',
        'show_construction': 'Konstruktionsstrahlen',
        'calculations': '📐 Berechnungen',
        'lens_equation': 'Linsengleichung',
        'focal_length': 'Brennweite',
        'optical_power': 'Brechkraft',
        'object_distance': 'Gegenstandsweite g (m) für',
        'image_distance': 'Bildweite',
        'magnification': 'Vergrößerung',
        'image_position': 'Bildposition',
        'image_type': 'Bildtyp',
        'real_inverted': 'Reell, umgekehrt',
        'virtual_upright': 'Virtuell, aufrecht',
        'statistics': '📊 Statistik',
        'elements': 'Elemente',
        'light_sources': 'Lichtquellen',
        'rays': 'Strahlen',
        
        # Export
        'data_export': 'Datenexport',
        'export_csv': 'CSV Export',
        'export_json': 'JSON Export (Preset)',
        'generate_csv': '📄 CSV generieren',
        'download_csv': '⬇️ CSV herunterladen',
        'generate_json': '💾 JSON generieren',
        'download_json': '⬇️ JSON herunterladen',
        'data_summary': 'Zusammenfassung',
        'data_points': 'Datenpunkte',
        'time_span': 'Zeitspanne',
        'objects': 'Objekte',
        
        # Meldungen
        'success': '✅',
        'error': '❌ Fehler',
        'warning': '⚠️',
        'info': 'ℹ️',
        'enter_name': 'Bitte geben Sie einen Namen ein!',
        'preset_exists': 'Preset existiert bereits!',
        'preset_saved': 'Preset gespeichert!',
        'preset_loaded': 'geladen',
        'preset_imported': 'Preset importiert!',
        'export_prompt': 'Exportiere',
        'delete_prompt': 'Lösche',
        'csv_ready': 'CSV bereit zum Download',
        'json_ready': 'JSON bereit zum Download',
        'run_simulation_first': 'Führe zuerst eine Simulation durch',
        'no_data': 'Noch keine Daten. Führe eine Simulation aus.',
        'simulation_running': 'Simulation läuft...',
        'simulation_complete': 'Simulation abgeschlossen!',
        'rays_traced': 'Strahlen verfolgt',
        'data_editor_unavailable': '⚠️ Datentabellen-Editor nicht verfügbar.',
    },
    
    'en': {
        # Main menu
        'title': '🔬 Physics Teaching Simulator - Enhanced Version',
        'subtitle': 'Interactive simulation for gravity, electrodynamics, collisions, connections and optics',
        'language': 'Language',
        
        # Sidebar
        'configuration': '⚙️ Configuration',
        'presets': 'Presets',
        'choose_preset': 'Choose a preset',
        'none': '(None)',
        'load_preset': '📥 Load preset',
        'physics_parameters': 'Physics Parameters',
        'restitution_coeff': 'Restitution coefficient (collision)',
        'restitution_help': '1.0 = elastic, 0.0 = completely inelastic',
        'air_resistance': 'Air resistance',
        'quadratic': 'Quadratic',
        'relativistic_correction': 'Relativistic correction',
        'relativistic_help': 'Takes relativistic effects into account',
        'magnetic_field': 'Magnetic Field',
        'b_field_input': 'B-field (Bx,By,Bz) [T]',
        'visualization': 'Visualization',
        'coordinate_systems': 'Coordinate systems',
        'coord_help': 'Select multiple views',
        'show_conservation': 'Show conservation quantities',
        'show_collision_analysis': 'Show collision analysis',
        'show_electrostatic': 'Electrostatic potential',
        
        # Tabs
        'tab_object_editor': '📊 Object Editor',
        'tab_simulation': '▶️ Simulation',
        'tab_optics': '🔬 Optics',
        'tab_export': '💾 Export',
        
        # Object Editor
        'object_configuration': 'Object Configuration',
        'num_objects': 'Number of objects',
        'edit_parameters': '**Edit the parameters directly in the table:**',
        'connections_between': 'Connections between objects',
        'connections_format': 'Format: `i-j:type:strength`',
        'connections': 'Connections',
        'connections_example': 'Example: 0-1:elastic:10.0',
        'save_own_preset': '💾 Save custom preset',
        'preset_name': 'Preset name',
        'preset_example': 'e.g. My Experiment',
        'save': '💾 Save',
        'saved_presets': '📚 Saved custom presets',
        'import_preset': '📥 Import preset',
        'upload_json': 'Upload JSON file',
        'upload_help': 'Load an exported preset',
        
        # Simulation
        'start_simulation': 'Start simulation',
        'run_simulation': '▶️ Run simulation',
        'reset': '🗑️ Reset',
        
        # Optics
        'optics_header': '🔬 Optics Simulation',
        'optics_unavailable': 'Optics module unavailable',
        'optics_install': 'Install optics_module.py in the same directory',
        'optical_setup': 'Optical System',
        'select_preset': 'Select preset',
        'single_lens': 'Single lens',
        'two_lens': 'Two-lens system',
        'telescope': 'Telescope',
        'microscope': 'Microscope',
        'light_source_config': 'Light Source Configuration',
        'source_type': 'Source type',
        'point_source': 'Point source',
        'parallel_beam': 'Parallel beam',
        'num_rays': 'Number of rays',
        'source_position': 'Source position x (m)',
        'source_height': 'Source height y (m)',
        'run_optics': '🔬 Trace rays',
        'show_focal': 'Show focal points',
        'show_construction': 'Construction rays',
        'calculations': '📐 Calculations',
        'lens_equation': 'Lens equation',
        'focal_length': 'Focal length',
        'optical_power': 'Optical power',
        'object_distance': 'Object distance g (m) for',
        'image_distance': 'Image distance',
        'magnification': 'Magnification',
        'image_position': 'Image position',
        'image_type': 'Image type',
        'real_inverted': 'Real, inverted',
        'virtual_upright': 'Virtual, upright',
        'statistics': '📊 Statistics',
        'elements': 'Elements',
        'light_sources': 'Light sources',
        'rays': 'Rays',
        
        # Export
        'data_export': 'Data Export',
        'export_csv': 'CSV Export',
        'export_json': 'JSON Export (Preset)',
        'generate_csv': '📄 Generate CSV',
        'download_csv': '⬇️ Download CSV',
        'generate_json': '💾 Generate JSON',
        'download_json': '⬇️ Download JSON',
        'data_summary': 'Summary',
        'data_points': 'Data points',
        'time_span': 'Time span',
        'objects': 'Objects',
        
        # Messages
        'success': '✅',
        'error': '❌ Error',
        'warning': '⚠️',
        'info': 'ℹ️',
        'enter_name': 'Please enter a name!',
        'preset_exists': 'Preset already exists!',
        'preset_saved': 'Preset saved!',
        'preset_loaded': 'loaded',
        'preset_imported': 'Preset imported!',
        'export_prompt': 'Export',
        'delete_prompt': 'Delete',
        'csv_ready': 'CSV ready for download',
        'json_ready': 'JSON ready for download',
        'run_simulation_first': 'Run a simulation first',
        'no_data': 'No data yet. Run a simulation.',
        'simulation_running': 'Simulation running...',
        'simulation_complete': 'Simulation complete!',
        'rays_traced': 'rays traced',
        'data_editor_unavailable': '⚠️ Data table editor not available.',
    }
}

def get_translation(key: str, lang: str = 'de') -> str:
    """
    Gibt die Übersetzung für einen Schlüssel zurück.
    Fallback zu Deutsch wenn Übersetzung fehlt.
    """
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][key]
    elif key in TRANSLATIONS['de']:
        return TRANSLATIONS['de'][key]
    else:
        return key


# Physikalische Konstanten (SI)
G = 6.67430e-11
k_e = 8.9875517923e9
c = 299792458.0
softening = 1e-9

# Convenience units
AU = 1.495978707e11
KM = 1000.0
M_sun = 1.98847e30
DAY = 86400.0

@dataclass
class Body:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    mass: float
    charge: float
    t0: float
    dt: float
    t_end: float
    color: str = None

@dataclass
class Connection:
    i: int
    j: int
    typ: str
    strength: float
    rest_length: float

@dataclass
class CollisionEvent:
    time: float
    body1: int
    body2: int
    momentum_before: float
    momentum_after: float
    energy_before: float
    energy_after: float

class Simulator:
    def __init__(self, bodies: List[Body], connections: List[Connection]=None,
                 restitution: float=1.0, drag_coeff: float=0.0, drag_quadratic: bool=False,
                 B_field: Optional[Callable[[np.ndarray, float], np.ndarray]]=None,
                 adaptive: bool=False, min_dt: float=1e-6, max_dt: float=0.01,
                 relativistic: bool=False, pinned: Optional[List[int]]=None):
        self.bodies = bodies
        self.N = len(bodies)
        self.connections = connections if connections is not None else []
        self.restitution = restitution
        self.drag_coeff = drag_coeff
        self.drag_quadratic = drag_quadratic
        self.B_field = B_field
        self.adaptive = adaptive
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.relativistic = relativistic
        self.pinned = pinned if pinned is not None else []
        masses = np.array([max(b.mass, 1e-12) for b in self.bodies]) if self.bodies else np.array([])
        self.radii = (masses ** (1/3.0)) * 0.1 if masses.size>0 else np.zeros((self.N,))
        self.pairs = [(i,j) for i in range(self.N) for j in range(i+1,self.N)]
        self._initial_positions = np.array([b.pos.copy() for b in self.bodies]) if self.bodies else np.zeros((0,3))
        self._initial_velocities = np.array([b.vel.copy() for b in self.bodies]) if self.bodies else np.zeros((0,3))
        self.collision_events = []

    def compute_pair_forces(self, positions, velocities, t):
        forces = np.zeros((self.N,3))
        for i,j in self.pairs:
            r_vec = positions[j] - positions[i]
            dist2 = np.dot(r_vec,r_vec) + softening**2
            dist = math.sqrt(dist2)
            if dist < 1e-12:
                continue
            r_hat = r_vec / dist
            m1 = max(self.bodies[i].mass, 1e-12)
            m2 = max(self.bodies[j].mass, 1e-12)
            Fg = G * m1 * m2 / dist2
            q1 = self.bodies[i].charge
            q2 = self.bodies[j].charge
            Fe = k_e * q1 * q2 / dist2
            F_on_i = (Fg + Fe) * r_hat
            forces[i] += F_on_i
            forces[j] -= F_on_i
        return forces

    def connection_forces(self, positions):
        forces = np.zeros((self.N,3))
        for conn in self.connections:
            i,j = conn.i, conn.j
            if i<0 or j<0 or i>=self.N or j>=self.N:
                continue
            r_vec = positions[j] - positions[i]
            dist = np.linalg.norm(r_vec) + 1e-12
            r_hat = r_vec / dist
            if conn.typ == 'elastic':
                k = conn.strength
                Fmag = -k * (dist - conn.rest_length)
                forces[i] += Fmag * r_hat
                forces[j] -= Fmag * r_hat
            elif conn.typ == 'rigid':
                k = conn.strength if conn.strength>0 else 1e8
                Fmag = -k * (dist - conn.rest_length)
                forces[i] += Fmag * r_hat
                forces[j] -= Fmag * r_hat
        return forces

    def drag_force(self, vel):
        if self.drag_coeff == 0.0:
            return np.zeros(3)
        if self.drag_quadratic:
            return -self.drag_coeff * np.linalg.norm(vel) * vel
        else:
            return -self.drag_coeff * vel

    def lorentz_force(self, i, pos, vel, t):
        q = self.bodies[i].charge
        if q == 0.0 or self.B_field is None:
            return np.zeros(3)
        B = self.B_field(pos[i], t)
        return q * np.cross(vel[i], B)

    def compute_forces(self, positions, velocities, t):
        forces = self.compute_pair_forces(positions, velocities, t)
        forces += self.connection_forces(positions)
        for i in range(self.N):
            forces[i] += self.drag_force(velocities[i])
            if self.bodies[i].charge != 0.0 and self.B_field is not None:
                forces[i] += self.lorentz_force(i, positions, velocities, t)
        return forces

    def acceleration(self, i, F, v):
        m = max(self.bodies[i].mass, 1e-12)
        if not self.relativistic:
            return F / m
        v2 = np.dot(v,v)
        beta2 = v2 / c**2
        if beta2 >= 0.999999:
            beta2 = 0.999999
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return F / (gamma * m)
        v_hat = v / v_norm
        F_par = np.dot(F, v_hat) * v_hat
        F_perp = F - F_par
        a = F_perp / (gamma * m) + F_par / (gamma**3 * m)
        return a

    def step_velocity_verlet(self, positions, velocities, t, dt):
        if self.N == 0:
            return positions, velocities
        forces = self.compute_forces(positions, velocities, t)
        acc = np.zeros_like(velocities)
        for i in range(self.N):
            acc[i] = self.acceleration(i, forces[i], velocities[i])
        v_half = velocities + 0.5 * dt * acc
        new_pos = positions + dt * v_half
        forces_new = self.compute_forces(new_pos, v_half, t + dt)
        acc_new = np.zeros_like(velocities)
        for i in range(self.N):
            acc_new[i] = self.acceleration(i, forces_new[i], v_half[i])
        new_vel = v_half + 0.5 * dt * acc_new
        return new_pos, new_vel

    def resolve_collisions(self, positions, velocities, t):
        for i,j in self.pairs:
            r_vec = positions[j] - positions[i]
            dist = np.linalg.norm(r_vec) + 1e-12
            overlap = self.radii[i] + self.radii[j] - dist
            if overlap > 0:
                m1 = max(self.bodies[i].mass, 1e-12)
                m2 = max(self.bodies[j].mass, 1e-12)
                
                v1_before = velocities[i].copy()
                v2_before = velocities[j].copy()
                p_before = m1 * np.linalg.norm(v1_before) + m2 * np.linalg.norm(v2_before)
                E_before = 0.5 * m1 * np.dot(v1_before, v1_before) + 0.5 * m2 * np.dot(v2_before, v2_before)
                
                total = m1 + m2
                correction = (overlap + 1e-9)*(r_vec/dist)
                positions[i] -= correction*(m2/total)
                positions[j] += correction*(m1/total)
                v_rel = velocities[j] - velocities[i]
                n_hat = r_vec/dist
                v_rel_n = np.dot(v_rel, n_hat)
                if v_rel_n < 0:
                    e = self.restitution
                    J = -(1+e)*v_rel_n / (1/m1 + 1/m2)
                    velocities[i] -= (J/m1)*n_hat
                    velocities[j] += (J/m2)*n_hat
                    
                    v1_after = velocities[i].copy()
                    v2_after = velocities[j].copy()
                    p_after = m1 * np.linalg.norm(v1_after) + m2 * np.linalg.norm(v2_after)
                    E_after = 0.5 * m1 * np.dot(v1_after, v1_after) + 0.5 * m2 * np.dot(v2_after, v2_after)
                    
                    self.collision_events.append(CollisionEvent(
                        time=t, body1=i, body2=j,
                        momentum_before=p_before, momentum_after=p_after,
                        energy_before=E_before, energy_after=E_after
                    ))
        return positions, velocities

    def enforce_rigid_constraints(self, positions, velocities, iterations=5):
        for _ in range(iterations):
            for conn in self.connections:
                if conn.typ != 'rigid': continue
                i,j = conn.i, conn.j
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r) + 1e-12
                diff = dist - conn.rest_length
                if abs(diff) > 1e-9:
                    m1 = max(self.bodies[i].mass, 1e-12)
                    m2 = max(self.bodies[j].mass, 1e-12)
                    total = m1 + m2
                    corr = (diff)*(r/dist)
                    positions[i] += corr*(m2/total)
                    positions[j] -= corr*(m1/total)
        return positions, velocities

    def kinetic_energy(self, velocities):
        KE = 0.0
        for i in range(self.N):
            m = max(self.bodies[i].mass, 1e-12)
            v2 = np.dot(velocities[i], velocities[i])
            if not self.relativistic:
                KE += 0.5*m*v2
            else:
                gamma = 1.0/math.sqrt(1.0 - min(v2/c**2, 0.999999))
                KE += (gamma - 1.0)*m*c**2
        return KE

    def potential_energy(self, positions):
        PE = 0.0
        for i,j in self.pairs:
            r_vec = positions[j] - positions[i]
            dist = np.linalg.norm(r_vec) + 1e-12
            m1 = max(self.bodies[i].mass, 1e-12)
            m2 = max(self.bodies[j].mass, 1e-12)
            PE -= G*m1*m2/dist
            q1 = self.bodies[i].charge
            q2 = self.bodies[j].charge
            PE += k_e*q1*q2/dist
        return PE

    def total_momentum(self, velocities):
        p = np.zeros(3)
        for i in range(self.N):
            p += self.bodies[i].mass * velocities[i]
        return p

    def angular_momentum(self, positions, velocities):
        L = np.zeros(3)
        for i in range(self.N):
            L += self.bodies[i].mass * np.cross(positions[i], velocities[i])
        return L

    def run(self, record_every:int=1, max_steps:int=800000):
        if self.N == 0:
            return {'times': np.array([]), 'positions': np.zeros((0,0,3)), 
                    'velocities': np.zeros((0,0,3)), 'KE': np.array([]), 
                    'PE': np.array([]), 'momentum': np.zeros((0,3)),
                    'angular_momentum': np.zeros((0,3))}
        
        t_start = min(b.t0 for b in self.bodies)
        t_end = max(b.t_end for b in self.bodies)
        dt_candidates = [b.dt for b in self.bodies if b.dt>0]
        dt = min(dt_candidates) if dt_candidates else self.max_dt
        dt = max(self.min_dt, min(dt, self.max_dt))

        positions = np.array([b.pos.copy() for b in self.bodies])
        velocities = np.array([b.vel.copy() for b in self.bodies])
        
        times = []
        pos_hist, vel_hist = [], []
        KE_hist, PE_hist = [], []
        momentum_hist, angular_momentum_hist = [], []
        
        t = t_start
        step = 0
        
        while t <= t_end + 1e-12:
            times.append(t)
            pos_hist.append(positions.copy())
            vel_hist.append(velocities.copy())
            KE_hist.append(self.kinetic_energy(velocities))
            PE_hist.append(self.potential_energy(positions))
            momentum_hist.append(self.total_momentum(velocities))
            angular_momentum_hist.append(self.angular_momentum(positions, velocities))

            new_pos, new_vel = self.step_velocity_verlet(positions, velocities, t, dt)
            new_pos, new_vel = self.resolve_collisions(new_pos, new_vel, t)
            new_pos, new_vel = self.enforce_rigid_constraints(new_pos, new_vel)

            positions = new_pos
            velocities = new_vel

            if self.pinned:
                for idx in self.pinned:
                    if 0 <= idx < self.N:
                        positions[idx] = self._initial_positions[idx].copy()
                        velocities[idx] = self._initial_velocities[idx].copy()

            t += dt
            step += 1
            if step > max_steps:
                break

        return {
            'times': np.array(times), 
            'positions': np.array(pos_hist), 
            'velocities': np.array(vel_hist), 
            'KE': np.array(KE_hist), 
            'PE': np.array(PE_hist),
            'momentum': np.array(momentum_hist),
            'angular_momentum': np.array(angular_momentum_hist),
            'collision_events': self.collision_events
        }

# ====================== PRESETS ======================

def scenario_charged_pair(high_charge=1e-6, v=20.0):
    a = Body('A', pos=np.array([-0.5,0.0,0.0]), vel=np.array([v,0.0,0.0]),
             mass=1.0, charge=high_charge, t0=0.0, dt=0.001, t_end=2.0, color='red')
    b = Body('B', pos=np.array([0.5,0.0,0.0]), vel=np.array([-v,0.0,0.0]),
             mass=1.0, charge=-high_charge, t0=0.0, dt=0.001, t_end=2.0, color='blue')
    return [a,b], [], 'Geladenes Paar (±{} C)'.format(high_charge)

def scenario_three_charges_demo(scale_uC=1.0):
    q1 = +3.0e-6 * scale_uC
    q2 = -1.0e-6 * scale_uC
    q3 = -3.0e-6 * scale_uC
    a = Body('Q1', pos=np.array([-1.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, 
             charge=q1, t0=0.0, dt=0.001, t_end=2.0, color='red')
    b = Body('Q2', pos=np.array([0.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, 
             charge=q2, t0=0.0, dt=0.001, t_end=2.0, color='blue')
    c = Body('Q3', pos=np.array([1.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, 
             charge=q3, t0=0.0, dt=0.001, t_end=2.0, color='red')
    return [a,b,c], [], 'Drei Ladungen (+3µC, -1µC, -3µC)'

def scenario_elastic_collision():
    a = Body('A', pos=np.array([-1.0,0.0,0.0]), vel=np.array([2.0,0.0,0.0]),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, color='red')
    b = Body('B', pos=np.array([1.0,0.0,0.0]), vel=np.array([-2.0,0.0,0.0]),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, color='blue')
    return [a,b], [], 'Elastischer Stoß (e=1.0)'

def scenario_inelastic_collision():
    a = Body('A', pos=np.array([-1.0,0.0,0.0]), vel=np.array([2.0,0.0,0.0]),
             mass=2.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, color='red')
    b = Body('B', pos=np.array([1.0,0.0,0.0]), vel=np.array([-1.0,0.0,0.0]),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=3.0, color='blue')
    return [a,b], [], 'Inelastischer Stoß (e<1.0)'

def scenario_spring_system():
    a = Body('A', pos=np.array([-1.0,0.0,0.0]), vel=np.zeros(3),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=5.0, color='red')
    b = Body('B', pos=np.array([1.0,0.0,0.0]), vel=np.zeros(3),
             mass=1.0, charge=0.0, t0=0.0, dt=0.001, t_end=5.0, color='blue')
    conn = [Connection(0, 1, 'elastic', 10.0, 2.0)]
    return [a,b], conn, 'Federsystem'

def scenario_planetary_scaled(scale_mass=1e14, scale_distance=1.0, add_second=True, 
                              t_end=50.0, dt=0.001, pin_sun=False):
    M = float(scale_mass)
    sun = Body('Sun', pos=np.array([0.0,0.0,0.0]), vel=np.array([0.0,0.0,0.0]),
               mass=M, charge=0.0, t0=0.0, dt=float(dt), t_end=float(t_end), color='yellow')
    r1 = 1.0 * float(scale_distance)
    v1 = math.sqrt(G * M / r1)
    p1 = Body('P1', pos=np.array([r1,0.0,0.0]), vel=np.array([0.0,v1,0.0]),
              mass=1.0, charge=0.0, t0=0.0, dt=float(dt), t_end=float(t_end), color='blue')
    bodies = [sun, p1]
    if add_second:
        r2 = 1.6 * float(scale_distance)
        v2 = math.sqrt(G * M / r2)
        p2 = Body('P2', pos=np.array([r2,0.0,0.0]), vel=np.array([0.0,v2,0.0]),
                  mass=0.5, charge=0.0, t0=0.0, dt=float(dt), t_end=float(t_end), color='green')
        bodies.append(p2)
    if not pin_sun:
        total_p = np.zeros(3)
        for b in bodies:
            total_p += b.mass * b.vel
        sun.vel = - total_p / sun.mass
    return bodies, [], f"Planetensystem (M={scale_mass})"

PRESETS = {
    'Geladenes Paar': scenario_charged_pair,
    'Drei Ladungen': scenario_three_charges_demo,
    'Elastischer Stoß': scenario_elastic_collision,
    'Inelastischer Stoß': scenario_inelastic_collision,
    'Federsystem': scenario_spring_system,
    'Planetensystem': scenario_planetary_scaled,
}

# ====================== VISUALISIERUNG ======================

def plot_trajectory_3d(bodies, data, coordinate_system='cartesian', key_suffix=""):
    if not HAS_PLOTLY:
        return
    
    positions = data['positions']
    velocities = data['velocities']
    times = data['times']
    
    fig = go.Figure()
    
    if coordinate_system == 'cartesian':
        for i, b in enumerate(bodies):
            traj = positions[:, i, :]
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines', name=b.name,
                line=dict(color=b.color if b.color else None, width=2)
            ))
            fig.add_trace(go.Scatter3d(
                x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                mode='markers', name=f'{b.name} Start',
                marker=dict(size=5, color='green')
            ))
        fig.update_layout(
            scene=dict(xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)'),
            title='Trajektorien (Kartesische Koordinaten)'
        )
    
    elif coordinate_system == 'momentum':
        for i, b in enumerate(bodies):
            momentum = b.mass * velocities[:, i, :]
            fig.add_trace(go.Scatter3d(
                x=momentum[:, 0], y=momentum[:, 1], z=momentum[:, 2],
                mode='lines+markers', name=b.name,
                line=dict(color=b.color if b.color else None, width=2),
                marker=dict(size=2)
            ))
        fig.update_layout(
            scene=dict(xaxis_title='p_x (kg·m/s)', yaxis_title='p_y (kg·m/s)', zaxis_title='p_z (kg·m/s)'),
            title='Trajektorien im Impulsraum'
        )
    
    elif coordinate_system == 'energy':
        KE = data['KE']
        PE = data['PE']
        Total = KE + PE
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=KE, mode='lines', name='Kinetische Energie', 
                                 line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=times, y=PE, mode='lines', name='Potentielle Energie', 
                                 line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=times, y=Total, mode='lines', name='Gesamtenergie', 
                                 line=dict(color='green', width=3)))
        fig.update_layout(
            xaxis_title='Zeit (s)', yaxis_title='Energie (J)',
            title='Energiedarstellung (Energieerhaltung)'
        )
    
    elif coordinate_system == 'com_relative':
        masses = np.array([b.mass for b in bodies])
        total_mass = masses.sum()
        COM = np.tensordot(positions, masses, axes=([1], [0])) / total_mass
        positions_rel = positions - COM[:, None, :]
        
        for i, b in enumerate(bodies):
            traj = positions_rel[:, i, :]
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines', name=b.name,
                line=dict(color=b.color if b.color else None, width=2)
            ))
        fig.update_layout(
            scene=dict(xaxis_title='x (COM)', yaxis_title='y (COM)', zaxis_title='z (COM)'),
            title='Trajektorien (relativ zum Schwerpunkt)'
        )
    
    key = f"plot_{coordinate_system}_{key_suffix}_{int(time.time()*1000)}"
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_conservation_laws(data, key_suffix=""):
    if not HAS_PLOTLY:
        return
    
    times = data['times']
    KE = data['KE']
    PE = data['PE']
    Total = KE + PE
    momentum = data['momentum']
    angular_momentum = data['angular_momentum']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Energie', 'Impuls (Betrag)', 'Drehimpuls (Betrag)', 'Energieabweichung (%)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    fig.add_trace(go.Scatter(x=times, y=KE, name='KE', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=PE, name='PE', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=Total, name='Total', line=dict(color='green', width=3)), row=1, col=1)
    
    p_magnitude = np.linalg.norm(momentum, axis=1)
    fig.add_trace(go.Scatter(x=times, y=p_magnitude, name='|p|', line=dict(color='purple')), row=1, col=2)
    
    L_magnitude = np.linalg.norm(angular_momentum, axis=1)
    fig.add_trace(go.Scatter(x=times, y=L_magnitude, name='|L|', line=dict(color='orange')), row=2, col=1)
    
    if len(Total) > 0 and abs(Total[0]) > 1e-12:
        energy_deviation = 100 * (Total - Total[0]) / abs(Total[0])
        fig.add_trace(go.Scatter(x=times, y=energy_deviation, name='ΔE%', line=dict(color='red')), row=2, col=2)
    
    fig.update_xaxes(title_text="Zeit (s)", row=1, col=1)
    fig.update_xaxes(title_text="Zeit (s)", row=1, col=2)
    fig.update_xaxes(title_text="Zeit (s)", row=2, col=1)
    fig.update_xaxes(title_text="Zeit (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="Energie (J)", row=1, col=1)
    fig.update_yaxes(title_text="Impuls (kg·m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Drehimpuls (kg·m²/s)", row=2, col=1)
    fig.update_yaxes(title_text="Abweichung (%)", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, title_text="Erhaltungsgrößen")
    
    key = f"conservation_{key_suffix}_{int(time.time()*1000)}"
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_collision_analysis(collision_events, key_suffix=""):
    if not HAS_PLOTLY or len(collision_events) == 0:
        return
    
    times = [c.time for c in collision_events]
    momentum_loss = [(c.momentum_before - c.momentum_after) for c in collision_events]
    energy_loss = [(c.energy_before - c.energy_after) for c in collision_events]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Impulsverlust bei Stößen', 'Energieverlust bei Stößen')
    )
    
    fig.add_trace(go.Scatter(x=times, y=momentum_loss, mode='markers+lines',
                            name='Impulsverlust', marker=dict(size=8, color='purple')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=times, y=energy_loss, mode='markers+lines',
                            name='Energieverlust', marker=dict(size=8, color='red')),
                 row=1, col=2)
    
    fig.update_xaxes(title_text="Zeit (s)", row=1, col=1)
    fig.update_xaxes(title_text="Zeit (s)", row=1, col=2)
    fig.update_yaxes(title_text="Δp (kg·m/s)", row=1, col=1)
    fig.update_yaxes(title_text="ΔE (J)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, title_text="Stoßanalyse")
    
    key = f"collision_{key_suffix}_{int(time.time()*1000)}"
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_potential_field(bodies, region=None, grid_n=100):
    if not HAS_MATPLOTLIB:
        return
    
    pos = np.array([b.pos for b in bodies])
    if region is None:
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        span = (maxs - mins)
        pad = max(span[0], span[1]) * 0.6 + 0.5
        cx = 0.5*(mins[0]+maxs[0])
        cy = 0.5*(mins[1]+maxs[1])
        xmin, xmax = cx - pad, cx + pad
        ymin, ymax = cy - pad, cy + pad
    else:
        xmin, xmax, ymin, ymax = region
    
    nx = ny = int(min(200, max(50, grid_n)))
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    V = np.zeros_like(X)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(X)
    eps = 1e-12
    
    for b in bodies:
        q = b.charge
        if abs(q) < 1e-20:
            continue
        bx = b.pos[0]
        by = b.pos[1]
        dx = X - bx
        dy = Y - by
        r = np.sqrt(dx*dx + dy*dy + eps)
        V += k_e * q / r
        Ex += k_e * q * dx / (r**3)
        Ey += k_e * q * dy / (r**3)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        levels = np.linspace(np.percentile(V, 5), np.percentile(V, 95), 14)
        cs = ax.contourf(X, Y, V, levels=levels, cmap='RdBu_r', alpha=0.8)
    except:
        cs = ax.contourf(X, Y, V, cmap='RdBu_r', alpha=0.8)
    
    plt.colorbar(cs, ax=ax, label='Potential V (V)')
    
    strm = ax.streamplot(xs, ys, Ex.T, Ey.T, density=1.2, color='k', 
                        linewidth=0.5, arrowsize=1)
    
    for b in bodies:
        if abs(b.charge) > 1e-20:
            if b.charge > 0:
                ax.scatter(b.pos[0], b.pos[1], color='red', s=80, marker='o', 
                          edgecolors='black', linewidths=2, zorder=5)
                ax.text(b.pos[0], b.pos[1], f" {b.name} (+)", color='darkred', 
                       fontsize=10, fontweight='bold')
            else:
                ax.scatter(b.pos[0], b.pos[1], color='blue', s=80, marker='o', 
                          edgecolors='black', linewidths=2, zorder=5)
                ax.text(b.pos[0], b.pos[1], f" {b.name} (-)", color='darkblue', 
                       fontsize=10, fontweight='bold')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Elektrostatisches Potential & Feldlinien (Ebene z=0)')
    ax.grid(True, alpha=0.3)
    
    return fig

def export_csv(bodies, data):
    times = data['times']
    positions = data['positions']
    n = len(bodies)
    out = io.StringIO()
    w = csv.writer(out)
    header = ['t'] + [f'{bodies[i].name}_x' for i in range(n)] + \
             [f'{bodies[i].name}_y' for i in range(n)] + \
             [f'{bodies[i].name}_z' for i in range(n)]
    w.writerow(header)
    for idx, t in enumerate(times):
        row = [t]
        for i in range(n):
            row.append(positions[idx, i, 0])
        for i in range(n):
            row.append(positions[idx, i, 1])
        for i in range(n):
            row.append(positions[idx, i, 2])
        w.writerow(row)
    return out.getvalue()

def export_preset_json(preset_name, bodies, connections):
    preset_data = {
        'name': preset_name,
        'bodies': [],
        'connections': []
    }
    
    for b in bodies:
        preset_data['bodies'].append({
            'name': b.name,
            'pos': b.pos.tolist(),
            'vel': b.vel.tolist(),
            'mass': b.mass,
            'charge': b.charge,
            't0': b.t0,
            'dt': b.dt,
            't_end': b.t_end,
            'color': b.color
        })
    
    for c in connections:
        preset_data['connections'].append({
            'i': c.i,
            'j': c.j,
            'typ': c.typ,
            'strength': c.strength,
            'rest_length': c.rest_length
        })
    
    return json.dumps(preset_data, indent=2)

def import_preset_json(json_string):
    try:
        preset_data = json.loads(json_string)
        
        bodies = []
        for b_data in preset_data['bodies']:
            bodies.append(Body(
                name=b_data['name'],
                pos=np.array(b_data['pos']),
                vel=np.array(b_data['vel']),
                mass=b_data['mass'],
                charge=b_data['charge'],
                t0=b_data['t0'],
                dt=b_data['dt'],
                t_end=b_data['t_end'],
                color=b_data.get('color', None)
            ))
        
        connections = []
        for c_data in preset_data['connections']:
            connections.append(Connection(
                i=c_data['i'],
                j=c_data['j'],
                typ=c_data['typ'],
                strength=c_data['strength'],
                rest_length=c_data['rest_length']
            ))
        
        return preset_data['name'], bodies, connections
    except Exception as e:
        raise ValueError(f"Fehler beim Importieren: {e}")

# ====================== STREAMLIT UI ======================

if HAS_STREAMLIT:
    def build_body_from_row(r):
        return Body(
            name=str(r.get('name', 'B')),
            pos=np.array([float(r.get('x', 0.0)), float(r.get('y', 0.0)), float(r.get('z', 0.0))]),
            vel=np.array([float(r.get('vx', 0.0)), float(r.get('vy', 0.0)), float(r.get('vz', 0.0))]),
            mass=float(r.get('mass', 1.0)),
            charge=float(r.get('charge', 0.0)),
            t0=float(r.get('t0', 0.0)),
            dt=float(r.get('dt', 0.001)),
            t_end=float(r.get('t_end', 2.0)),
            color=r.get('color', None)
        )

    def main():
        st.set_page_config(layout='wide', page_title="Physics Teaching Simulator")
        
        # ============================================================
        # SPRACHAUSWAHL / LANGUAGE SELECTION
        # ============================================================
        # Muss VOR der ersten Verwendung von t() erfolgen
        with st.sidebar:
            lang = st.selectbox(
                '🌐 Language / Sprache',
                options=['de', 'en'],
                format_func=lambda x: 'Deutsch 🇩🇪' if x == 'de' else 'English 🇬🇧',
                key='language_selector'
            )
            st.markdown("---")
        
        # Übersetzungsfunktion mit gewählter Sprache (im Hauptscope)
        def t(key):
            return get_translation(key, lang)
        
        # Titel und Untertitel
        st.title(t("title"))
        st.markdown(f"*{t('subtitle')}*")
        
        # Sidebar-Inhalt
        with st.sidebar:
            st.header(t("configuration"))
            
            st.subheader(t("presets"))
            preset_choice = st.selectbox(t("choose_preset"), [t("none")] + list(PRESETS.keys()))
            
            if st.button(t("load_preset"), use_container_width=True):
                if preset_choice != t("none"):
                    try:
                        bodies_preset, conns_preset, note = PRESETS[preset_choice]()
                        rows = []
                        for b in bodies_preset:
                            rows.append({
                                'name': b.name, 'x': float(b.pos[0]), 'y': float(b.pos[1]), 
                                'z': float(b.pos[2]), 'vx': float(b.vel[0]), 'vy': float(b.vel[1]), 
                                'vz': float(b.vel[2]), 'mass': float(b.mass), 
                                'charge': float(b.charge), 't0': float(b.t0), 
                                'dt': float(b.dt), 't_end': float(b.t_end),
                                'color': b.color
                            })
                        st.session_state['preset_data'] = pd.DataFrame(rows) if HAS_PANDAS else rows
                        st.session_state['preset_conns'] = conns_preset
                        st.success(f"{t('success')} {note}")
                    except Exception as e:
                        st.error(f"❌ Fehler: {e}")
            
            st.markdown("---")
            
            st.subheader(t("physics_parameters"))
            restitution = st.slider(t("restitution_coeff"), 0.0, 1.0, 1.0, 0.05,
                                   help=t("restitution_help"))
            
            col1, col2 = st.columns(2)
            with col1:
                drag = st.number_input(t("air_resistance"), 0.0, 10.0, 0.0, 0.1, format="%.2f")
            with col2:
                drag_quad = st.checkbox(t("quadratic"), value=False)
            
            relativistic = st.checkbox(t("relativistic_correction"), value=False,
                                      help=t("relativistic_help"))
            
            st.markdown("---")
            
            st.subheader(t("magnetic_field"))
            b_input = st.text_input(t("b_field_input"), "0,0,0")
            try:
                B_vec = np.array([float(x.strip()) for x in b_input.split(',')])
                if B_vec.size != 3:
                    B_vec = np.zeros(3)
            except:
                B_vec = np.zeros(3)
            
            st.markdown("---")
            
            st.subheader(t("visualization"))
            coord_systems = st.multiselect(t("coordinate_systems"),
                ['cartesian', 'momentum', 'energy', 'com_relative'],
                default=['cartesian', 'energy'],
                help="Wähle mehrere Darstellungen"
            )
            
            show_conservation = st.checkbox(t("show_conservation"), value=True)
            show_collisions = st.checkbox(t("show_collision_analysis"), value=False)
            show_potential = st.checkbox(t("show_electrostatic"), value=False)
        
        tabs = st.tabs([t("tab_object_editor"), t("tab_simulation"), t("tab_optics"), t("tab_export")])
        
        with tabs[0]:
            st.header(t("object_configuration"))
            
            n_objects = st.number_input(t("num_objects"), 1, 10, 3, 1)
            
            if 'preset_data' in st.session_state and st.session_state['preset_data'] is not None:
                if HAS_PANDAS:
                    df = st.session_state['preset_data']
                else:
                    df = pd.DataFrame(st.session_state['preset_data']) if HAS_PANDAS else st.session_state['preset_data']
                st.session_state.pop('preset_data')
            else:
                if 'current_data' not in st.session_state:
                    rows = []
                    for i in range(n_objects):
                        rows.append({
                            'name': f'Obj{i}', 'x': 0.0, 'y': 0.0, 'z': 0.0,
                            'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'mass': 1.0,
                            'charge': 0.0, 't0': 0.0, 'dt': 0.001, 't_end': 2.0,
                            'color': None
                        })
                    df = pd.DataFrame(rows) if HAS_PANDAS else rows
                else:
                    df = st.session_state['current_data']
            
            if HAS_PANDAS and hasattr(st, 'data_editor'):
                st.markdown(t("edit_parameters"))
                edited_df = st.data_editor(
                    df,
                    num_rows="fixed",
                    use_container_width=True,
                    column_config={
                        "name": st.column_config.TextColumn("Name", width="small"),
                        "x": st.column_config.NumberColumn("x", format="%.3f"),
                        "y": st.column_config.NumberColumn("y", format="%.3f"),
                        "z": st.column_config.NumberColumn("z", format="%.3f"),
                        "vx": st.column_config.NumberColumn("vx", format="%.3f"),
                        "vy": st.column_config.NumberColumn("vy", format="%.3f"),
                        "vz": st.column_config.NumberColumn("vz", format="%.3f"),
                        "mass": st.column_config.NumberColumn("Masse", format="%.6g"),
                        "charge": st.column_config.NumberColumn("Ladung (C)", format="%.6g"),
                        "dt": st.column_config.NumberColumn("dt", format="%.6g"),
                        "t_end": st.column_config.NumberColumn("t_end", format="%.3f"),
                    }
                )
                st.session_state['current_data'] = edited_df
            else:
                st.warning(t("data_editor_unavailable"))
                edited_df = df
            
            st.markdown("---")
            
            st.subheader(t("connections_between"))
            st.markdown(t("connections_format"))
            
            conn_default = ""
            if 'preset_conns' in st.session_state and st.session_state['preset_conns']:
                conn_lines = []
                for c in st.session_state['preset_conns']:
                    conn_lines.append(f"{c.i}-{c.j}:{c.typ}:{c.strength}")
                conn_default = "\n".join(conn_lines)
                st.session_state.pop('preset_conns')
            
            conns_text = st.text_area(t("connections"),
                value=conn_default,
                height=100,
                placeholder="Beispiel: 0-1:elastic:10.0"
            )
            
            st.markdown("---")
            
            st.subheader(t("save_own_preset"))
            
            col1, col2 = st.columns([3, 1])
            with col1:
                preset_name = st.text_input(t("preset_name"),
                    value="",
                    placeholder=t("preset_example"),
                    key="new_preset_name"
                )
            with col2:
                st.write("")
                st.write("")
                save_preset_btn = st.button(t("save"), use_container_width=True)
            
            if save_preset_btn:
                if not preset_name or preset_name.strip() == "":
                    st.error(f"{t('warning')} {t('enter_name')}")
                elif preset_name in PRESETS:
                    st.error(f"{t('warning')} {t('preset_exists')}")
                else:
                    try:
                        current_bodies = []
                        df_save = st.session_state.get('current_data', edited_df)
                        
                        for i in range(len(df_save)):
                            row = df_save.iloc[i] if HAS_PANDAS else df_save[i]
                            b = build_body_from_row(row)
                            current_bodies.append(b)
                        
                        current_conns = []
                        for line in conns_text.strip().split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                parts = line.split(':')
                                if len(parts) != 3:
                                    continue
                                pair, typ, strength_s = parts
                                i_s, j_s = pair.split('-')
                                i, j = int(i_s), int(j_s)
                                strength = float(strength_s)
                                typ = typ.strip().lower()
                                
                                if 0 <= i < len(current_bodies) and 0 <= j < len(current_bodies):
                                    rest_length = np.linalg.norm(current_bodies[j].pos - current_bodies[i].pos)
                                    current_conns.append(Connection(i, j, typ, strength, rest_length))
                            except:
                                continue
                        
                        def create_custom_preset(bodies_copy, conns_copy, name):
                            def custom_preset():
                                bodies_new = []
                                for b in bodies_copy:
                                    bodies_new.append(Body(
                                        name=b.name, pos=b.pos.copy(), vel=b.vel.copy(),
                                        mass=b.mass, charge=b.charge, t0=b.t0,
                                        dt=b.dt, t_end=b.t_end, color=b.color
                                    ))
                                conns_new = []
                                for c in conns_copy:
                                    conns_new.append(Connection(
                                        i=c.i, j=c.j, typ=c.typ,
                                        strength=c.strength, rest_length=c.rest_length
                                    ))
                                return bodies_new, conns_new, f"Eigenes Preset: {name}"
                            return custom_preset
                        
                        PRESETS[preset_name] = create_custom_preset(current_bodies, current_conns, preset_name)
                        
                        if 'custom_presets' not in st.session_state:
                            st.session_state['custom_presets'] = {}
                        
                        st.session_state['custom_presets'][preset_name] = {
                            'bodies': current_bodies,
                            'connections': current_conns
                        }
                        
                        st.success(f"{t('success')} {t('preset_saved')}")
                        
                    except Exception as e:
                        st.error(f"❌ Fehler: {e}")
            
            if 'custom_presets' in st.session_state and st.session_state['custom_presets']:
                with st.expander(t("saved_presets"), expanded=False):
                    for name in list(st.session_state['custom_presets'].keys()):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{name}**")
                        with col2:
                            preset_data = st.session_state['custom_presets'][name]
                            json_str = export_preset_json(name, preset_data['bodies'], preset_data['connections'])
                            st.download_button(
                                label="💾",
                                data=json_str,
                                file_name=f"preset_{name.replace(' ', '_')}.json",
                                mime="application/json",
                                key=f"export_{name}",
                                help=f"Exportiere '{name}'"
                            )
                        with col3:
                            if st.button("🗑️", key=f"del_{name}", help=f"Lösche '{name}'"):
                                if name in PRESETS:
                                    del PRESETS[name]
                                del st.session_state['custom_presets'][name]
                                st.rerun()
            
            st.markdown("---")
            st.subheader(t("import_preset"))
            uploaded_file = st.file_uploader(t("upload_json"),
                type=['json'],
                help="Lade ein exportiertes Preset"
            )
            
            if uploaded_file is not None:
                try:
                    json_str = uploaded_file.read().decode('utf-8')
                    import_name, import_bodies, import_conns = import_preset_json(json_str)
                    
                    original_name = import_name
                    counter = 1
                    while import_name in PRESETS:
                        import_name = f"{original_name} ({counter})"
                        counter += 1
                    
                    def create_imported_preset(bodies_copy, conns_copy, name):
                        def imported_preset():
                            bodies_new = []
                            for b in bodies_copy:
                                bodies_new.append(Body(
                                    name=b.name, pos=b.pos.copy(), vel=b.vel.copy(),
                                    mass=b.mass, charge=b.charge, t0=b.t0,
                                    dt=b.dt, t_end=b.t_end, color=b.color
                                ))
                            conns_new = []
                            for c in conns_copy:
                                conns_new.append(Connection(
                                    i=c.i, j=c.j, typ=c.typ,
                                    strength=c.strength, rest_length=c.rest_length
                                ))
                            return bodies_new, conns_new, f"Importiert: {name}"
                        return imported_preset
                    
                    PRESETS[import_name] = create_imported_preset(import_bodies, import_conns, import_name)
                    
                    if 'custom_presets' not in st.session_state:
                        st.session_state['custom_presets'] = {}
                    st.session_state['custom_presets'][import_name] = {
                        'bodies': import_bodies,
                        'connections': import_conns
                    }
                    
                    st.success(f"{t('success')} {t('preset_imported')}")
                    
                except Exception as e:
                    st.error(f"❌ Fehler: {e}")
        
        with tabs[1]:
            st.header(t("start_simulation"))
            
            col1, col2 = st.columns([2, 1])
            with col1:
                run_sim = st.button(t("run_simulation"), use_container_width=True, type="primary")
            with col2:
                clear_sim = st.button(t("reset"), use_container_width=True)
            
            if clear_sim:
                if 'last_sim' in st.session_state:
                    st.session_state.pop('last_sim')
                st.info("Simulation zurückgesetzt")
            
            if run_sim:
                try:
                    bodies = []
                    df_work = st.session_state.get('current_data', edited_df)
                    
                    for i in range(len(df_work)):
                        row = df_work.iloc[i] if HAS_PANDAS else df_work[i]
                        bodies.append(build_body_from_row(row))
                    
                    connections = []
                    for line in conns_text.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parts = line.split(':')
                            if len(parts) != 3:
                                continue
                            pair, typ, strength_s = parts
                            i_s, j_s = pair.split('-')
                            i, j = int(i_s), int(j_s)
                            strength = float(strength_s)
                            typ = typ.strip().lower()
                            
                            if 0 <= i < len(bodies) and 0 <= j < len(bodies):
                                rest_length = np.linalg.norm(bodies[j].pos - bodies[i].pos)
                                connections.append(Connection(i, j, typ, strength, rest_length))
                        except:
                            continue
                    
                    def B_field(r, t):
                        return B_vec
                    
                    with st.spinner("🔄 Simulation läuft..."):
                        sim = Simulator(
                            bodies, connections,
                            restitution=restitution,
                            drag_coeff=drag,
                            drag_quadratic=drag_quad,
                            B_field=B_field if np.any(B_vec != 0) else None,
                            relativistic=relativistic
                        )
                        
                        data = sim.run()
                        
                        st.session_state['last_sim'] = {
                            'bodies': bodies,
                            'data': data,
                            'sim': sim
                        }
                    
                    st.success(f"{t('success')} {t('simulation_complete')}")
                    
                except Exception as e:
                    st.error(f"❌ Fehler: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            if 'last_sim' in st.session_state:
                st.markdown("---")
                st.subheader("Simulationsergebnisse")
                
                last_sim = st.session_state['last_sim']
                bodies = last_sim['bodies']
                data = last_sim['data']
                sim = last_sim['sim']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Zeitschritte", len(data['times']))
                with col2:
                    st.metric("Simulationszeit", f"{data['times'][-1]:.3f} s")
                with col3:
                    st.metric("Objekte", len(bodies))
                with col4:
                    st.metric("Stöße", len(sim.collision_events))
                
                st.markdown("---")
                
                for coord_sys in coord_systems:
                    try:
                        plot_trajectory_3d(bodies, data, coord_sys, key_suffix=str(time.time()))
                    except Exception as e:
                        st.error(f"Fehler bei Plot ({coord_sys}): {e}")
                
                if show_conservation:
                    st.markdown("---")
                    st.subheader("Erhaltungsgrößen")
                    try:
                        plot_conservation_laws(data, key_suffix=str(time.time()))
                    except Exception as e:
                        st.error(f"Fehler: {e}")
                
                if show_collisions and len(sim.collision_events) > 0:
                    st.markdown("---")
                    st.subheader("Stoßanalyse")
                    try:
                        plot_collision_analysis(sim.collision_events, key_suffix=str(time.time()))
                        
                        st.markdown("**Stoß-Details:**")
                        collision_data = []
                        for c in sim.collision_events:
                            collision_data.append({
                                'Zeit (s)': f"{c.time:.4f}",
                                'Körper': f"{bodies[c.body1].name} ↔ {bodies[c.body2].name}",
                                'Δp': f"{c.momentum_before - c.momentum_after:.6g}",
                                'ΔE': f"{c.energy_before - c.energy_after:.6g}"
                            })
                        st.table(pd.DataFrame(collision_data) if HAS_PANDAS else collision_data)
                    except Exception as e:
                        st.error(f"Fehler: {e}")
                
                if show_potential:
                    has_charges = any(abs(b.charge) > 1e-20 for b in bodies)
                    if has_charges:
                        st.markdown("---")
                        st.subheader("Elektrostatisches Potential")
                        try:
                            fig_pot = plot_potential_field(bodies)
                            st.pyplot(fig_pot)
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                    else:
                        st.info("ℹ️ Keine geladenen Objekte")
        
        with tabs[2]:
            st.header("🔬 Optik-Modul: Strahlenoptik")
            
            if not HAS_OPTICS:
                st.error("❌ Optik-Modul nicht verfügbar. Stelle sicher, dass optics_module.py im gleichen Verzeichnis liegt.")
                st.info("Die Datei optics_module.py wurde gespeichert in /Users/drhenrich/")
            else:
                st.markdown("*Geometrische Optik: Linsen, Spiegel, Strahlengang*")
                
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.subheader("Konfiguration")
                    
                    optics_preset = st.selectbox(
                        "Optik-Preset",
                        ["(Manuell)"] + list(OPTICS_PRESETS.keys()),
                        key="optics_preset"
                    )
                    
                    if st.button("📥 Optik-Preset laden", key="load_optics_preset"):
                        if optics_preset != "(Manuell)":
                            try:
                                elements, sources, description = OPTICS_PRESETS[optics_preset]()
                                st.session_state['optics_elements'] = elements
                                st.session_state['optics_sources'] = sources
                                st.success(f"✅ {description}")
                            except Exception as e:
                                st.error(f"❌ Fehler: {e}")
                    
                    st.markdown("---")
                    
                    st.subheader("Optische Elemente")
                    
                    if 'optics_elements' not in st.session_state:
                        st.session_state['optics_elements'] = []
                    if 'optics_sources' not in st.session_state:
                        st.session_state['optics_sources'] = []
                    
                    element_type = st.selectbox(
                        "Element-Typ",
                        ["Linse", "Spiegel", "Schirm", "Blende"],
                        key="new_element_type"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        elem_pos = st.number_input("Position (m)", value=0.0, format="%.3f", key="elem_pos")
                        elem_name = st.text_input("Name", value=element_type, key="elem_name")
                    
                    with col2:
                        if element_type == "Linse":
                            elem_f = st.number_input("Brennweite (m)", value=0.2, format="%.3f", key="elem_f")
                            elem_d = st.number_input("Durchmesser (m)", value=0.1, format="%.3f", key="elem_d")
                        elif element_type == "Spiegel":
                            elem_angle = st.number_input("Winkel (°)", value=0.0, format="%.1f", key="elem_angle")
                            elem_h = st.number_input("Höhe (m)", value=0.1, format="%.3f", key="elem_h")
                        elif element_type == "Schirm":
                            elem_h = st.number_input("Höhe (m)", value=0.2, format="%.3f", key="elem_screen_h")
                        elif element_type == "Blende":
                            elem_d = st.number_input("Öffnung (m)", value=0.05, format="%.3f", key="elem_ap_d")
                    
                    if st.button("➕ Element hinzufügen", use_container_width=True):
                        try:
                            if element_type == "Linse":
                                new_elem = Lens(position=elem_pos, focal_length=elem_f, 
                                              diameter=elem_d, name=elem_name)
                            elif element_type == "Spiegel":
                                new_elem = Mirror(position=elem_pos, angle=elem_angle, 
                                                height=elem_h, name=elem_name)
                            elif element_type == "Schirm":
                                new_elem = Screen(position=elem_pos, height=elem_h, name=elem_name)
                            elif element_type == "Blende":
                                new_elem = Aperture(position=elem_pos, diameter=elem_d, name=elem_name)
                            
                            st.session_state['optics_elements'].append(new_elem)
                            st.success(f"✅ {elem_name} hinzugefügt")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fehler: {e}")
                    
                    st.markdown("---")
                    
                    st.subheader("Lichtquelle")
                    
                    source_type = st.selectbox(
                        "Quelltyp",
                        ["parallel", "point", "object"],
                        key="source_type",
                        format_func=lambda x: {"parallel": "Parallel", "point": "Punkt", "object": "Objekt"}[x]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        source_x = st.number_input("X-Position (m)", value=-0.5, format="%.3f", key="source_x")
                        source_y = st.number_input("Y-Position (m)", value=0.0, format="%.3f", key="source_y")
                    with col2:
                        source_rays = st.number_input("Anzahl Strahlen", value=5, min_value=1, max_value=20, key="source_rays")
                        source_wl = st.number_input("Wellenlänge (nm)", value=550, min_value=380, max_value=780, key="source_wl")
                    
                    if st.button("➕ Lichtquelle hinzufügen", use_container_width=True):
                        new_source = LightSource(
                            position=np.array([source_x, source_y]),
                            source_type=source_type,
                            num_rays=source_rays,
                            wavelength=source_wl * 1e-9
                        )
                        st.session_state['optics_sources'].append(new_source)
                        st.success("✅ Lichtquelle hinzugefügt")
                        st.rerun()
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Alles löschen", use_container_width=True):
                            st.session_state['optics_elements'] = []
                            st.session_state['optics_sources'] = []
                            st.rerun()
                    with col2:
                        trace_btn = st.button("▶️ Strahlengang berechnen", use_container_width=True, type="primary")
                
                with col_right:
                    st.subheader(t("optical_setup"))
                    
                    if st.session_state['optics_elements']:
                        st.markdown("**Elemente:**")
                        for i, elem in enumerate(st.session_state['optics_elements']):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                if isinstance(elem, Lens):
                                    st.write(f"🔍 {elem.name}: f={elem.focal_length:.3f}m @ x={elem.position:.3f}m")
                                elif isinstance(elem, Screen):
                                    st.write(f"📺 {elem.name} @ x={elem.position:.3f}m")
                                elif isinstance(elem, Mirror):
                                    st.write(f"🪞 {elem.name} @ x={elem.position:.3f}m")
                                elif isinstance(elem, Aperture):
                                    st.write(f"⭕ {elem.name} @ x={elem.position:.3f}m")
                            with col2:
                                elem.active = st.checkbox("", value=elem.active, key=f"active_opt_{i}", label_visibility="collapsed")
                            with col3:
                                if st.button("🗑️", key=f"del_opt_{i}"):
                                    st.session_state['optics_elements'].pop(i)
                                    st.rerun()
                    else:
                        st.info("ℹ️ Keine Elemente. Fügen Sie Linsen, Spiegel oder Schirme hinzu.")
                    
                    if st.session_state['optics_sources']:
                        st.markdown("**Lichtquellen:**")
                        for i, source in enumerate(st.session_state['optics_sources']):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"💡 {source.source_type} @ ({source.position[0]:.3f}, {source.position[1]:.3f})m")
                            with col2:
                                if st.button("🗑️", key=f"del_src_{i}"):
                                    st.session_state['optics_sources'].pop(i)
                                    st.rerun()
                    
                    st.markdown("---")
                    
                    if trace_btn or 'optics_system' in st.session_state:
                        if len(st.session_state['optics_elements']) == 0:
                            st.warning("⚠️ Bitte Element hinzufügen.")
                        elif len(st.session_state['optics_sources']) == 0:
                            st.warning("⚠️ Bitte Lichtquelle hinzufügen.")
                        else:
                            try:
                                with st.spinner("🔄 Berechne Strahlengang..."):
                                    system = OpticalSystem(
                                        st.session_state['optics_elements'],
                                        st.session_state['optics_sources']
                                    )
                                    system.trace_rays(max_distance=2.0)
                                    st.session_state['optics_system'] = system
                                
                                st.success(f"{t('success')} {len(system.rays)} {t('rays_traced')}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    show_focal = st.checkbox(t("show_focal"), value=True, key="show_focal_opt")
                                with col2:
                                    show_constr = st.checkbox(t("show_construction"), value=False, key="show_constr_opt")
                                
                                fig = plot_optical_system(system, show_focal_points=show_focal, 
                                                        show_construction_rays=show_constr)
                                st.pyplot(fig)
                                
                                st.markdown("---")
                                st.subheader(t("calculations"))
                                
                                lenses = [e for e in system.elements if isinstance(e, Lens)]
                                if lenses:
                                    for lens in lenses:
                                        with st.expander(f"{t('lens_equation')}: {lens.name}", expanded=False):
                                            st.write(f"**{t('focal_length')}:** f = {lens.focal_length:.4f} m")
                                            st.write(f"**{t('optical_power')}:** D = {lens.optical_power:.2f} dpt")
                                            
                                            g = st.number_input(f"{t('object_distance')} {lens.name}",
                                                value=0.3,
                                                format="%.4f",
                                                key=f"g_{lens.name}"
                                            )
                                            
                                            if abs(g) > 1e-6:
                                                b, V, h = system.calculate_image_position(g, lens)
                                                
                                                if abs(b) < 100:
                                                    st.write(f"**{t('image_distance')}:** b = {b:.4f} m")
                                                    st.write(f"**{t('magnification')}:** V = {V:.3f}x")
                                                    
                                                    image_pos = lens.position + b
                                                    st.write(f"**{t('image_position')}:** x = {image_pos:.4f} m")
                                                    
                                                    if V < 0:
                                                        st.write("**Bildtyp:** Reell, umgekehrt")
                                                    else:
                                                        st.write("**Bildtyp:** Virtuell, aufrecht")
                                                else:
                                                    st.write("**Bildweite:** b → ∞")
                                
                                st.markdown("---")
                                st.subheader(t("statistics"))
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(t("elements"), len(system.elements))
                                with col2:
                                    st.metric(t("light_sources"), len(system.light_sources))
                                with col3:
                                    st.metric(t("rays"), len(system.rays))
                                
                            except Exception as e:
                                st.error(f"❌ Fehler: {e}")
                                import traceback
                                st.code(traceback.format_exc())
        
        with tabs[3]:
            st.header("Daten exportieren")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Mechanik-Simulation")
                
                if 'last_sim' not in st.session_state:
                    st.info("ℹ️ Führe zuerst eine Mechanik-Simulation durch")
                else:
                    last_sim = st.session_state['last_sim']
                    
                    if st.button("CSV generieren", use_container_width=True, key="csv_mech"):
                        try:
                            csv_data = export_csv(last_sim['bodies'], last_sim['data'])
                            st.download_button(
                                label=t("download_csv"),
                                data=csv_data,
                                file_name=f"simulation_{int(time.time())}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success("✅ CSV bereit")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                    
                    st.markdown("---")
                    st.subheader("📊 Daten-Zusammenfassung")
                    data = last_sim['data']
                    st.write(f"**{t('data_points')}:** {len(data['times'])}")
                    st.write(f"**Zeitbereich:** {data['times'][0]:.3f} - {data['times'][-1]:.3f} s")
                    st.write(f"**{t('objects')}:** {len(last_sim['bodies'])}")
            
            with col2:
                st.subheader("🔬 Optik-System")
                
                if HAS_OPTICS and 'optics_system' in st.session_state:
                    system = st.session_state['optics_system']
                    
                    if st.button("JSON Export", use_container_width=True, key="json_optics"):
                        try:
                            optics_data = {
                                'elements': [],
                                'sources': []
                            }
                            
                            for elem in system.elements:
                                elem_dict = {
                                    'type': elem.__class__.__name__,
                                    'position': elem.position,
                                    'name': elem.name,
                                    'active': elem.active
                                }
                                
                                if isinstance(elem, Lens):
                                    elem_dict.update({
                                        'focal_length': elem.focal_length,
                                        'diameter': elem.diameter,
                                        'n': elem.n
                                    })
                                elif isinstance(elem, Screen):
                                    elem_dict['height'] = elem.height
                                elif isinstance(elem, Mirror):
                                    elem_dict.update({
                                        'angle': elem.angle,
                                        'height': elem.height,
                                        'curvature_radius': elem.curvature_radius
                                    })
                                elif isinstance(elem, Aperture):
                                    elem_dict['diameter'] = elem.diameter
                                
                                optics_data['elements'].append(elem_dict)
                            
                            for source in system.light_sources:
                                optics_data['sources'].append({
                                    'position': source.position.tolist(),
                                    'source_type': source.source_type,
                                    'num_rays': source.num_rays,
                                    'angle_spread': source.angle_spread,
                                    'wavelength': source.wavelength
                                })
                            
                            json_str = json.dumps(optics_data, indent=2)
                            st.download_button(
                                label=t("download_json"),
                                data=json_str,
                                file_name=f"optics_system_{int(time.time())}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                            st.success("✅ JSON bereit")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                    
                    st.markdown("---")
                    st.subheader("📊 System-Info")
                    st.write(f"**Elemente:** {len(system.elements)}")
                    st.write(f"**Lichtquellen:** {len(system.light_sources)}")
                    st.write(f"**Strahlen:** {len(system.rays)}")
                else:
                    st.info("ℹ️ Führe zuerst eine Optik-Berechnung durch")
    
    try:
        main()
    except Exception as e:
        st.error(f"Anwendungsfehler: {e}")
        import traceback
        st.code(traceback.format_exc())

if not HAS_STREAMLIT and __name__ == '__main__':
    print("Streamlit nicht installiert. Führe Demo aus...")
    bodies, conns, note = scenario_elastic_collision()
    sim = Simulator(bodies, conns, restitution=1.0)
    data = sim.run()
    print(f"Demo abgeschlossen: {note}")
    print(f"Zeitschritte: {len(data['times'])}")
    print(f"Stöße: {len(sim.collision_events)}")


# ---- ADDED: Electric field controls (Ex, Ey, Ez) ----
try:
    if hasattr(self, 'layout'):
        efield_group = QtWidgets.QGroupBox("Elektrisches Feld")
        ef_layout = QtWidgets.QFormLayout()
        self.ex_spin = QtWidgets.QDoubleSpinBox(); self.ex_spin.setRange(-1e6, 1e6); self.ex_spin.setDecimals(6)
        self.ey_spin = QtWidgets.QDoubleSpinBox(); self.ey_spin.setRange(-1e6, 1e6); self.ey_spin.setDecimals(6)
        self.ez_spin = QtWidgets.QDoubleSpinBox(); self.ez_spin.setRange(-1e6, 1e6); self.ez_spin.setDecimals(6)
        ef_layout.addRow("Ex", self.ex_spin)
        ef_layout.addRow("Ey", self.ey_spin)
        ef_layout.addRow("Ez", self.ez_spin)
        efield_group.setLayout(ef_layout)
        try:
            # try to insert under existing magnet field group if present
            self.layout.addWidget(efield_group)
        except Exception:
            pass
except Exception:
    pass
# Connect to a simple setter if model exists
try:
    if hasattr(self, 'model'):
        self.ex_spin.valueChanged.connect(lambda v: getattr(self.model, 'set_e_field', lambda i,v: None)(0, v))
        self.ey_spin.valueChanged.connect(lambda v: getattr(self.model, 'set_e_field', lambda i,v: None)(1, v))
        self.ez_spin.valueChanged.connect(lambda v: getattr(self.model, 'set_e_field', lambda i,v: None)(2, v))
except Exception:
    pass
# ---- end efield addition ----


# ---- ADDED: helper to call bloch_module if available ----
try:
    from .bloch_module import simulate as bloch_simulate  # package-style if module in same package
except Exception:
    try:
        from bloch_module import simulate as bloch_simulate
    except Exception:
        bloch_simulate = None

def run_bloch_preset(name, t_end=5.0):
    if bloch_simulate is None:
        print("bloch_module not available")
        return None
    cfg = name
    if isinstance(name, str):
        cfg = name
    t, y = bloch_simulate(cfg, t_span=(0, t_end))
    return t, y
# ---- end bloch helper ----



# ---- ADDED: orbital preset generators (rows) ----
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

lim = 32 * 1.496e11
zoom = False

plt.rcParams['font.family'] = 'cambria'
fig, ax = plt.subplots()
ax.set_aspect('equal')
fig.set_facecolor('#010b19')
ax.set_facecolor('#010b19')

# -- Constants -- #
dt = 86400
planet_data = {
    "sun":      [1.989e30,     [0.0, 0.0],           [0.0, 0.0]],
    "mercury":  [3.301e23,     [5.79e10, 0.0],       [0.0, 47890.0]],
    "venus":    [4.867e24,     [1.082e11, 0.0],      [0.0, 35020.0]],
    "earth":    [5.972e24,     [1.496e11, 0.0],      [0.0, 29780.0]],
    "mars":     [6.417e23,     [2.279e11, 0.0],      [0.0, 24130.0]],
    "jupiter":  [1.899e27,     [7.785e11, 0.0],      [0.0, 13070.0]],
    "saturn":   [5.685e26,     [1.433e12, 0.0],      [0.0, 9680.0]],
    "uranus":   [8.682e25,     [2.877e12, 0.0],      [0.0, 6810.0]],
    "neptune":  [1.024e26,     [4.503e12, 0.0],      [0.0, 5430.0]]
}
planet_colors = {
    "sun": "yellow",
    "mercury": "gray",
    "venus": "orange",
    "earth": "blue",
    "mars": "red",
    "jupiter": "brown",
    "saturn": "gold",
    "uranus": "cyan",
    "neptune": "violet"
}

planet_sizes = {
    "sun": 22,        # ~109x Earth's size → scaled down to stay visible
    "mercury": 2,     # 0.38x Earth
    "venus": 5,       # 0.95x Earth
    "earth": 5,
    "mars": 3,        # 0.53x Earth
    "jupiter": 11,    # 11x Earth
    "saturn": 10,     # 9x Earth
    "uranus": 6,      # 4x Earth
    "neptune": 6      # 3.9x Earth
}

if not zoom:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("Solar System (time = 10 days)", color='white')
    planet_sizes.update({"sun":1})
    dt = 864000
else:
    ax.set_xlim(-lim / 10, lim / 10)
    ax.set_ylim(-lim / 10, lim / 10)
    del planet_data["jupiter"], planet_data["saturn"], planet_data["uranus"], planet_data["neptune"]
    ax.set_title("Inner Planets (time = 1 day)", color='white')

class Planet:
    def __init__(self, name, mass, pos, vel):
        self.name = name
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(vel, dtype=float)
        self.acceleration = np.zeros(2)

planets = []



for name, (mass, pos, vel) in planet_data.items():
    planet = Planet(name, mass, pos, vel)
    planets.append(planet)
    color = planet_colors.get(name, "white")
    size = planet_sizes.get(name, 4)
    planet.color = color
    planet.marker, = ax.plot([], [], 'o', color=color, markersize=size)
    planet.trail, = ax.plot([], [], '-', lw=0.7, color=color, alpha=0.6)
    planet.trail_x = []
    planet.trail_y = []

    # Assign trail length per planet
    if name in {"mercury", "venus", "earth", "mars"}:
        planet.trail_length = 250
    else:
        planet.trail_length = 700

def init():
    for p in planets:
        p.marker.set_data([p.pos[0]], [p.pos[1]])
    return [p.marker for p in planets]


def compute_acceleration(planets):
    G = 6.67430e-11

    for p in planets:
        total_acceleration = np.zeros(2)

        for other in planets:
            if p is other:
                continue

            r = other.pos - p.pos
            dist = np.linalg.norm(r)
            if dist == 0:
                continue

            force_dir = r / dist
            acc = G * other.mass / dist ** 2
            total_acceleration += acc * force_dir
        p.acceleration = total_acceleration


# noinspection SpellCheckingInspection
def velocity_verlet_step(planets, dt):

    for p in planets:
        p.pos += p.velocity * dt + 0.5 * p.acceleration * dt ** 2

    old_accelerations = [p.acceleration.copy() for p in planets]
    compute_acceleration(planets)

    for p, a_old in zip(planets, old_accelerations):
        p.velocity += 0.5 * (a_old + p.acceleration) * dt

compute_acceleration(planets)

def update(frame):
    velocity_verlet_step(planets, dt)

    for p in planets:
        p.marker.set_data([p.pos[0]], [p.pos[1]])
        p.trail_x.append(p.pos[0])
        p.trail_y.append(p.pos[1])
        if len(p.trail_x) > p.trail_length:
            p.trail_x.pop(0)
            p.trail_y.pop(0)
        p.trail.set_data(p.trail_x, p.trail_y)
    return [p.marker for p in planets] + [p.trail for p in planets]



ani = FuncAnimation(fig, update, init_func=init, frames=1000, interval=15, blit=True)
plt.show()

# ---- end presets ----
