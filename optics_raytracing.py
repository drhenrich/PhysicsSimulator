# ============================================================
# physics_simulator/optics_raytracing.py
# Strahlenoptik: Ray-Tracing für optische Systeme
# Linsen, Spiegel, Schirme - für Physiklehre
# ============================================================

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ============================================================
# OPTISCHE ELEMENTE
# ============================================================

@dataclass
class LightRay:
    """Ein Lichtstrahl mit Position und Richtung"""
    origin: np.ndarray  # Startpunkt (x, y)
    direction: np.ndarray  # Richtungsvektor (normiert)
    wavelength: float = 550e-9  # Wellenlänge in m (Standard: grün)
    intensity: float = 1.0
    path: List[np.ndarray] = field(default_factory=list)  # Verlauf des Strahls
    active: bool = True  # Ob Strahl noch aktiv ist
    
    def __post_init__(self):
        if not self.path:
            self.path = [self.origin.copy()]
        # Richtung normieren
        norm = np.linalg.norm(self.direction)
        if norm > 1e-10:
            self.direction = self.direction / norm


@dataclass
class OpticalElement:
    """Basisklasse für optische Elemente"""
    position: float  # x-Position
    name: str
    active: bool = True


@dataclass
class Lens(OpticalElement):
    """Dünne Linse"""
    focal_length: float = 0.2  # Brennweite in m (positiv=konvex, negativ=konkav)
    diameter: float = 0.1  # Durchmesser in m
    n: float = 1.5  # Brechungsindex
    
    @property
    def optical_power(self):
        """Brechkraft in Dioptrien"""
        return 1.0 / self.focal_length if abs(self.focal_length) > 1e-10 else 0
    
    def refract_ray(self, ray: LightRay, y_intersect: float) -> Optional[LightRay]:
        """
        Brechung eines Strahls an der Linse (Dünne-Linsen-Näherung)
        
        Args:
            ray: Eingehender Lichtstrahl
            y_intersect: y-Koordinate des Auftreffpunkts
            
        Returns:
            Gebrochener Strahl oder None (wenn außerhalb der Linse)
        """
        if abs(y_intersect) > self.diameter / 2:
            return None  # Strahl außerhalb der Linse
        
        # Dünne Linse: Ablenkung proportional zu Abstand von optischer Achse
        deflection_angle = -y_intersect / self.focal_length if abs(self.focal_length) > 1e-10 else 0
        
        # Neue Richtung
        old_angle = math.atan2(ray.direction[1], ray.direction[0])
        new_angle = old_angle + deflection_angle
        
        new_direction = np.array([math.cos(new_angle), math.sin(new_angle)])
        
        return LightRay(
            origin=np.array([self.position, y_intersect]),
            direction=new_direction,
            wavelength=ray.wavelength,
            intensity=ray.intensity * 0.95  # 5% Verlust durch Reflexion
        )


@dataclass
class Mirror(OpticalElement):
    """Ebener oder gekrümmter Spiegel"""
    angle: float = 0.0  # Neigungswinkel in Grad
    curvature_radius: float = float('inf')  # Krümmungsradius (inf = eben)
    height: float = 0.1  # Höhe in m
    
    def reflect_ray(self, ray: LightRay, y_intersect: float) -> Optional[LightRay]:
        """
        Reflexion eines Strahls am Spiegel
        
        Args:
            ray: Eingehender Lichtstrahl
            y_intersect: y-Koordinate des Auftreffpunkts
            
        Returns:
            Reflektierter Strahl oder None
        """
        if abs(y_intersect) > self.height / 2:
            return None
        
        # Normale des Spiegels
        angle_rad = math.radians(self.angle)
        normal = np.array([-math.sin(angle_rad), math.cos(angle_rad)])
        
        # Reflexionsgesetz: r = d - 2(d·n)n
        incident = ray.direction
        dot_product = np.dot(incident, normal)
        reflected = incident - 2 * dot_product * normal
        
        return LightRay(
            origin=np.array([self.position, y_intersect]),
            direction=reflected,
            wavelength=ray.wavelength,
            intensity=ray.intensity * 0.9  # 10% Verlust
        )


@dataclass
class Screen(OpticalElement):
    """Auffangschirm"""
    height: float = 0.2  # Höhe in m
    
    def collect_intersection(self, ray: LightRay) -> Optional[float]:
        """
        Gibt y-Koordinate zurück, wo Strahl auftrifft
        
        Args:
            ray: Lichtstrahl
            
        Returns:
            y-Koordinate oder None
        """
        if abs(ray.direction[0]) < 1e-10:
            return None
        
        # Schnittpunkt mit vertikaler Linie bei x = position
        t = (self.position - ray.origin[0]) / ray.direction[0]
        if t < 0:
            return None
        
        y = ray.origin[1] + t * ray.direction[1]
        
        if abs(y) <= self.height / 2:
            return y
        return None


@dataclass
class Aperture(OpticalElement):
    """Blende"""
    diameter: float = 0.05  # Öffnung in m
    
    def blocks_ray(self, y_intersect: float) -> bool:
        """Prüft, ob Strahl blockiert wird"""
        return abs(y_intersect) > self.diameter / 2


@dataclass
class LightSource:
    """Lichtquelle"""
    position: np.ndarray  # (x, y)
    source_type: str  # 'point', 'parallel', 'object'
    num_rays: int = 5
    angle_spread: float = 30.0  # Öffnungswinkel in Grad
    wavelength: float = 550e-9
    object_height: float = 0.02  # Höhe des Objekts (für 'object'-Typ)
    
    def generate_rays(self) -> List[LightRay]:
        """
        Erzeugt Lichtstrahlen abhängig vom Quellentyp
        
        Returns:
            Liste von LightRay-Objekten
        """
        rays = []
        
        if self.source_type == 'point':
            # Punktquelle: Strahlen in verschiedene Richtungen
            angles = np.linspace(-self.angle_spread/2, self.angle_spread/2, self.num_rays)
            for angle_deg in angles:
                angle_rad = math.radians(angle_deg)
                direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
                rays.append(LightRay(
                    origin=self.position.copy(),
                    direction=direction,
                    wavelength=self.wavelength
                ))
        
        elif self.source_type == 'parallel':
            # Parallele Strahlen (z.B. von ferner Quelle)
            y_positions = np.linspace(-0.05, 0.05, self.num_rays)
            for y in y_positions:
                origin = self.position + np.array([0, y])
                direction = np.array([1.0, 0.0])
                rays.append(LightRay(
                    origin=origin,
                    direction=direction,
                    wavelength=self.wavelength
                ))
        
        elif self.source_type == 'object':
            # Objekt: Mehrere Punkte mit je mehreren Strahlen
            y_positions = np.linspace(-self.object_height/2, self.object_height/2, 
                                     max(2, self.num_rays // 3))
            angles = [-10, 0, 10]  # Drei Strahlen pro Objektpunkt
            
            for y in y_positions:
                origin = self.position + np.array([0, y])
                for angle_deg in angles:
                    angle_rad = math.radians(angle_deg)
                    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
                    rays.append(LightRay(
                        origin=origin,
                        direction=direction,
                        wavelength=self.wavelength
                    ))
        
        return rays


# ============================================================
# RAY TRACER
# ============================================================

class OpticalSystem:
    """Optisches System mit mehreren Elementen"""
    
    def __init__(self, elements: List[OpticalElement], light_sources: List[LightSource]):
        """
        Initialisierung des optischen Systems
        
        Args:
            elements: Liste optischer Elemente
            light_sources: Liste von Lichtquellen
        """
        self.elements = sorted(elements, key=lambda e: e.position)
        self.light_sources = light_sources
        self.rays = []
        self.screen_hits = []
    
    def trace_rays(self, max_distance: float = 2.0, max_bounces: int = 20):
        """
        Verfolgt alle Lichtstrahlen durch das System
        
        Args:
            max_distance: Maximale Propagationsdistanz
            max_bounces: Maximale Anzahl von Interaktionen
        """
        self.rays = []
        self.screen_hits = []
        
        # Generiere alle Strahlen von allen Quellen
        for source in self.light_sources:
            initial_rays = source.generate_rays()
            
            for ray in initial_rays:
                self._trace_single_ray(ray, max_distance, max_bounces)
    
    def _trace_single_ray(self, ray: LightRay, max_distance: float, max_bounces: int):
        """Verfolgt einen einzelnen Strahl durch das System"""
        current_ray = ray
        bounces = 0
        
        while current_ray.active and bounces < max_bounces:
            # Finde nächstes Element in Strahlrichtung
            next_element, intersect_y = self._find_next_intersection(current_ray)
            
            if next_element is None:
                # Kein weiteres Element -> Strahl endet
                end_x = current_ray.origin[0] + max_distance * current_ray.direction[0]
                end_y = current_ray.origin[1] + max_distance * current_ray.direction[1]
                current_ray.path.append(np.array([end_x, end_y]))
                current_ray.active = False
                break
            
            # Propagiere zum Element
            current_ray.path.append(np.array([next_element.position, intersect_y]))
            
            # Interaktion mit Element
            if isinstance(next_element, Lens):
                new_ray = next_element.refract_ray(current_ray, intersect_y)
                if new_ray is None:
                    current_ray.active = False
                    break
                current_ray = new_ray
            
            elif isinstance(next_element, Mirror):
                new_ray = next_element.reflect_ray(current_ray, intersect_y)
                if new_ray is None:
                    current_ray.active = False
                    break
                current_ray = new_ray
            
            elif isinstance(next_element, Screen):
                # Strahl erreicht Schirm
                self.screen_hits.append({
                    'position': intersect_y,
                    'intensity': current_ray.intensity,
                    'wavelength': current_ray.wavelength
                })
                current_ray.active = False
                break
            
            elif isinstance(next_element, Aperture):
                if next_element.blocks_ray(intersect_y):
                    current_ray.active = False
                    break
                # Sonst: Strahl passiert Blende
                current_ray.origin = np.array([next_element.position, intersect_y])
            
            bounces += 1
        
        self.rays.append(current_ray)
    
    def _find_next_intersection(self, ray: LightRay) -> Tuple[Optional[OpticalElement], Optional[float]]:
        """
        Findet das nächste Element, das der Strahl trifft
        
        Returns:
            (element, y_intersect) oder (None, None)
        """
        if abs(ray.direction[0]) < 1e-10:
            return None, None
        
        min_distance = float('inf')
        closest_element = None
        closest_y = None
        
        for element in self.elements:
            if not element.active:
                continue
            
            # Ist Element vor dem Strahl?
            if ray.direction[0] > 0 and element.position <= ray.origin[0]:
                continue
            if ray.direction[0] < 0 and element.position >= ray.origin[0]:
                continue
            
            # Berechne Schnittpunkt
            t = (element.position - ray.origin[0]) / ray.direction[0]
            if t < 1e-10:  # Zu nah oder hinter dem Strahl
                continue
            
            y_intersect = ray.origin[1] + t * ray.direction[1]
            distance = abs(element.position - ray.origin[0])
            
            if distance < min_distance:
                min_distance = distance
                closest_element = element
                closest_y = y_intersect
        
        return closest_element, closest_y


# ============================================================
# VISUALISIERUNG
# ============================================================

def plot_optical_system(
    system: OpticalSystem,
    title: str = "Optisches System",
    show_elements: bool = True,
    wavelength_colors: bool = True
) -> Optional[object]:
    """
    Visualisiert das optische System mit Strahlengang
    
    Args:
        system: OpticalSystem-Objekt
        title: Plot-Titel
        show_elements: Zeige optische Elemente
        wavelength_colors: Färbe Strahlen nach Wellenlänge
        
    Returns:
        Plotly oder Matplotlib Figure
    """
    if not HAS_PLOTLY and not HAS_MATPLOTLIB:
        return None
    
    if HAS_PLOTLY:
        fig = go.Figure()
        
        # Zeichne Strahlen
        for ray in system.rays:
            if len(ray.path) < 2:
                continue
            
            path_array = np.array(ray.path)
            x_vals = path_array[:, 0]
            y_vals = path_array[:, 1]
            
            # Farbe basierend auf Wellenlänge
            if wavelength_colors:
                color = wavelength_to_rgb(ray.wavelength)
            else:
                color = 'orange'
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color=color, width=1),
                opacity=min(1.0, ray.intensity),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Zeichne optische Elemente
        if show_elements:
            for element in system.elements:
                if isinstance(element, Lens):
                    # Linse als vertikale Linie mit Pfeil
                    y_range = element.diameter / 2
                    fig.add_trace(go.Scatter(
                        x=[element.position, element.position],
                        y=[-y_range, y_range],
                        mode='lines+markers',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8, symbol='diamond'),
                        name=f'{element.name} (f={element.focal_length:.3f}m)',
                        showlegend=True
                    ))
                
                elif isinstance(element, Mirror):
                    y_range = element.height / 2
                    fig.add_trace(go.Scatter(
                        x=[element.position, element.position],
                        y=[-y_range, y_range],
                        mode='lines',
                        line=dict(color='silver', width=4),
                        name=element.name,
                        showlegend=True
                    ))
                
                elif isinstance(element, Screen):
                    y_range = element.height / 2
                    fig.add_trace(go.Scatter(
                        x=[element.position, element.position],
                        y=[-y_range, y_range],
                        mode='lines',
                        line=dict(color='gray', width=2, dash='dot'),
                        name=element.name,
                        showlegend=True
                    ))
                
                elif isinstance(element, Aperture):
                    y_range = element.diameter / 2
                    # Blende: Zwei Linien oben und unten
                    fig.add_trace(go.Scatter(
                        x=[element.position, element.position],
                        y=[y_range, 0.5],
                        mode='lines',
                        line=dict(color='black', width=4),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[element.position, element.position],
                        y=[-y_range, -0.5],
                        mode='lines',
                        line=dict(color='black', width=4),
                        name=element.name,
                        showlegend=True
                    ))
        
        # Markiere Lichtquellen
        for source in system.light_sources:
            fig.add_trace(go.Scatter(
                x=[source.position[0]],
                y=[source.position[1]],
                mode='markers',
                marker=dict(size=12, color='yellow', symbol='star'),
                name='Lichtquelle',
                showlegend=True
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='x [m]',
            yaxis_title='y [m]',
            height=600,
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='white',
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        return fig
    
    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Zeichne Strahlen
        for ray in system.rays:
            if len(ray.path) < 2:
                continue
            path_array = np.array(ray.path)
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   'orange', linewidth=0.8, alpha=min(1.0, ray.intensity))
        
        # Zeichne Elemente
        if show_elements:
            for element in system.elements:
                if isinstance(element, Lens):
                    y_range = element.diameter / 2
                    ax.plot([element.position, element.position], 
                           [-y_range, y_range], 'b-', linewidth=2, 
                           label=f'{element.name}')
                elif isinstance(element, Screen):
                    y_range = element.height / 2
                    ax.plot([element.position, element.position],
                           [-y_range, y_range], 'k--', linewidth=1,
                           label=element.name)
        
        # Lichtquellen
        for source in system.light_sources:
            ax.plot(source.position[0], source.position[1], 
                   'y*', markersize=15, label='Quelle')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        return fig
    
    return None


def wavelength_to_rgb(wavelength: float) -> str:
    """
    Konvertiert Wellenlänge (in m) zu RGB-Farbe
    
    Args:
        wavelength: Wellenlänge in Metern
        
    Returns:
        RGB-String im Format 'rgb(r,g,b)'
    """
    wl_nm = wavelength * 1e9  # Konvertiere zu Nanometern
    
    if wl_nm < 380:
        r, g, b = 138, 43, 226  # Violett
    elif wl_nm < 440:
        r = int(-(wl_nm - 440) / (440 - 380) * 255)
        g, b = 0, 255
    elif wl_nm < 490:
        r = 0
        g = int((wl_nm - 440) / (490 - 440) * 255)
        b = 255
    elif wl_nm < 510:
        r, g = 0, 255
        b = int(-(wl_nm - 510) / (510 - 490) * 255)
    elif wl_nm < 580:
        r = int((wl_nm - 510) / (580 - 510) * 255)
        g, b = 255, 0
    elif wl_nm < 645:
        r = 255
        g = int(-(wl_nm - 645) / (645 - 580) * 255)
        b = 0
    elif wl_nm < 781:
        r, g, b = 255, 0, 0
    else:
        r, g, b = 255, 0, 0
    
    return f'rgb({r},{g},{b})'


# ============================================================
# VORDEFINIERTE PRESETS
# ============================================================

def preset_converging_lens() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Sammellinse mit parallelen Strahlen"""
    lens = Lens(position=0.0, name="Sammellinse", focal_length=0.2, diameter=0.1)
    screen = Screen(position=0.2, name="Schirm", height=0.15)
    
    source = LightSource(
        position=np.array([-0.5, 0.0]),
        source_type='parallel',
        num_rays=7
    )
    
    return [lens, screen], [source], "Sammellinse mit parallelen Strahlen - Fokussierung im Brennpunkt"


def preset_diverging_lens() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Zerstreuungslinse"""
    lens = Lens(position=0.0, name="Zerstreuungslinse", focal_length=-0.2, diameter=0.1)
    screen = Screen(position=0.4, name="Schirm", height=0.2)
    
    source = LightSource(
        position=np.array([-0.5, 0.0]),
        source_type='parallel',
        num_rays=7
    )
    
    return [lens, screen], [source], "Zerstreuungslinse - Strahlen divergieren"


def preset_magnifying_glass() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Lupe (Vergrößerungsglas)"""
    lens = Lens(position=0.0, name="Lupe", focal_length=0.1, diameter=0.08)
    screen = Screen(position=0.3, name="Auge", height=0.2)
    
    source = LightSource(
        position=np.array([-0.05, 0.01]),
        source_type='object',
        num_rays=3,
        object_height=0.02
    )
    
    return [lens, screen], [source], "Lupe - Objekt innerhalb der Brennweite → virtuelles, vergrößertes Bild"


def preset_telescope() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Kepler-Teleskop"""
    objective = Lens(position=0.0, name="Objektiv", focal_length=0.5, diameter=0.1)
    eyepiece = Lens(position=0.55, name="Okular", focal_length=0.05, diameter=0.04)
    screen = Screen(position=0.8, name="Auge", height=0.1)
    
    source = LightSource(
        position=np.array([-2.0, 0.05]),
        source_type='parallel',
        num_rays=5,
        angle_spread=2.0
    )
    
    return [objective, eyepiece, screen], [source], "Kepler-Teleskop - Zwei-Linsen-System für ferne Objekte"


def preset_microscope() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Einfaches Mikroskop"""
    objective = Lens(position=0.0, name="Objektiv", focal_length=0.02, diameter=0.03)
    eyepiece = Lens(position=0.2, name="Okular", focal_length=0.05, diameter=0.04)
    screen = Screen(position=0.35, name="Auge", height=0.08)
    
    source = LightSource(
        position=np.array([-0.025, 0.002]),
        source_type='object',
        num_rays=3,
        object_height=0.004
    )
    
    return [objective, eyepiece, screen], [source], "Mikroskop - Starke Vergrößerung naher Objekte"


def preset_projector() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Projektor / Beamer"""
    lens = Lens(position=0.0, name="Projektionslinse", focal_length=0.15, diameter=0.1)
    screen = Screen(position=0.6, name="Leinwand", height=0.4)
    
    source = LightSource(
        position=np.array([-0.2, 0.02]),
        source_type='object',
        num_rays=5,
        object_height=0.04
    )
    
    return [lens, screen], [source], "Projektor - Wirft vergrößertes Bild auf Leinwand"


def preset_two_lenses() -> Tuple[List[OpticalElement], List[LightSource], str]:
    """Zwei-Linsen-System"""
    lens1 = Lens(position=-0.2, name="Linse 1", focal_length=0.15, diameter=0.08)
    lens2 = Lens(position=0.2, name="Linse 2", focal_length=0.15, diameter=0.08)
    screen = Screen(position=0.5, name="Schirm", height=0.2)
    
    source = LightSource(
        position=np.array([-0.5, 0.02]),
        source_type='object',
        num_rays=5,
        object_height=0.04
    )
    
    return [lens1, lens2, screen], [source], "Zwei-Linsen-System - Kombinierte optische Elemente"


OPTICS_PRESETS = {
    'converging_lens': {
        'function': preset_converging_lens,
        'name': 'Sammellinse',
        'name_en': 'Converging Lens',
        'description': 'Parallele Strahlen werden im Brennpunkt fokussiert',
        'description_en': 'Parallel rays are focused at focal point'
    },
    'diverging_lens': {
        'function': preset_diverging_lens,
        'name': 'Zerstreuungslinse',
        'name_en': 'Diverging Lens',
        'description': 'Konkave Linse streut parallele Strahlen',
        'description_en': 'Concave lens diverges parallel rays'
    },
    'magnifying_glass': {
        'function': preset_magnifying_glass,
        'name': 'Lupe',
        'name_en': 'Magnifying Glass',
        'description': 'Vergrößerung durch Objekt innerhalb der Brennweite',
        'description_en': 'Magnification with object inside focal length'
    },
    'telescope': {
        'function': preset_telescope,
        'name': 'Teleskop (Kepler)',
        'name_en': 'Telescope (Kepler)',
        'description': 'Zweistufiges System für ferne Objekte',
        'description_en': 'Two-stage system for distant objects'
    },
    'microscope': {
        'function': preset_microscope,
        'name': 'Mikroskop',
        'name_en': 'Microscope',
        'description': 'Starke Vergrößerung naher, kleiner Objekte',
        'description_en': 'Strong magnification of nearby small objects'
    },
    'projector': {
        'function': preset_projector,
        'name': 'Projektor',
        'name_en': 'Projector',
        'description': 'Wirft vergrößertes Bild auf Leinwand',
        'description_en': 'Projects magnified image onto screen'
    },
    'two_lenses': {
        'function': preset_two_lenses,
        'name': 'Zwei-Linsen-System',
        'name_en': 'Two-Lens System',
        'description': 'Kombination zweier Linsen',
        'description_en': 'Combination of two lenses'
    }
}


# ============================================================
# Schlanker Ray-Matrix Layer für Streamlit-UI (ui_optics_raytracing.py)
# ============================================================
import io


@dataclass
class ThinLens:
    """Paraxiale dünne Linse, benutzt nur Brennweite f."""
    f: float


@dataclass
class Space:
    """Freier Raum/Sektion der optischen Achse."""
    L: float


@dataclass
class Aperture:
    """Einfache Kreisblende (1D: Radius um optische Achse)."""
    radius: float


def fan_rays(n: int = 21, height: float = 0.01, ang: float = 0.05):
    """Erzeuge Start-Höhen und Winkel für Strahlenfächer (paraxiale Näherung)."""
    n = max(3, int(n))
    y0 = np.linspace(-height / 2.0, height / 2.0, n)
    th0 = np.linspace(-ang / 2.0, ang / 2.0, n)
    return y0, th0


def trace_system(elems, y0, th0):
    """
    Paraxiales Raytracing mit ABCD-Matrizen.
    elems: Liste von ("space"/"lens"/"aperture", obj)
    returns (xs, ys) mit shape (n_steps, n_rays) für Plotting.
    """
    y = np.array(y0, dtype=float)
    th = np.array(th0, dtype=float)
    xs = [np.zeros_like(y)]
    ys = [y.copy()]
    x_pos = 0.0

    for kind, elem in elems:
        if kind == "space":
            x_pos += elem.L
            y = y + th * elem.L
        elif kind == "lens":
            th = th - y / elem.f
        elif kind == "aperture":
            mask = np.abs(y) <= elem.radius
            y = np.where(mask, y, np.nan)
            th = np.where(mask, th, np.nan)
        xs.append(np.full_like(y, x_pos))
        ys.append(y.copy())

    return np.vstack(xs).T, np.vstack(ys).T  # (n_rays, n_steps)


def draw_system(xs, ys, figsize=(8, 3)):
    """Erzeuge PNG-Array für Streamlit-Anzeige aus (xs,ys) Linien."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return np.zeros((300, 800, 3), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.array(xs)
    ys = np.array(ys)
    for i in range(xs.shape[0]):
        ax.plot(xs[i], ys[i], color="orange", linewidth=1, alpha=0.9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.axvline(0.0, color="k", linestyle="--", alpha=0.2)
    try:
        y_min = np.nanmin(ys)
        y_max = np.nanmax(ys)
    except Exception:
        y_min, y_max = -0.05, 0.05
    if not np.isfinite(y_min):
        y_min = -0.05
    if not np.isfinite(y_max):
        y_max = 0.05
    pad = 0.02 * max(1.0, abs(y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    return img


def presets():
    """Vorkonfigurierte einfache Linsenzüge für die UI."""
    return {
        "Sammellinse": [
            ("space", Space(0.15)),
            ("lens", ThinLens(0.2)),
            ("space", Space(0.25)),
        ],
        "Chromatik Sammellinse": [
            ("space", Space(0.15)),
            ("lens", ThinLens(0.2)),
            ("space", Space(0.25)),
        ],
        "Zerstreuungslinse": [
            ("space", Space(0.15)),
            ("lens", ThinLens(-0.2)),
            ("space", Space(0.3)),
        ],
        "Teleskop (zwei Linsen)": [
            ("space", Space(0.1)),
            ("lens", ThinLens(0.5)),
            ("space", Space(0.55)),
            ("lens", ThinLens(0.05)),
            ("space", Space(0.3)),
        ],
        "Apertur-Blende": [
            ("space", Space(0.1)),
            ("aperture", Aperture(0.03)),
            ("space", Space(0.2)),
            ("lens", ThinLens(0.15)),
            ("space", Space(0.2)),
        ],
    }


# ============================================================
# TEST & DEMO
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Optik Ray-Tracing Modul Test")
    print("=" * 60)
    
    # Test aller Presets
    for preset_key, preset_data in OPTICS_PRESETS.items():
        print(f"\n🔬 Teste Preset: {preset_data['name']}")
        print(f"   {preset_data['description']}")
        
        elements, sources, note = preset_data['function']()
        system = OpticalSystem(elements, sources)
        system.trace_rays()
        
        print(f"   ✓ Elemente: {len(elements)}")
        print(f"   ✓ Lichtquellen: {len(sources)}")
        print(f"   ✓ Strahlen verfolgt: {len(system.rays)}")
        print(f"   ✓ Schirmtreffer: {len(system.screen_hits)}")
    
    print("\n" + "=" * 60)
    print("✅ Alle Tests erfolgreich!")
    print("=" * 60)
