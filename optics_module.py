# optics_module.py
# Optik-Modul für den Physics Teaching Simulator
# Strahlenoptik: Linsen, Spiegel, Schirme, Strahlengang

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math

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
    
    def __post_init__(self):
        if not self.path:
            self.path = [self.origin.copy()]
        # Richtung normieren
        norm = np.linalg.norm(self.direction)
        if norm > 0:
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
        return 1.0 / self.focal_length if self.focal_length != 0 else 0
    
    def refract_ray(self, ray: LightRay, y_intersect: float) -> Optional[LightRay]:
        """
        Brechung eines Strahls an der Linse (Dünne-Linsen-Näherung)
        y_intersect: y-Koordinate des Auftreffpunkts
        """
        if abs(y_intersect) > self.diameter / 2:
            return None  # Strahl außerhalb der Linse
        
        # Dünne Linse: Ablenkung proportional zu Abstand von opt. Achse
        deflection_angle = -y_intersect / self.focal_length
        
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
        """Reflexion eines Strahls am Spiegel"""
        if abs(y_intersect) > self.height / 2:
            return None
        
        # Normale des Spiegels
        angle_rad = math.radians(self.angle)
        normal = np.array([-math.sin(angle_rad), math.cos(angle_rad)])
        
        # Reflexionsgesetz: r = d - 2(d·n)n
        incident = ray.direction
        reflected = incident - 2 * np.dot(incident, normal) * normal
        
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
        """Gibt y-Koordinate zurück, wo Strahl auftrifft"""
        if abs(ray.direction[0]) < 1e-10:
            return None
        
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
    
    def generate_rays(self) -> List[LightRay]:
        """Erzeugt Lichtstrahlen"""
        rays = []
        
        if self.source_type == 'point':
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
            y_positions = np.linspace(-0.02, 0.02, self.num_rays)
            for y in y_positions:
                origin = self.position + np.array([0, y])
                angles = [-10, 0, 10]
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
        self.elements = sorted(elements, key=lambda e: e.position)
        self.light_sources = light_sources
        self.rays = []
    
    def trace_rays(self, max_distance: float = 2.0):
        """Verfolgt alle Lichtstrahlen durch das System"""
        self.rays = []
        
        for source in self.light_sources:
            initial_rays = source.generate_rays()
            
            for ray in initial_rays:
                self._trace_single_ray(ray, max_distance)
    
    def _trace_single_ray(self, ray: LightRay, max_distance: float):
        """Verfolgt einen einzelnen Strahl"""
        current_ray = ray
        max_steps = 50
        
        for step in range(max_steps):
            next_element, distance = self._find_next_element(current_ray)
            
            if next_element is None or distance > max_distance:
                end_point = current_ray.origin + max_distance * current_ray.direction
                current_ray.path.append(end_point)
                self.rays.append(current_ray)
                break
            
            intersection_point = current_ray.origin + distance * current_ray.direction
            y_intersect = intersection_point[1]
            
            current_ray.path.append(intersection_point)
            
            if isinstance(next_element, Lens):
                new_ray = next_element.refract_ray(current_ray, y_intersect)
                if new_ray is None:
                    self.rays.append(current_ray)
                    break
                new_ray.path = current_ray.path.copy()
                current_ray = new_ray
            
            elif isinstance(next_element, Mirror):
                new_ray = next_element.reflect_ray(current_ray, y_intersect)
                if new_ray is None:
                    self.rays.append(current_ray)
                    break
                new_ray.path = current_ray.path.copy()
                current_ray = new_ray
            
            elif isinstance(next_element, Screen):
                self.rays.append(current_ray)
                break
            
            elif isinstance(next_element, Aperture):
                if next_element.blocks_ray(y_intersect):
                    self.rays.append(current_ray)
                    break
                continue
        
        else:
            self.rays.append(current_ray)
    
    def _find_next_element(self, ray: LightRay) -> Tuple[Optional[OpticalElement], float]:
        """Findet das nächste Element in Strahlrichtung"""
        min_distance = float('inf')
        next_element = None
        
        for element in self.elements:
            if not element.active:
                continue
            
            if ray.direction[0] == 0:
                continue
            
            t = (element.position - ray.origin[0]) / ray.direction[0]
            
            if t > 1e-6 and t < min_distance:
                min_distance = t
                next_element = element
        
        return next_element, min_distance
    
    def find_focal_points(self, lens: Lens) -> Tuple[Optional[float], Optional[float]]:
        """Findet die Brennpunkte einer Linse"""
        f1 = lens.position - lens.focal_length
        f2 = lens.position + lens.focal_length
        return f1, f2
    
    def calculate_image_position(self, object_distance: float, lens: Lens) -> Tuple[float, float, float]:
        """
        Berechnet Bildposition mit Linsengleichung
        Returns: (image_distance, magnification, image_height)
        """
        if abs(object_distance) < 1e-10:
            return float('inf'), 0, 0
        
        f = lens.focal_length
        
        try:
            image_distance = 1 / (1/f - 1/object_distance)
            magnification = -image_distance / object_distance
            return image_distance, magnification, 0
        except:
            return float('inf'), 0, 0

# ============================================================
# OPTIK PRESETS
# ============================================================

def preset_converging_lens():
    """Sammellinse mit parallelen Strahlen"""
    lens = Lens(position=0.0, name="Sammellinse", focal_length=0.2, diameter=0.1)
    screen = Screen(position=0.2, name="Schirm", height=0.15)
    
    source = LightSource(
        position=np.array([-0.5, 0.0]),
        source_type='parallel',
        num_rays=7
    )
    
    return [lens, screen], [source], "Sammellinse mit parallelen Strahlen"

def preset_diverging_lens():
    """Zerstreuungslinse"""
    lens = Lens(position=0.0, name="Zerstreuungslinse", focal_length=-0.2, diameter=0.1)
    screen = Screen(position=0.4, name="Schirm", height=0.2)
    
    source = LightSource(
        position=np.array([-0.5, 0.0]),
        source_type='parallel',
        num_rays=7
    )
    
    return [lens, screen], [source], "Zerstreuungslinse"

def preset_magnifying_glass():
    """Lupe (Vergrößerungsglas)"""
    lens = Lens(position=0.0, name="Lupe", focal_length=0.1, diameter=0.08)
    screen = Screen(position=0.3, name="Auge", height=0.2)
    
    source = LightSource(
        position=np.array([-0.05, 0.01]),
        source_type='object',
        num_rays=3
    )
    
    return [lens, screen], [source], "Lupe (Objekt innerhalb Brennweite)"

def preset_telescope():
    """Einfaches Teleskop (Kepler)"""
    objective = Lens(position=0.0, name="Objektiv", focal_length=0.5, diameter=0.1)
    eyepiece = Lens(position=0.55, name="Okular", focal_length=0.05, diameter=0.04)
    screen = Screen(position=0.8, name="Auge", height=0.1)
    
    source = LightSource(
        position=np.array([-2.0, 0.05]),
        source_type='parallel',
        num_rays=5,
        angle_spread=2.0
    )
    
    return [objective, eyepiece, screen], [source], "Kepler-Teleskop"

def preset_microscope():
    """Einfaches Mikroskop"""
    objective = Lens(position=0.0, name="Objektiv", focal_length=0.02, diameter=0.03)
    eyepiece = Lens(position=0.2, name="Okular", focal_length=0.05, diameter=0.04)
    screen = Screen(position=0.35, name="Auge", height=0.08)
    
    source = LightSource(
        position=np.array([-0.025, 0.002]),
        source_type='object',
        num_rays=3
    )
    
    return [objective, eyepiece, screen], [source], "Mikroskop"

def preset_projector():
    """Projektor"""
    lens = Lens(position=0.0, name="Projektionslinse", focal_length=0.15, diameter=0.1)
    screen = Screen(position=0.6, name="Leinwand", height=0.4)
    
    source = LightSource(
        position=np.array([-0.2, 0.02]),
        source_type='object',
        num_rays=5
    )
    
    return [lens, screen], [source], "Projektor (Dia/Beamer)"

def preset_two_lenses():
    """Zwei Linsen System"""
    lens1 = Lens(position=-0.2, name="Linse 1", focal_length=0.15, diameter=0.08)
    lens2 = Lens(position=0.2, name="Linse 2", focal_length=0.15, diameter=0.08)
    screen = Screen(position=0.5, name="Schirm", height=0.2)
    
    source = LightSource(
        position=np.array([-0.5, 0.02]),
        source_type='object',
        num_rays=5
    )
    
    return [lens1, lens2, screen], [source], "Zwei-Linsen-System"

OPTICS_PRESETS = {
    'Sammellinse': preset_converging_lens,
    'Zerstreuungslinse': preset_diverging_lens,
    'Lupe': preset_magnifying_glass,
    'Teleskop': preset_telescope,
    'Mikroskop': preset_microscope,
    'Projektor': preset_projector,
    'Zwei Linsen': preset_two_lenses,
}

# ============================================================
# VISUALISIERUNG
# ============================================================

def plot_optical_system(system: OpticalSystem, show_focal_points: bool = True,
                       show_construction_rays: bool = True):
    """Plottet das optische System mit Matplotlib"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_min = min(s.position[0] for s in system.light_sources) - 0.1
    x_max = max(e.position for e in system.elements) + 0.3
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.plot([x_min, x_max], [0, 0], 'k--', linewidth=0.5, alpha=0.3, label='Optische Achse')
    
    for element in system.elements:
        if isinstance(element, Lens):
            h = element.diameter / 2
            ax.plot([element.position, element.position], [-h, h], 'b-', linewidth=3)
            
            if element.focal_length > 0:
                ax.plot(element.position, h, 'b>', markersize=10)
                ax.plot(element.position, -h, 'b>', markersize=10)
                ax.text(element.position, h + 0.03, element.name, ha='center', fontsize=9)
            else:
                ax.plot(element.position, h, 'b<', markersize=10)
                ax.plot(element.position, -h, 'b<', markersize=10)
                ax.text(element.position, h + 0.03, element.name, ha='center', fontsize=9)
            
            if show_focal_points:
                f1, f2 = system.find_focal_points(element)
                ax.plot(f1, 0, 'rx', markersize=8)
                ax.plot(f2, 0, 'rx', markersize=8)
                ax.text(f1, -0.02, 'F₁', ha='center', fontsize=8, color='red')
                ax.text(f2, -0.02, 'F₂', ha='center', fontsize=8, color='red')
        
        elif isinstance(element, Screen):
            h = element.height / 2
            ax.plot([element.position, element.position], [-h, h], 'g-', linewidth=4)
            ax.text(element.position, h + 0.03, element.name, ha='center', fontsize=9)
        
        elif isinstance(element, Mirror):
            h = element.height / 2
            ax.plot([element.position, element.position], [-h, h], 'silver', linewidth=5)
            ax.text(element.position, h + 0.03, element.name, ha='center', fontsize=9)
        
        elif isinstance(element, Aperture):
            d = element.diameter / 2
            ax.plot([element.position, element.position], [-0.2, -d], 'k-', linewidth=3)
            ax.plot([element.position, element.position], [d, 0.2], 'k-', linewidth=3)
            ax.text(element.position, 0.22, element.name, ha='center', fontsize=9)
    
    for source in system.light_sources:
        if source.source_type == 'object':
            ax.plot(source.position[0], source.position[1], 'yo', markersize=12, 
                   markeredgecolor='orange', markeredgewidth=2, label='Objekt')
            ax.arrow(source.position[0], source.position[1], 0, 0.02, 
                    head_width=0.01, head_length=0.005, fc='orange', ec='orange')
        else:
            ax.plot(source.position[0], source.position[1], 'y*', markersize=15, label='Lichtquelle')
    
    for ray in system.rays:
        path = np.array(ray.path)
        if len(path) > 1:
            color = wavelength_to_rgb(ray.wavelength)
            alpha = min(ray.intensity, 1.0)
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, alpha=alpha)
    
    ax.set_xlabel('Position (m)', fontsize=12)
    ax.set_ylabel('Höhe (m)', fontsize=12)
    ax.set_title('Optisches System - Strahlengang', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

def wavelength_to_rgb(wavelength):
    """Konvertiert Wellenlänge (m) zu RGB-Farbe"""
    wl = wavelength * 1e9  # in nm
    
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0, 1
    elif 440 <= wl < 490:
        r, g, b = 0, (wl - 440) / (490 - 440), 1
    elif 490 <= wl < 510:
        r, g, b = 0, 1, -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1, 0
    elif 580 <= wl < 645:
        r, g, b = 1, -(wl - 645) / (645 - 580), 0
    elif 645 <= wl <= 780:
        r, g, b = 1, 0, 0
    else:
        r, g, b = 0.5, 0.5, 0.5
    
    return (r, g, b)

if __name__ == '__main__':
    print("Optik-Modul geladen")
    print("Verfügbare Presets:")
    for name in OPTICS_PRESETS.keys():
        print(f"  - {name}")
    
    elements, sources, description = preset_converging_lens()
    system = OpticalSystem(elements, sources)
    system.trace_rays()
    
    print(f"\nTest: {description}")
    print(f"Anzahl Elemente: {len(elements)}")
    print(f"Anzahl Lichtquellen: {len(sources)}")
    print(f"Anzahl verfolgter Strahlen: {len(system.rays)}")
