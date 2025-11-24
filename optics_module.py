# Lightweight adapter to reuse optics_raytracing as optics_module
from optics_raytracing import (
    OpticalSystem,
    Lens,
    Mirror,
    Screen,
    Aperture,
    LightSource,
    LightRay,
    OpticalElement,
    OPTICS_PRESETS,
    plot_optical_system,
)

__all__ = [
    "OpticalSystem",
    "Lens",
    "Mirror",
    "Screen",
    "Aperture",
    "LightSource",
    "LightRay",
    "OpticalElement",
    "OPTICS_PRESETS",
    "plot_optical_system",
]
