# ============================================================
# scenario_framework.py — minimal registry to expose scenarios to the UI
# ============================================================
from __future__ import annotations

from typing import Callable, Dict

def build_default_registry() -> Dict[str, Callable[[], object]]:
    """Return a dict: scenario-name -> launcher callable.
    Launchers return either (bodies, connections, note) or a Scenario with .run()/.plot().
    """
    reg: Dict[str, Callable[[], object]] = {}

    # Try to import from current package or physics_simulator.*
    def _import(name: str):
        try:
            mod = __import__(name, fromlist=['*'])
            return mod
        except Exception:
            mod = __import__(f"physics_simulator.{name}", fromlist=['*'])
            return mod

    try:
        scenarios = _import("scenarios")
        # Common presets — adapt to your scenarios.py
        if hasattr(scenarios, "scenario_elastic_collision"):
            reg["Elastischer Stoß (2 Körper)"] = getattr(scenarios, "scenario_elastic_collision")
        # Add more:
        for k in dir(scenarios):
            if k.startswith("scenario_") and k not in ("scenario_elastic_collision",):
                reg[k] = getattr(scenarios, k)
    except Exception:
        # No scenarios available — registry remains minimal
        pass

    return reg
