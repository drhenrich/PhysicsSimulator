# ============================================================
# physics_simulator/io_utils.py
# CSV/JSON Import-Export für Presets und Simulationsdaten
# ============================================================

import io
import csv
import json
from typing import List, Tuple

import numpy as np

# Hybrid-Imports für direkte Ausführung und Modul-Import
try:
    from .core import Body, Connection
except ImportError:
    from core import Body, Connection


# ------------------------------------------------------------
# CSV-Export von Trajektorien
# ------------------------------------------------------------

def export_csv(bodies: List[Body], data: dict) -> bytes:
    """
    Exportiert Zeiten, Positionen und Geschwindigkeiten aller Körper als CSV.
    Rückgabe: bytes-Objekt (z.B. für Streamlit download_button).
    """
    times = data.get("times")
    positions = data.get("positions", {})
    velocities = data.get("velocities", {})

    if times is None:
        raise ValueError("Daten enthalten keine 'times'.")

    output = io.StringIO()
    writer = csv.writer(output)

    header = ["t"]
    for b in bodies:
        header += [f"{b.name}_x", f"{b.name}_y", f"{b.name}_z"]
    for b in bodies:
        header += [f"{b.name}_vx", f"{b.name}_vy", f"{b.name}_vz"]
    writer.writerow(header)

    n_steps = len(times)
    for idx in range(n_steps):
        row = [times[idx]]
        for i, b in enumerate(bodies):
            pos = positions.get(i)
            if pos is None or idx >= len(pos):
                row += ["", "", ""]
            else:
                row += list(pos[idx])
        for i, b in enumerate(bodies):
            vel = velocities.get(i)
            if vel is None or idx >= len(vel):
                row += ["", "", ""]
            else:
                row += list(vel[idx])
        writer.writerow(row)

    return output.getvalue().encode("utf-8")


# ------------------------------------------------------------
# JSON-Export eines Presets
# ------------------------------------------------------------

def export_preset_json(preset_name: str,
                       bodies: List[Body],
                       connections: List[Connection]) -> str:
    """
    Serialisiert ein Szenario (Preset) als JSON-String.
    """
    doc = {
        "name": preset_name,
        "bodies": [],
        "connections": [],
    }

    for b in bodies:
        doc["bodies"].append({
            "name": b.name,
            "pos": b.pos.tolist(),
            "vel": b.vel.tolist(),
            "mass": b.mass,
            "radius": getattr(b, "radius", 0.0),
            "charge": b.charge,
            "t0": b.t0,
            "dt": b.dt,
            "t_end": b.t_end,
            "color": b.color,
        })

    for c in connections:
        doc["connections"].append({
            "i": c.i,
            "j": c.j,
            "typ": c.typ,
            "strength": c.strength,
            "rest_length": c.rest_length,
        })

    return json.dumps(doc, indent=2)


# ------------------------------------------------------------
# JSON-Import eines Presets
# ------------------------------------------------------------

def import_preset_json(json_string: str) -> Tuple[List[Body], List[Connection]]:
    """
    Baut aus einem JSON-String wieder eine Liste von Body- und Connection-Objekten.
    """
    doc = json.loads(json_string)
    bodies = []
    connections = []

    for b in doc.get("bodies", []):
        bodies.append(Body(
            name=b["name"],
            pos=np.array(b["pos"], dtype=float),
            vel=np.array(b["vel"], dtype=float),
            mass=float(b["mass"]),
            charge=float(b["charge"]),
            t0=float(b["t0"]),
            dt=float(b["dt"]),
            t_end=float(b["t_end"]),
            radius=float(b.get("radius", 0.0)),
            color=b.get("color"),
        ))

    for c in doc.get("connections", []):
        connections.append(Connection(
            i=int(c["i"]),
            j=int(c["j"]),
            typ=str(c["typ"]),
            strength=float(c["strength"]),
            rest_length=float(c["rest_length"]),
        ))

    return bodies, connections
