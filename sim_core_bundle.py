import math
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
import numpy as np

# Constants
G = 6.67430e-11
k_e = 8.9875517923e9
c = 299792458.0
softening = 1e-9
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
    t0: float
    dt: float
    t_end: float
    charge: float = 0.0
    radius: float = 0.0
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
    def __init__(self, bodies: List[Body], connections: Optional[List[Connection]] = None,
                 forces: Optional[List[Callable]] = None, restitution: float = 1.0, drag: float = 0.0):
        self.bodies = bodies
        self.connections = connections if connections else []
        self.forces = forces if forces else []
        self.restitution = restitution
        self.drag = drag
        self.collision_events: List[CollisionEvent] = []

    def gravitational_force(self, i, j):
        bi, bj = self.bodies[i], self.bodies[j]
        r = bj.pos - bi.pos
        d = np.linalg.norm(r) + softening
        return G * bi.mass * bj.mass * r / (d ** 3)

    def electric_force(self, i, j):
        bi, bj = self.bodies[i], self.bodies[j]
        r = bj.pos - bi.pos
        d = np.linalg.norm(r) + softening
        return k_e * bi.charge * bj.charge * r / (d ** 3)

    def spring_force(self, b1, b2, strength, rest_length):
        r = b2.pos - b1.pos
        d = np.linalg.norm(r) + softening
        return -strength * (d - rest_length) * (r / d)

    def detect_collision(self, i, j):
        bi, bj = self.bodies[i], self.bodies[j]
        dist = np.linalg.norm(bi.pos - bj.pos)
        r_sum = (bi.radius or 0.0) + (bj.radius or 0.0)
        if r_sum <= 0:
            r_sum = (bi.mass + bj.mass) ** (1/3) * 0.001
        return dist < r_sum

    def resolve_collision(self, i, j, time):
        bi, bj = self.bodies[i], self.bodies[j]
        m1, m2 = bi.mass, bj.mass
        v1, v2 = bi.vel, bj.vel
        momentum_before = m1 * np.linalg.norm(v1) + m2 * np.linalg.norm(v2)
        energy_before = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
        v_rel = v1 - v2
        v1_new = v1 - (1 + self.restitution) * (m2 / (m1 + m2)) * v_rel
        v2_new = v2 + (1 + self.restitution) * (m1 / (m1 + m2)) * v_rel
        bi.vel = v1_new
        bj.vel = v2_new
        momentum_after = m1 * np.linalg.norm(v1_new) + m2 * np.linalg.norm(v2_new)
        energy_after = 0.5 * m1 * np.dot(v1_new, v1_new) + 0.5 * m2 * np.dot(v2_new, v2_new)
        self.collision_events.append(CollisionEvent(time, i, j, momentum_before, momentum_after, energy_before, energy_after))

    def run(self):
        if not self.bodies:
            return {}
        dt = min(b.dt for b in self.bodies)
        t_end = min(b.t_end for b in self.bodies)
        times = []
        positions = {i: [] for i in range(len(self.bodies))}
        velocities = {i: [] for i in range(len(self.bodies))}
        energies = []
        t = 0.0
        while t <= t_end:
            times.append(t)
            for i, b in enumerate(self.bodies):
                positions[i].append(b.pos.copy())
                velocities[i].append(b.vel.copy())
            accel = [np.zeros(3) for _ in self.bodies]
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    if self.bodies[i].mass > 0 and self.bodies[j].mass > 0:
                        Fg = self.gravitational_force(i, j)
                        accel[i] += Fg / self.bodies[i].mass
                        accel[j] -= Fg / self.bodies[j].mass
                    if self.bodies[i].charge or self.bodies[j].charge:
                        Fe = self.electric_force(i, j)
                        accel[i] += Fe / self.bodies[i].mass
                        accel[j] -= Fe / self.bodies[j].mass
            for c in self.connections:
                b1 = self.bodies[c.i]; b2 = self.bodies[c.j]
                Fs = self.spring_force(b1, b2, c.strength, c.rest_length)
                accel[c.i] += Fs / b1.mass
                accel[c.j] -= Fs / b2.mass
            if self.drag > 0:
                for i, b in enumerate(self.bodies):
                    accel[i] -= self.drag * b.vel / b.mass
            for f in self.forces:
                f(self.bodies, accel, t)
            for i, b in enumerate(self.bodies):
                b.vel += accel[i] * dt
                b.pos += b.vel * dt
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    if self.detect_collision(i, j):
                        self.resolve_collision(i, j, t)
            total_E = sum(0.5 * b.mass * np.dot(b.vel, b.vel) for b in self.bodies)
            energies.append(total_E)
            t += dt
        return {"times": np.array(times), "positions": {k: np.array(v) for k, v in positions.items()}, "velocities": {k: np.array(v) for k, v in velocities.items()}, "energies": np.array(energies)}

# Plotting helpers
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


def plot_trajectory_3d(bodies, data):
    positions = data.get("positions", {})
    if HAS_PLOTLY:
        fig = go.Figure()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink']
        for i, b in enumerate(bodies):
            pos = positions.get(i)
            if pos is None or len(pos) == 0:
                continue
            x, y, z = pos[:,0], pos[:,1], pos[:,2]
            color = getattr(b, 'color', None) or colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=color, width=3), name=b.name))
            fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers", 
                marker=dict(size=8, color=color), showlegend=False))
        fig.update_layout(
            title="3D-Trajektorien / 3D Trajectories",
            scene=dict(aspectmode='data', xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='z [m]'),
            height=600, margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig
    elif HAS_MATPLOTLIB:
        fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
        for i, b in enumerate(bodies):
            pos = positions.get(i)
            if pos is None: continue
            ax.plot(pos[:,0], pos[:,1], pos[:,2], label=b.name)
        ax.legend(); return fig
    else:
        return None


def plot_conservation_laws(bodies, data):
    times = data.get("times"); energies = data.get("energies")
    if times is None or energies is None or len(times) == 0:
        return None
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=energies, mode="lines", name="Gesamtenergie / Total Energy", line=dict(color='blue', width=2)))
        fig.update_layout(
            title="Energieerhaltung / Energy Conservation",
            xaxis_title="Zeit t [s]", yaxis_title="Energie E [J]",
            height=350, margin=dict(l=60, r=20, t=40, b=40)
        )
        return fig
    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots(); ax.plot(times, energies); ax.set_xlabel("t"); ax.set_ylabel("E"); return fig
    return None


def plot_collision_analysis(collision_events):
    if not collision_events:
        return None
    times = [ev.time for ev in collision_events]
    dP = [ev.momentum_after - ev.momentum_before for ev in collision_events]
    dE = [ev.energy_after - ev.energy_before for ev in collision_events]
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=times, y=dP, name="Δp [kg·m/s]", marker_color='blue'))
        fig.add_trace(go.Bar(x=times, y=dE, name="ΔE [J]", marker_color='red'))
        fig.update_layout(
            title="Stoßanalyse / Collision Analysis",
            xaxis_title="Zeit t [s]", yaxis_title="Änderung",
            barmode='group', height=350, margin=dict(l=60, r=20, t=40, b=40)
        )
        return fig
    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots(); ax.bar(times, dP); ax.bar(times, dE, alpha=0.7); return fig
    return None

# IO utils

def export_preset_json(preset_name: str, bodies: List[Body], connections: List[Connection]) -> str:
    import json
    doc = {"name": preset_name, "bodies": [], "connections": []}
    for b in bodies:
        doc["bodies"].append({"name": b.name, "pos": b.pos.tolist(), "vel": b.vel.tolist(), "mass": b.mass, "charge": b.charge, "t0": b.t0, "dt": b.dt, "t_end": b.t_end, "radius": b.radius, "color": b.color})
    for c in connections:
        doc["connections"].append({"i": c.i, "j": c.j, "typ": c.typ, "strength": c.strength, "rest_length": c.rest_length})
    return json.dumps(doc, indent=2)

# Enhanced scenarios

def scenario_charged_pair(high_charge=1e-6, v=20.0):
    a = Body('A', pos=np.array([-0.5,0.0,0.0]), vel=np.array([v,0.0,0.0]), mass=1.0, t0=0.0, dt=0.001, t_end=2.0, charge=high_charge, color='red')
    b = Body('B', pos=np.array([0.5,0.0,0.0]), vel=np.array([-v,0.0,0.0]), mass=1.0, t0=0.0, dt=0.001, t_end=2.0, charge=-high_charge, color='blue')
    return [a,b], [], f"Geladenes Paar (±{high_charge} C)"

def scenario_three_charges_demo(scale_uC=1.0):
    q1,q2,q3 = +3.0e-6*scale_uC, -1.0e-6*scale_uC, -3.0e-6*scale_uC
    a = Body('Q1', pos=np.array([-1.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, t0=0.0, dt=0.001, t_end=2.0, charge=q1, color='red')
    b = Body('Q2', pos=np.array([0.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, t0=0.0, dt=0.001, t_end=2.0, charge=q2, color='blue')
    c = Body('Q3', pos=np.array([1.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, t0=0.0, dt=0.001, t_end=2.0, charge=q3, color='red')
    return [a,b,c], [], "Drei Ladungen (+3µC, -1µC, -3µC)"

def scenario_elastic_collision():
    a = Body('A', pos=np.array([-1.0,0.0,0.0]), vel=np.array([2.0,0.0,0.0]), mass=1.0, t0=0.0, dt=0.001, t_end=3.0, charge=0.0, color='red')
    b = Body('B', pos=np.array([1.0,0.0,0.0]), vel=np.array([-2.0,0.0,0.0]), mass=1.0, t0=0.0, dt=0.001, t_end=3.0, charge=0.0, color='blue')
    return [a,b], [], "Elastischer Stoß (e=1.0)"

def scenario_inelastic_collision():
    a = Body('A', pos=np.array([-1.0,0.0,0.0]), vel=np.array([2.0,0.0,0.0]), mass=2.0, t0=0.0, dt=0.001, t_end=3.0, charge=0.0, color='red')
    b = Body('B', pos=np.array([1.0,0.0,0.0]), vel=np.array([-1.0,0.0,0.0]), mass=1.0, t0=0.0, dt=0.001, t_end=3.0, charge=0.0, color='blue')
    return [a,b], [], "Inelastischer Stoß (e<1.0)"

def scenario_spring_system():
    a = Body('A', pos=np.array([-1.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, t0=0.0, dt=0.001, t_end=5.0, charge=0.0, color='red')
    b = Body('B', pos=np.array([1.0,0.0,0.0]), vel=np.zeros(3), mass=1.0, t0=0.0, dt=0.001, t_end=5.0, charge=0.0, color='blue')
    conn = [Connection(0,1,'elastic',10.0,2.0)]
    return [a,b], conn, "Federsystem"

def scenario_planetary_scaled(scale_mass=1e14, scale_distance=1.0, add_second=True, t_end=50.0, dt=0.001, pin_sun=False):
    M = float(scale_mass)
    sun = Body('Sun', pos=np.array([0.0,0.0,0.0]), vel=np.array([0.0,0.0,0.0]), mass=M, t0=0.0, dt=float(dt), t_end=float(t_end), charge=0.0, color='yellow')
    r1 = 1.0 * float(scale_distance); v1 = math.sqrt(G * M / r1)
    p1 = Body('P1', pos=np.array([r1,0.0,0.0]), vel=np.array([0.0,v1,0.0]), mass=1.0, t0=0.0, dt=float(dt), t_end=float(t_end), charge=0.0, color='blue')
    bodies = [sun, p1]
    if add_second:
        r2 = 1.6 * float(scale_distance); v2 = math.sqrt(G * M / r2)
        p2 = Body('P2', pos=np.array([r2,0.0,0.0]), vel=np.array([0.0,v2,0.0]), mass=0.5, t0=0.0, dt=float(dt), t_end=float(t_end), charge=0.0, color='green')
        bodies.append(p2)
    if not pin_sun:
        total_p = np.zeros(3)
        for b in bodies:
            total_p += b.mass * b.vel
        sun.vel = - total_p / sun.mass
    return bodies, [], f"Planetensystem (M={scale_mass})"

PRESETS_ENHANCED = {
    'Geladenes Paar': scenario_charged_pair,
    'Drei Ladungen': scenario_three_charges_demo,
    'Elastischer Stoß': scenario_elastic_collision,
    'Inelastischer Stoß': scenario_inelastic_collision,
    'Federsystem': scenario_spring_system,
    'Planetensystem': scenario_planetary_scaled,
}

# Simplified presets used elsewhere
PRESENTS_BLOCH = {
    "Standard": {"T1": 1000.0, "T2": 80.0, "TR": 800.0, "TE": 20.0},
    "Lang T1": {"T1": 2400.0, "T2": 120.0, "TR": 3000.0, "TE": 80.0},
}
PRESENTS_OPT_WAVE = {
    "Einzelspalt 532nm": {"N": 256, "lam": 532.0, "z": 1.0, "method": "Fraunhofer", "shape": "Einzelspalt", "w": 0.1, "p": 0.2, "incoh": 1.0},
    "Doppelspalt": {"N": 256, "lam": 532.0, "z": 1.0, "method": "Fraunhofer", "shape": "Doppelspalt", "w": 0.1, "p": 0.3, "incoh": 1.0}
}
PRESENTS_CT = {
    "Shepp-Logan Schnell": {"phantom": "Shepp-Logan", "N": 128, "geom": "Parallel", "ndet": 180, "nproj": 120, "kVp": 80.0, "filt": 2.5, "poly": False, "noise": 0.0, "budget": 4},
    "Knochen/Luft Scan": {"phantom": "Zylinder (Wasser/Knochen/Luft)", "N": 192, "geom": "Parallel", "ndet": 256, "nproj": 180, "kVp": 120.0, "filt": 5.0, "poly": True, "noise": 0.01, "budget": 6}
}
PRESENTS_MECH = {
    "2 Teilchenstoß": {"n": 2, "positions": [[-0.5, 0.2], [0.5, -0.2]], "velocities": [[0.5, 0.2], [-0.6, 0]], "t_end": 10.0, "dt": 0.02},
    "Kreisbewegung": {"n": 1, "positions": [[0.0, 1.0]], "velocities": [[-1.0, 0.0]], "t_end": 6.0, "dt": 0.01},
}
