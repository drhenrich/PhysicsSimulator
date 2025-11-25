"""
Mechanik-Modul (Komplett Neu)
- 2D-Mechanik: Projektilbewegung, Pendel, Federschwingungen, Schiefe Ebene
- 3D-Mehrkörpersimulation mit Gravitation und Coulomb-Kräften
- Himmelsmechanik: Sonnensystem, Kepler-Bahnen, Dreikörperproblem
- Stöße: Elastisch/Inelastisch in 1D und 2D mit Animationen
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PHYSICS CONSTANTS
# ============================================================

G = 6.67430e-11          # Gravitationskonstante [m³/(kg·s²)]
g_earth = 9.81           # Erdbeschleunigung [m/s²]
AU = 1.495978707e11      # Astronomische Einheit [m]
M_sun = 1.98847e30       # Sonnenmasse [kg]
M_earth = 5.972e24       # Erdmasse [kg]
DAY = 86400.0            # Sekunden pro Tag
YEAR = 365.25 * DAY      # Sekunden pro Jahr


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Body2D:
    """2D-Körper für Mechanik-Simulationen"""
    name: str
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    radius: float = 0.1
    color: str = "blue"
    trail_x: List[float] = field(default_factory=list)
    trail_y: List[float] = field(default_factory=list)
    
    def update_trail(self, max_length: int = 500):
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)
        if len(self.trail_x) > max_length:
            self.trail_x = self.trail_x[-max_length:]
            self.trail_y = self.trail_y[-max_length:]


@dataclass
class Body3D:
    """3D-Körper für Mehrkörpersimulationen"""
    name: str
    pos: np.ndarray
    vel: np.ndarray
    mass: float
    radius: float = 0.1
    color: str = "blue"
    charge: float = 0.0
    trail: List[np.ndarray] = field(default_factory=list)
    
    def update_trail(self, max_length: int = 500):
        self.trail.append(self.pos.copy())
        if len(self.trail) > max_length:
            self.trail = self.trail[-max_length:]


@dataclass
class CollisionEvent:
    """Kollisionsereignis"""
    time: float
    body1_name: str
    body2_name: str
    momentum_before: np.ndarray
    momentum_after: np.ndarray
    energy_before: float
    energy_after: float
    position: np.ndarray


# ============================================================
# 2D MECHANICS SIMULATIONS
# ============================================================

def projectile_motion(v0: float, angle_deg: float, h0: float = 0, 
                     g: float = g_earth, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schiefer Wurf ohne Luftwiderstand"""
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    t_flight = (vy0 + np.sqrt(vy0**2 + 2*g*h0)) / g
    t = np.arange(0, t_flight + dt, dt)
    x = vx0 * t
    y = h0 + vy0 * t - 0.5 * g * t**2
    mask = y >= 0
    return t[mask], x[mask], y[mask]


def projectile_with_drag(v0: float, angle_deg: float, h0: float = 0,
                         g: float = g_earth, drag_coeff: float = 0.1,
                         mass: float = 1.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schiefer Wurf mit Luftwiderstand"""
    angle_rad = np.radians(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x, y = 0.0, h0
    ts, xs, ys = [0], [x], [y]
    t = 0
    while y >= 0 and t < 100:
        v_mag = np.sqrt(vx**2 + vy**2)
        ax = -drag_coeff * v_mag * vx / mass
        ay = -g - drag_coeff * v_mag * vy / mass
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        ts.append(t)
        xs.append(x)
        ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)


def simple_pendulum(theta0_deg: float, L: float, g: float = g_earth,
                    t_max: float = 10, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Einfaches Pendel (numerisch, nicht-linear)"""
    theta = np.radians(theta0_deg)
    omega = 0.0
    ts, thetas, omegas = [0], [theta], [omega]
    t = 0
    while t < t_max:
        alpha = -(g / L) * np.sin(theta)
        omega += alpha * dt
        theta += omega * dt
        t += dt
        ts.append(t)
        thetas.append(theta)
        omegas.append(omega)
    return np.array(ts), np.array(thetas), np.array(omegas)


def coupled_pendulums(theta1_0: float, theta2_0: float, L: float, m: float,
                      k: float, g: float = g_earth, t_max: float = 20,
                      dt: float = 0.005) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zwei gekoppelte Pendel"""
    theta1, theta2 = np.radians(theta1_0), np.radians(theta2_0)
    omega1, omega2 = 0.0, 0.0
    ts, thetas1, thetas2 = [0], [theta1], [theta2]
    t = 0
    omega0_sq = g / L
    while t < t_max:
        coupling = k / (m * L) * (theta2 - theta1)
        alpha1 = -omega0_sq * np.sin(theta1) + coupling
        alpha2 = -omega0_sq * np.sin(theta2) - coupling
        omega1 += alpha1 * dt
        omega2 += alpha2 * dt
        theta1 += omega1 * dt
        theta2 += omega2 * dt
        t += dt
        ts.append(t)
        thetas1.append(theta1)
        thetas2.append(theta2)
    return np.array(ts), np.array(thetas1), np.array(thetas2)


def spring_oscillator(x0: float, v0: float, m: float, k: float,
                      damping: float = 0, t_max: float = 10,
                      dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gedämpfter harmonischer Oszillator"""
    x, v = x0, v0
    ts, xs, vs = [0], [x], [v]
    t = 0
    while t < t_max:
        a = (-k * x - damping * v) / m
        v += a * dt
        x += v * dt
        t += dt
        ts.append(t)
        xs.append(x)
        vs.append(v)
    return np.array(ts), np.array(xs), np.array(vs)


def inclined_plane(h: float, angle_deg: float, mu: float = 0,
                   g: float = g_earth, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bewegung auf schiefer Ebene"""
    angle_rad = np.radians(angle_deg)
    L = h / np.sin(angle_rad)
    a_eff = g * (np.sin(angle_rad) - mu * np.cos(angle_rad))
    if a_eff <= 0:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    s, v = 0.0, 0.0
    ts, ss, vs, accs = [0], [0], [0], [a_eff]
    t = 0
    while s < L and t < 100:
        v += a_eff * dt
        s += v * dt
        t += dt
        ts.append(t)
        ss.append(min(s, L))
        vs.append(v)
        accs.append(a_eff)
    return np.array(ts), np.array(ss), np.array(vs), np.array(accs)


# ============================================================
# 3D N-BODY SIMULATION
# ============================================================

class NBodySimulator:
    """N-Körper-Simulator mit Gravitation"""
    
    def __init__(self, bodies: List[Body3D], softening: float = 1e-9):
        self.bodies = bodies
        self.softening = softening
        self.collision_events: List[CollisionEvent] = []
        self.time = 0.0
    
    def compute_accelerations(self) -> List[np.ndarray]:
        n = len(self.bodies)
        accelerations = [np.zeros(3) for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = self.bodies[i], self.bodies[j]
                r_vec = bj.pos - bi.pos
                r = np.linalg.norm(r_vec) + self.softening
                r_hat = r_vec / r
                F_grav = G * bi.mass * bj.mass / r**2
                k_e = 8.9875517923e9
                F_coulomb = k_e * bi.charge * bj.charge / r**2
                F_total = (F_grav - F_coulomb) * r_hat
                accelerations[i] += F_total / bi.mass
                accelerations[j] -= F_total / bj.mass
        return accelerations
    
    def detect_collision(self, i: int, j: int) -> bool:
        bi, bj = self.bodies[i], self.bodies[j]
        dist = np.linalg.norm(bi.pos - bj.pos)
        return dist < (bi.radius + bj.radius)
    
    def resolve_collision(self, i: int, j: int, restitution: float = 1.0):
        bi, bj = self.bodies[i], self.bodies[j]
        p_before = bi.mass * bi.vel + bj.mass * bj.vel
        E_before = 0.5 * bi.mass * np.dot(bi.vel, bi.vel) + 0.5 * bj.mass * np.dot(bj.vel, bj.vel)
        n = (bj.pos - bi.pos)
        n = n / (np.linalg.norm(n) + 1e-12)
        v_rel = bi.vel - bj.vel
        v_rel_n = np.dot(v_rel, n)
        if v_rel_n > 0:
            return
        j_mag = -(1 + restitution) * v_rel_n / (1/bi.mass + 1/bj.mass)
        bi.vel += (j_mag / bi.mass) * n
        bj.vel -= (j_mag / bj.mass) * n
        overlap = (bi.radius + bj.radius) - np.linalg.norm(bj.pos - bi.pos)
        if overlap > 0:
            bi.pos -= 0.5 * overlap * n
            bj.pos += 0.5 * overlap * n
        p_after = bi.mass * bi.vel + bj.mass * bj.vel
        E_after = 0.5 * bi.mass * np.dot(bi.vel, bi.vel) + 0.5 * bj.mass * np.dot(bj.vel, bj.vel)
        self.collision_events.append(CollisionEvent(
            time=self.time, body1_name=bi.name, body2_name=bj.name,
            momentum_before=p_before, momentum_after=p_after,
            energy_before=E_before, energy_after=E_after,
            position=(bi.pos + bj.pos) / 2
        ))
    
    def step(self, dt: float, restitution: float = 1.0):
        n = len(self.bodies)
        acc_old = self.compute_accelerations()
        for i, b in enumerate(self.bodies):
            b.pos += b.vel * dt + 0.5 * acc_old[i] * dt**2
        acc_new = self.compute_accelerations()
        for i, b in enumerate(self.bodies):
            b.vel += 0.5 * (acc_old[i] + acc_new[i]) * dt
        for i in range(n):
            for j in range(i + 1, n):
                if self.detect_collision(i, j):
                    self.resolve_collision(i, j, restitution)
        for b in self.bodies:
            b.update_trail()
        self.time += dt
    
    def run(self, t_end: float, dt: float, restitution: float = 1.0, record_interval: int = 10) -> Dict:
        times, energies = [], []
        positions = {b.name: [] for b in self.bodies}
        velocities = {b.name: [] for b in self.bodies}
        step_count = 0
        while self.time < t_end:
            self.step(dt, restitution)
            if step_count % record_interval == 0:
                times.append(self.time)
                for b in self.bodies:
                    positions[b.name].append(b.pos.copy())
                    velocities[b.name].append(b.vel.copy())
                E_total = sum(0.5 * b.mass * np.dot(b.vel, b.vel) for b in self.bodies)
                energies.append(E_total)
            step_count += 1
        return {
            'times': np.array(times),
            'positions': {k: np.array(v) for k, v in positions.items()},
            'velocities': {k: np.array(v) for k, v in velocities.items()},
            'energies': np.array(energies),
            'collisions': self.collision_events
        }


# ============================================================
# SOLAR SYSTEM
# ============================================================

SOLAR_SYSTEM_DATA = {
    "Sonne": {"mass": M_sun, "distance": 0, "velocity": 0, "color": "#FFD700", "radius": 6.96e8},
    "Merkur": {"mass": 3.285e23, "distance": 0.387 * AU, "velocity": 47870, "color": "#A0522D", "radius": 2.44e6},
    "Venus": {"mass": 4.867e24, "distance": 0.723 * AU, "velocity": 35020, "color": "#DEB887", "radius": 6.05e6},
    "Erde": {"mass": M_earth, "distance": AU, "velocity": 29780, "color": "#4169E1", "radius": 6.37e6},
    "Mars": {"mass": 6.39e23, "distance": 1.524 * AU, "velocity": 24130, "color": "#CD5C5C", "radius": 3.39e6},
    "Jupiter": {"mass": 1.898e27, "distance": 5.203 * AU, "velocity": 13070, "color": "#F4A460", "radius": 6.99e7},
    "Saturn": {"mass": 5.683e26, "distance": 9.537 * AU, "velocity": 9690, "color": "#DAA520", "radius": 5.82e7},
}


def create_solar_system(planets: List[str]) -> List[Body3D]:
    bodies = []
    for name in planets:
        if name not in SOLAR_SYSTEM_DATA:
            continue
        data = SOLAR_SYSTEM_DATA[name]
        pos = np.array([data["distance"], 0.0, 0.0])
        vel = np.array([0.0, data["velocity"], 0.0])
        bodies.append(Body3D(name=name, pos=pos, vel=vel, mass=data["mass"],
                            radius=data["radius"], color=data["color"]))
    if len(bodies) > 1:
        total_momentum = sum(b.mass * b.vel for b in bodies[1:])
        bodies[0].vel = -total_momentum / bodies[0].mass
    return bodies


def three_body_figure_eight() -> List[Body3D]:
    """Spezielle Figure-8 Lösung"""
    x1 = np.array([0.97000436, -0.24308753, 0.0])
    x2 = np.array([-0.97000436, 0.24308753, 0.0])
    x3 = np.array([0.0, 0.0, 0.0])
    v3 = np.array([0.93240737, 0.86473146, 0.0])
    v1 = -v3 / 2
    v2 = -v3 / 2
    return [
        Body3D("Körper 1", x1, v1, 1.0, radius=0.05, color="red"),
        Body3D("Körper 2", x2, v2, 1.0, radius=0.05, color="green"),
        Body3D("Körper 3", x3, v3, 1.0, radius=0.05, color="blue"),
    ]


def lagrange_points_system(m1: float = 1.0, m2: float = 0.01, d: float = 1.0) -> Dict:
    """Berechne Lagrange-Punkte"""
    mu = m2 / (m1 + m2)
    cm = mu * d
    r_hill = d * (mu / 3) ** (1/3)
    L1 = np.array([d - r_hill - cm, 0, 0])
    L2 = np.array([d + r_hill - cm, 0, 0])
    L3 = np.array([-d - cm, 0, 0])
    L4 = np.array([d/2 - cm, d * np.sqrt(3)/2, 0])
    L5 = np.array([d/2 - cm, -d * np.sqrt(3)/2, 0])
    return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'L5': L5, 'mu': mu, 'cm': cm}


# ============================================================
# COLLISION PHYSICS
# ============================================================

def inelastic_collision_1d(m1: float, v1: float, m2: float, v2: float,
                           restitution: float = 0.0) -> Tuple[float, float]:
    """Inelastischer Stoß in 1D"""
    v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)
    v_rel_before = v1 - v2
    v_rel_after = -restitution * v_rel_before
    v1_new = v_cm + m2 / (m1 + m2) * v_rel_after
    v2_new = v_cm - m1 / (m1 + m2) * v_rel_after
    return v1_new, v2_new


def collision_2d(b1: Body2D, b2: Body2D, restitution: float = 1.0) -> Tuple[Body2D, Body2D]:
    """2D-Kollision zwischen zwei Kreisen"""
    nx = b2.x - b1.x
    ny = b2.y - b1.y
    d = np.sqrt(nx**2 + ny**2)
    if d == 0:
        return b1, b2
    nx, ny = nx/d, ny/d
    tx, ty = -ny, nx
    v1n = b1.vx * nx + b1.vy * ny
    v1t = b1.vx * tx + b1.vy * ty
    v2n = b2.vx * nx + b2.vy * ny
    v2t = b2.vx * tx + b2.vy * ty
    m1, m2 = b1.mass, b2.mass
    v1n_new = ((m1 - restitution * m2) * v1n + (1 + restitution) * m2 * v2n) / (m1 + m2)
    v2n_new = ((m2 - restitution * m1) * v2n + (1 + restitution) * m1 * v1n) / (m1 + m2)
    b1.vx = v1n_new * nx + v1t * tx
    b1.vy = v1n_new * ny + v1t * ty
    b2.vx = v2n_new * nx + v2t * tx
    b2.vy = v2n_new * ny + v2t * ty
    overlap = (b1.radius + b2.radius) - d
    if overlap > 0:
        b1.x -= 0.5 * overlap * nx
        b1.y -= 0.5 * overlap * ny
        b2.x += 0.5 * overlap * nx
        b2.y += 0.5 * overlap * ny
    return b1, b2


# ============================================================
# STREAMLIT UI - MAIN
# ============================================================

def render_mech_astro_tab():
    """Hauptfunktion für den Mechanik-Tab (Alias für Kompatibilität)"""
    render_mechanics_tab()


def render_mechanics_tab():
    """Hauptfunktion für den Mechanik-Tab"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("🚀 Mechanik & Himmelsmechanik", "🚀 Mechanics & Celestial Mechanics"))
    
    mech2d_tab, mech3d_tab, astro_tab, collision_tab = st.tabs([
        tr("2D-Mechanik", "2D Mechanics"),
        tr("3D-Mehrkörper", "3D N-Body"),
        tr("Himmelsmechanik", "Celestial Mechanics"),
        tr("Stöße & Kollisionen", "Collisions")
    ])
    
    with mech2d_tab:
        render_2d_mechanics_tab()
    with mech3d_tab:
        render_3d_nbody_tab()
    with astro_tab:
        render_celestial_tab()
    with collision_tab:
        render_collisions_tab()


def render_2d_mechanics_tab():
    """2D-Mechanik Simulationen"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    simulation_type = st.selectbox(
        tr("Simulation wählen", "Select simulation"),
        [tr("Schiefer Wurf", "Projectile Motion"), tr("Einfaches Pendel", "Simple Pendulum"),
         tr("Gekoppelte Pendel", "Coupled Pendulums"), tr("Federschwingung", "Spring Oscillator"),
         tr("Schiefe Ebene", "Inclined Plane")],
        key="mech2d_type"
    )
    
    st.markdown("---")
    
    if "Wurf" in simulation_type or "Projectile" in simulation_type:
        render_projectile_ui()
    elif "Einfaches Pendel" in simulation_type or "Simple Pendulum" in simulation_type:
        render_pendulum_ui()
    elif "Gekoppelte" in simulation_type or "Coupled" in simulation_type:
        render_coupled_pendulum_ui()
    elif "Feder" in simulation_type or "Spring" in simulation_type:
        render_spring_ui()
    elif "Schiefe Ebene" in simulation_type or "Inclined" in simulation_type:
        render_inclined_plane_ui()


def render_projectile_ui():
    """Schiefer Wurf UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Schiefer Wurf", "### Projectile Motion"))
    st.latex(r"x(t) = v_0 \cos(\alpha) \cdot t, \quad y(t) = h_0 + v_0 \sin(\alpha) \cdot t - \frac{1}{2}gt^2")
    
    col1, col2 = st.columns(2)
    with col1:
        v0 = st.slider(tr("Anfangsgeschwindigkeit v₀ [m/s]", "Initial velocity v₀ [m/s]"), 5.0, 50.0, 20.0, 1.0, key="proj_v0")
        angle = st.slider(tr("Abwurfwinkel α [°]", "Launch angle α [°]"), 5, 85, 45, 1, key="proj_angle")
    with col2:
        h0 = st.slider(tr("Anfangshöhe h₀ [m]", "Initial height h₀ [m]"), 0.0, 20.0, 0.0, 0.5, key="proj_h0")
        drag = st.slider(tr("Luftwiderstand", "Air drag"), 0.0, 0.5, 0.0, 0.01, key="proj_drag")
    
    animate = st.checkbox(tr("Animation", "Animation"), key="proj_animate")
    
    if st.button(tr("▶️ Simulation starten", "▶️ Start simulation"), key="proj_run", use_container_width=True):
        if drag > 0:
            t, x, y = projectile_with_drag(v0, angle, h0, drag_coeff=drag)
        else:
            t, x, y = projectile_motion(v0, angle, h0)
        
        if animate:
            run_projectile_animation(t, x, y, v0, angle)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=tr("Trajektorie", "Trajectory"), line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=[0], y=[h0], mode='markers', name=tr("Start", "Start"), marker=dict(size=12, color='green')))
            fig.add_trace(go.Scatter(x=[x[-1]], y=[0], mode='markers', name=tr("Aufprall", "Impact"), marker=dict(size=12, color='red')))
            fig.update_layout(title=tr("Wurfparabel", "Projectile Path"), xaxis_title="x [m]", yaxis_title="y [m]", height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(tr("Wurfweite", "Range"), f"{x[-1]:.2f} m")
        with col2:
            st.metric(tr("Flugzeit", "Flight time"), f"{t[-1]:.2f} s")
        with col3:
            st.metric(tr("Max. Höhe", "Max height"), f"{max(y):.2f} m")


def render_pendulum_ui():
    """Einfaches Pendel UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Mathematisches Pendel", "### Simple Pendulum"))
    st.latex(r"\ddot{\theta} = -\frac{g}{L} \sin(\theta)")
    
    col1, col2 = st.columns(2)
    with col1:
        theta0 = st.slider(tr("Anfangsauslenkung θ₀ [°]", "Initial angle θ₀ [°]"), 5, 170, 30, 5, key="pend_theta0")
        L = st.slider(tr("Pendellänge L [m]", "Pendulum length L [m]"), 0.5, 5.0, 1.0, 0.1, key="pend_L")
    with col2:
        t_max = st.slider(tr("Simulationszeit [s]", "Simulation time [s]"), 5, 30, 10, 1, key="pend_tmax")
    
    animate = st.checkbox(tr("Animation", "Animation"), key="pend_animate")
    
    if st.button(tr("▶️ Pendel simulieren", "▶️ Simulate pendulum"), key="pend_run", use_container_width=True):
        t, theta, omega = simple_pendulum(theta0, L, t_max=t_max)
        
        if animate:
            run_pendulum_animation(t, theta, L)
        else:
            fig = make_subplots(rows=1, cols=2, subplot_titles=[tr("Winkel θ(t)", "Angle θ(t)"), tr("Phasenraum", "Phase Space")])
            fig.add_trace(go.Scatter(x=t, y=np.degrees(theta), mode='lines', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=np.degrees(theta), y=omega, mode='lines', line=dict(color='red')), row=1, col=2)
            fig.update_xaxes(title_text="t [s]", row=1, col=1)
            fig.update_yaxes(title_text="θ [°]", row=1, col=1)
            fig.update_xaxes(title_text="θ [°]", row=1, col=2)
            fig.update_yaxes(title_text="ω [rad/s]", row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        T_small = 2 * np.pi * np.sqrt(L / g_earth)
        st.metric(tr("Schwingungsdauer (kleine Winkel)", "Period (small angles)"), f"{T_small:.3f} s")


def render_coupled_pendulum_ui():
    """Gekoppelte Pendel UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Gekoppelte Pendel", "### Coupled Pendulums"))
    
    col1, col2 = st.columns(2)
    with col1:
        theta1_0 = st.slider(tr("Pendel 1: θ₁ [°]", "Pendulum 1: θ₁ [°]"), 0, 45, 20, 1, key="coup_theta1")
        theta2_0 = st.slider(tr("Pendel 2: θ₂ [°]", "Pendulum 2: θ₂ [°]"), 0, 45, 0, 1, key="coup_theta2")
    with col2:
        L = st.slider(tr("Pendellänge L [m]", "Length L [m]"), 0.5, 2.0, 1.0, 0.1, key="coup_L")
        k = st.slider(tr("Kopplungsstärke k", "Coupling strength k"), 0.1, 5.0, 1.0, 0.1, key="coup_k")
    
    if st.button(tr("▶️ Simulation starten", "▶️ Start simulation"), key="coup_run", use_container_width=True):
        t, theta1, theta2 = coupled_pendulums(theta1_0, theta2_0, L, 1.0, k, t_max=20)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=np.degrees(theta1), mode='lines', name=tr("Pendel 1", "Pendulum 1"), line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=t, y=np.degrees(theta2), mode='lines', name=tr("Pendel 2", "Pendulum 2"), line=dict(color='red')))
        fig.update_layout(title=tr("Gekoppelte Schwingung", "Coupled Oscillation"), xaxis_title="t [s]", yaxis_title="θ [°]", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_spring_ui():
    """Federschwingung UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Gedämpfter harmonischer Oszillator", "### Damped Harmonic Oscillator"))
    st.latex(r"m\ddot{x} + c\dot{x} + kx = 0")
    
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.slider(tr("Anfangsauslenkung x₀ [m]", "Initial displacement x₀ [m]"), 0.1, 2.0, 1.0, 0.1, key="spring_x0")
        k = st.slider(tr("Federkonstante k [N/m]", "Spring constant k [N/m]"), 1.0, 50.0, 10.0, 1.0, key="spring_k")
    with col2:
        m = st.slider(tr("Masse m [kg]", "Mass m [kg]"), 0.1, 5.0, 1.0, 0.1, key="spring_m")
        damping = st.slider(tr("Dämpfung c [Ns/m]", "Damping c [Ns/m]"), 0.0, 5.0, 0.5, 0.1, key="spring_c")
    
    if st.button(tr("▶️ Oszillator simulieren", "▶️ Simulate oscillator"), key="spring_run", use_container_width=True):
        t, x, v = spring_oscillator(x0, 0, m, k, damping, t_max=15)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[tr("Position x(t)", "Position x(t)"), tr("Energie", "Energy")])
        fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='x(t)', line=dict(color='blue')), row=1, col=1)
        
        E_kin = 0.5 * m * v**2
        E_pot = 0.5 * k * x**2
        E_total = E_kin + E_pot
        
        fig.add_trace(go.Scatter(x=t, y=E_kin, mode='lines', name='E_kin', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=E_pot, mode='lines', name='E_pot', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=E_total, mode='lines', name='E_total', line=dict(color='black', dash='dash')), row=2, col=1)
        
        fig.update_xaxes(title_text="t [s]", row=2, col=1)
        fig.update_yaxes(title_text="x [m]", row=1, col=1)
        fig.update_yaxes(title_text="E [J]", row=2, col=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        omega0 = np.sqrt(k / m)
        T = 2 * np.pi / omega0
        damping_ratio = damping / (2 * np.sqrt(k * m))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(tr("Eigenfrequenz ω₀", "Natural freq. ω₀"), f"{omega0:.3f} rad/s")
        with col2:
            st.metric(tr("Periode T", "Period T"), f"{T:.3f} s")
        with col3:
            regime = tr("Unterdämpft", "Underdamped") if damping_ratio < 1 else (tr("Kritisch", "Critical") if damping_ratio == 1 else tr("Überdämpft", "Overdamped"))
            st.metric(tr("Dämpfungsregime", "Damping regime"), regime)


def render_inclined_plane_ui():
    """Schiefe Ebene UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Bewegung auf schiefer Ebene", "### Motion on Inclined Plane"))
    st.latex(r"a = g(\sin\alpha - \mu \cos\alpha)")
    
    col1, col2 = st.columns(2)
    with col1:
        h = st.slider(tr("Höhe h [m]", "Height h [m]"), 1.0, 10.0, 5.0, 0.5, key="incl_h")
        angle = st.slider(tr("Neigungswinkel α [°]", "Angle α [°]"), 10, 80, 30, 1, key="incl_angle")
    with col2:
        mu = st.slider(tr("Reibungskoeffizient μ", "Friction coefficient μ"), 0.0, 0.8, 0.0, 0.05, key="incl_mu")
    
    if st.button(tr("▶️ Simulation starten", "▶️ Start simulation"), key="incl_run", use_container_width=True):
        t, s, v, a = inclined_plane(h, angle, mu)
        
        if len(t) <= 1:
            st.error(tr("Keine Bewegung! Haftreibung zu groß.", "No motion! Static friction too high."))
            st.info(tr(f"Kritischer Winkel: {np.degrees(np.arctan(mu)):.1f}°", f"Critical angle: {np.degrees(np.arctan(mu)):.1f}°"))
        else:
            fig = make_subplots(rows=1, cols=2, subplot_titles=[tr("Weg s(t)", "Distance s(t)"), tr("Geschwindigkeit v(t)", "Velocity v(t)")])
            fig.add_trace(go.Scatter(x=t, y=s, mode='lines', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=v, mode='lines', line=dict(color='red')), row=1, col=2)
            fig.update_xaxes(title_text="t [s]", row=1, col=1)
            fig.update_xaxes(title_text="t [s]", row=1, col=2)
            fig.update_yaxes(title_text="s [m]", row=1, col=1)
            fig.update_yaxes(title_text="v [m/s]", row=1, col=2)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(tr("Beschleunigung a", "Acceleration a"), f"{a[0]:.2f} m/s²")
            with col2:
                st.metric(tr("Endgeschwindigkeit", "Final velocity"), f"{v[-1]:.2f} m/s")
            with col3:
                st.metric(tr("Rutschzeit", "Slide time"), f"{t[-1]:.2f} s")


def run_projectile_animation(t: np.ndarray, x: np.ndarray, y: np.ndarray, v0: float, angle: float):
    """Animation des Wurfs mit Plotly Frames"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    n_frames = min(80, len(t))
    step = max(1, len(t) // n_frames)
    
    # Frames vorberechnen
    frames = []
    for frame_idx in range(n_frames):
        i = min(frame_idx * step, len(t) - 1)
        
        vx = v0 * np.cos(np.radians(angle))
        vy = v0 * np.sin(np.radians(angle)) - g_earth * t[i]
        
        frame_data = [
            go.Scatter(x=x[:i+1], y=y[:i+1], mode='lines', 
                      line=dict(color='lightblue', width=2), showlegend=False),
            go.Scatter(x=[x[i]], y=[y[i]], mode='markers', 
                      marker=dict(size=15, color='blue'), showlegend=False)
        ]
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    fig = go.Figure(
        data=[
            go.Scatter(x=[x[0]], y=[y[0]], mode='markers',
                      marker=dict(size=15, color='blue'), name=tr("Objekt", "Object"))
        ],
        frames=frames
    )
    
    fig.update_layout(
        title=tr("Schiefer Wurf", "Projectile Motion"),
        xaxis=dict(range=[-1, max(x)*1.1], title="x [m]"),
        yaxis=dict(range=[-0.5, max(y)*1.2], title="y [m]"),
        height=400,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                dict(label="🔄 Reset", method="animate",
                     args=[["0"], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def run_pendulum_animation(t: np.ndarray, theta: np.ndarray, L: float):
    """Animation des Pendels mit Plotly Frames"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    n_frames = min(120, len(t))
    step = max(1, len(t) // n_frames)
    
    # Frames vorberechnen
    frames = []
    for frame_idx in range(n_frames):
        i = min(frame_idx * step, len(t) - 1)
        x_bob = L * np.sin(theta[i])
        y_bob = -L * np.cos(theta[i])
        
        frame_data = [
            go.Scatter(x=[0], y=[0], mode='markers', 
                      marker=dict(size=10, color='black'), showlegend=False),
            go.Scatter(x=[0, x_bob], y=[0, y_bob], mode='lines', 
                      line=dict(color='gray', width=3), showlegend=False),
            go.Scatter(x=[x_bob], y=[y_bob], mode='markers', 
                      marker=dict(size=25, color='blue'), showlegend=False)
        ]
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    # Initial
    x0 = L * np.sin(theta[0])
    y0 = -L * np.cos(theta[0])
    
    fig = go.Figure(
        data=[
            go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=10, color='black'), showlegend=False),
            go.Scatter(x=[0, x0], y=[0, y0], mode='lines', line=dict(color='gray', width=3), showlegend=False),
            go.Scatter(x=[x0], y=[y0], mode='markers', marker=dict(size=25, color='blue'), name=tr("Pendel", "Pendulum"))
        ],
        frames=frames
    )
    
    fig.update_layout(
        title=tr("Pendelbewegung", "Pendulum Motion"),
        xaxis=dict(range=[-L*1.3, L*1.3], visible=False),
        yaxis=dict(range=[-L*1.3, L*0.3], visible=False, scaleanchor="x"),
        height=400,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 30, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                dict(label="🔄 Reset", method="animate",
                     args=[["0"], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_3d_nbody_tab():
    """3D-Mehrkörpersimulation"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### 3D N-Körper-Simulation", "### 3D N-Body Simulation"))
    
    preset = st.selectbox(tr("Preset wählen", "Select preset"),
        [tr("2 Körper (Orbit)", "2 Bodies (Orbit)"), tr("3 Körper (chaotisch)", "3 Bodies (chaotic)"),
         tr("Figure-8 Lösung", "Figure-8 Solution"), tr("Elastische Stöße (4 Kugeln)", "Elastic Collisions (4 spheres)")],
        key="nbody_preset")
    
    col1, col2 = st.columns(2)
    with col1:
        t_end = st.slider(tr("Simulationszeit", "Simulation time"), 1.0, 50.0, 10.0, 1.0, key="nbody_tend")
        dt = st.select_slider(tr("Zeitschritt dt", "Timestep dt"), options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, key="nbody_dt")
    with col2:
        restitution = st.slider(tr("Stoßzahl (e)", "Restitution (e)"), 0.0, 1.0, 1.0, 0.05, key="nbody_rest")
    
    if st.button(tr("▶️ Simulation starten", "▶️ Start simulation"), key="nbody_run", use_container_width=True):
        if "2 Körper" in preset or "2 Bodies" in preset:
            M, m, r = 1e6, 1.0, 1.0
            v_orbit = np.sqrt(G * M / r)
            bodies = [
                Body3D("Zentralkörper", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), M, radius=0.15, color="yellow"),
                Body3D("Satellit", np.array([r, 0.0, 0.0]), np.array([0.0, v_orbit, 0.0]), m, radius=0.05, color="blue")
            ]
            bodies[0].vel = -bodies[1].mass * bodies[1].vel / bodies[0].mass
        elif "3 Körper" in preset or "3 Bodies" in preset:
            M = 1e5
            bodies = [
                Body3D("A", np.array([0.0, 1.0, 0.0]), np.array([0.5, 0.0, 0.0]), M, radius=0.08, color="red"),
                Body3D("B", np.array([0.87, -0.5, 0.0]), np.array([-0.25, 0.43, 0.0]), M, radius=0.08, color="green"),
                Body3D("C", np.array([-0.87, -0.5, 0.0]), np.array([-0.25, -0.43, 0.0]), M, radius=0.08, color="blue")
            ]
        elif "Figure-8" in preset:
            bodies = three_body_figure_eight()
            for b in bodies:
                b.mass = 1e6
        else:
            bodies = [
                Body3D("A", np.array([-1.0, 0.0, 0.0]), np.array([0.5, 0.1, 0.0]), 1.0, radius=0.15, color="red"),
                Body3D("B", np.array([1.0, 0.0, 0.0]), np.array([-0.5, -0.1, 0.0]), 1.0, radius=0.15, color="blue"),
                Body3D("C", np.array([0.0, 1.0, 0.0]), np.array([0.1, -0.5, 0.0]), 1.0, radius=0.15, color="green"),
                Body3D("D", np.array([0.0, -1.0, 0.0]), np.array([-0.1, 0.5, 0.0]), 1.0, radius=0.15, color="orange")
            ]
        
        sim = NBodySimulator(bodies)
        with st.spinner(tr("Berechne Trajektorien...", "Computing trajectories...")):
            results = sim.run(t_end, dt, restitution)
        
        fig = go.Figure()
        for b in bodies:
            if b.name in results['positions'] and len(results['positions'][b.name]) > 0:
                pos = results['positions'][b.name]
                fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='lines', line=dict(color=b.color, width=2), name=b.name))
                fig.add_trace(go.Scatter3d(x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]], mode='markers', marker=dict(size=8, color=b.color), showlegend=False))
        
        fig.update_layout(title=tr("3D-Trajektorien", "3D Trajectories"), scene=dict(aspectmode='cube'), height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        if len(results['energies']) > 0:
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(x=results['times'], y=results['energies'], mode='lines', name=tr("Gesamtenergie", "Total Energy")))
            fig_energy.update_layout(title=tr("Energieerhaltung", "Energy Conservation"), xaxis_title="t", yaxis_title="E [J]", height=300)
            st.plotly_chart(fig_energy, use_container_width=True)


def render_celestial_tab():
    """Himmelsmechanik"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Himmelsmechanik", "### Celestial Mechanics"))
    
    sim_type = st.selectbox(tr("Simulation wählen", "Select simulation"),
        [tr("Sonnensystem (innere Planeten)", "Solar System (inner planets)"),
         tr("Sonnensystem (alle Planeten)", "Solar System (all planets)"),
         tr("Kepler-Bahnen (Exzentrizität)", "Kepler Orbits (Eccentricity)"),
         tr("Lagrange-Punkte", "Lagrange Points")],
        key="celestial_type")
    
    if "Sonnensystem" in sim_type or "Solar System" in sim_type:
        if "inner" in sim_type.lower() or "innere" in sim_type.lower():
            planets = ["Sonne", "Merkur", "Venus", "Erde", "Mars"]
            default_years = 2
        else:
            planets = ["Sonne", "Merkur", "Venus", "Erde", "Mars", "Jupiter", "Saturn"]
            default_years = 12
        
        col1, col2 = st.columns(2)
        with col1:
            sim_years = st.slider(tr("Simulationsdauer [Jahre]", "Duration [years]"), 0.5, 30.0, float(default_years), 0.5, key="solar_years")
        with col2:
            dt_days = st.slider(tr("Zeitschritt [Tage]", "Timestep [days]"), 0.1, 5.0, 1.0, 0.1, key="solar_dt")
        
        if st.button(tr("▶️ Sonnensystem simulieren", "▶️ Simulate Solar System"), key="solar_run", use_container_width=True):
            bodies = create_solar_system(planets)
            sim = NBodySimulator(bodies, softening=1e6)
            t_end, dt = sim_years * YEAR, dt_days * DAY
            
            with st.spinner(tr("Berechne Planetenbahnen...", "Computing planetary orbits...")):
                results = sim.run(t_end, dt, record_interval=max(1, int(10 / dt_days)))
            
            fig = go.Figure()
            for b in bodies:
                if b.name in results['positions']:
                    pos = results['positions'][b.name]
                    x_au, y_au = pos[:, 0] / AU, pos[:, 1] / AU
                    fig.add_trace(go.Scatter(x=x_au, y=y_au, mode='lines', line=dict(color=b.color, width=2), name=b.name))
                    fig.add_trace(go.Scatter(x=[x_au[-1]], y=[y_au[-1]], mode='markers', marker=dict(size=10 if b.name == "Sonne" else 6, color=b.color), showlegend=False))
            
            max_dist = max(SOLAR_SYSTEM_DATA[p]["distance"] for p in planets) / AU * 1.1
            fig.update_layout(title=tr(f"Sonnensystem ({sim_years} Jahre)", f"Solar System ({sim_years} years)"),
                             xaxis=dict(title="x [AU]", range=[-max_dist, max_dist], scaleanchor="y"),
                             yaxis=dict(title="y [AU]", range=[-max_dist, max_dist]), height=550)
            st.plotly_chart(fig, use_container_width=True)
    
    elif "Kepler" in sim_type:
        st.latex(r"r = \frac{a(1-e^2)}{1 + e\cos\theta}")
        col1, col2 = st.columns(2)
        with col1:
            a = st.slider(tr("Große Halbachse a [AU]", "Semi-major axis a [AU]"), 0.5, 5.0, 1.0, 0.1, key="kepler_a")
        with col2:
            e = st.slider(tr("Exzentrizität e", "Eccentricity e"), 0.0, 0.95, 0.0, 0.05, key="kepler_e")
        
        if st.button(tr("📊 Kepler-Bahn berechnen", "📊 Compute Kepler orbit"), key="kepler_run", use_container_width=True):
            theta = np.linspace(0, 2*np.pi, 500)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            x, y = r * np.cos(theta), r * np.sin(theta)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='blue', width=2), name=tr("Orbit", "Orbit")))
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=15, color='yellow'), name=tr("Sonne", "Sun")))
            
            r_peri, r_aph = a * (1 - e), a * (1 + e)
            fig.add_trace(go.Scatter(x=[r_peri, -r_aph], y=[0, 0], mode='markers+text', marker=dict(size=8, color=['green', 'red']),
                                    text=[tr("Perihel", "Perihelion"), tr("Aphel", "Aphelion")], textposition="top center", showlegend=False))
            
            max_r = r_aph * 1.1
            fig.update_layout(title=tr(f"Kepler-Bahn (a={a} AU, e={e})", f"Kepler Orbit (a={a} AU, e={e})"),
                             xaxis=dict(title="x [AU]", range=[-max_r, max_r], scaleanchor="y"),
                             yaxis=dict(title="y [AU]", range=[-max_r, max_r]), height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            T = np.sqrt(a**3)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(tr("Umlaufzeit", "Orbital period"), f"{T:.2f} Jahre")
            with col2:
                st.metric(tr("Perihel", "Perihelion"), f"{r_peri:.2f} AU")
            with col3:
                st.metric(tr("Aphel", "Aphelion"), f"{r_aph:.2f} AU")
    
    elif "Lagrange" in sim_type:
        st.markdown(tr("Gleichgewichtspunkte im Dreikörperproblem.", "Equilibrium points in the three-body problem."))
        mass_ratio = st.slider(tr("Massenverhältnis m₂/m₁", "Mass ratio m₂/m₁"), 0.001, 0.1, 0.01, 0.001, key="lagrange_mu")
        
        if st.button(tr("📊 Lagrange-Punkte berechnen", "📊 Compute Lagrange points"), key="lagrange_run", use_container_width=True):
            lp = lagrange_points_system(m1=1.0, m2=mass_ratio, d=1.0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[-lp['cm']], y=[0], mode='markers+text', marker=dict(size=20, color='yellow'), text=["M₁"], textposition="bottom center", name="M₁"))
            fig.add_trace(go.Scatter(x=[1 - lp['cm']], y=[0], mode='markers+text', marker=dict(size=10, color='blue'), text=["M₂"], textposition="bottom center", name="M₂"))
            
            colors = {'L1': 'red', 'L2': 'orange', 'L3': 'purple', 'L4': 'green', 'L5': 'green'}
            for name, pos in [('L1', lp['L1']), ('L2', lp['L2']), ('L3', lp['L3']), ('L4', lp['L4']), ('L5', lp['L5'])]:
                stability = tr("stabil", "stable") if name in ['L4', 'L5'] else tr("instabil", "unstable")
                fig.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text', marker=dict(size=12, color=colors[name], symbol='diamond'),
                                        text=[f"{name} ({stability})"], textposition="top center", name=name))
            
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(x=np.cos(theta) - lp['cm'], y=np.sin(theta), mode='lines', line=dict(color='lightblue', dash='dot'), name=tr("Orbit M₂", "Orbit M₂")))
            
            fig.update_layout(title=tr(f"Lagrange-Punkte (μ = {mass_ratio:.3f})", f"Lagrange Points (μ = {mass_ratio:.3f})"),
                             xaxis=dict(title="x", range=[-1.5, 1.5], scaleanchor="y"), yaxis=dict(title="y", range=[-1.5, 1.5]), height=550)
            st.plotly_chart(fig, use_container_width=True)


def render_collisions_tab():
    """Stöße und Kollisionen"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("### Stöße & Kollisionen", "### Collisions"))
    
    collision_type = st.selectbox(tr("Stoßtyp wählen", "Select collision type"),
        [tr("1D Zentralstoß", "1D Central Collision"), tr("2D Stoß (zwei Kugeln)", "2D Collision (two spheres)"),
         tr("2D Billard-Simulation", "2D Billiard Simulation"), tr("Newton-Wiege", "Newton's Cradle")],
        key="collision_type")
    
    if "1D" in collision_type:
        render_1d_collision_ui()
    elif "2D Stoß" in collision_type or "2D Collision" in collision_type:
        render_2d_collision_ui()
    elif "Billard" in collision_type or "Billiard" in collision_type:
        render_billiard_ui()
    elif "Newton" in collision_type:
        render_newtons_cradle_ui()


def render_1d_collision_ui():
    """1D Stoß UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("#### Zentraler Stoß in einer Dimension", "#### Central Collision in 1D"))
    st.latex(r"m_1 v_1 + m_2 v_2 = m_1 v_1' + m_2 v_2'")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(tr("**Körper 1**", "**Body 1**"))
        m1 = st.slider("m₁ [kg]", 0.5, 5.0, 1.0, 0.1, key="c1d_m1")
        v1 = st.slider("v₁ [m/s]", -5.0, 5.0, 2.0, 0.1, key="c1d_v1")
    with col2:
        st.markdown(tr("**Körper 2**", "**Body 2**"))
        m2 = st.slider("m₂ [kg]", 0.5, 5.0, 2.0, 0.1, key="c1d_m2")
        v2 = st.slider("v₂ [m/s]", -5.0, 5.0, -1.0, 0.1, key="c1d_v2")
    
    restitution = st.slider(tr("Stoßzahl e (0=inelastisch, 1=elastisch)", "Restitution e (0=inelastic, 1=elastic)"), 0.0, 1.0, 1.0, 0.05, key="c1d_e")
    animate = st.checkbox(tr("Animation", "Animation"), key="c1d_animate")
    
    if st.button(tr("▶️ Stoß berechnen", "▶️ Compute collision"), key="c1d_run", use_container_width=True):
        v1_new, v2_new = inelastic_collision_1d(m1, v1, m2, v2, restitution)
        
        p_before = m1 * v1 + m2 * v2
        p_after = m1 * v1_new + m2 * v2_new
        E_before = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
        E_after = 0.5 * m1 * v1_new**2 + 0.5 * m2 * v2_new**2
        
        if animate:
            run_1d_collision_animation(m1, v1, m2, v2, v1_new, v2_new)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("v₁'", f"{v1_new:.2f} m/s", f"{v1_new - v1:.2f}")
        with col2:
            st.metric("v₂'", f"{v2_new:.2f} m/s", f"{v2_new - v2:.2f}")
        with col3:
            st.metric(tr("Impuls p", "Momentum p"), f"{p_after:.2f} kg·m/s")
        with col4:
            energy_loss = (E_before - E_after) / E_before * 100 if E_before > 0 else 0
            st.metric(tr("Energieverlust", "Energy loss"), f"{energy_loss:.1f}%")


def render_2d_collision_ui():
    """2D Stoß UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("#### Schiefer Stoß in 2D", "#### Oblique Collision in 2D"))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(tr("**Kugel 1 (rot)**", "**Ball 1 (red)**"))
        v1x = st.slider("v₁ₓ [m/s]", -3.0, 3.0, 2.0, 0.1, key="c2d_v1x")
        v1y = st.slider("v₁ᵧ [m/s]", -3.0, 3.0, 0.5, 0.1, key="c2d_v1y")
    with col2:
        st.markdown(tr("**Kugel 2 (blau)**", "**Ball 2 (blue)**"))
        v2x = st.slider("v₂ₓ [m/s]", -3.0, 3.0, -1.0, 0.1, key="c2d_v2x")
        v2y = st.slider("v₂ᵧ [m/s]", -3.0, 3.0, -0.3, 0.1, key="c2d_v2y")
    
    impact_param = st.slider(tr("Stoßparameter b", "Impact parameter b"), 0.0, 1.0, 0.3, 0.05, key="c2d_b")
    restitution = st.slider(tr("Stoßzahl e", "Restitution e"), 0.0, 1.0, 1.0, 0.05, key="c2d_e")
    
    if st.button(tr("▶️ 2D-Stoß animieren", "▶️ Animate 2D collision"), key="c2d_run", use_container_width=True):
        run_2d_collision_animation(v1x, v1y, v2x, v2y, impact_param, restitution)


def render_billiard_ui():
    """Billard UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("#### Billard-Simulation", "#### Billiard Simulation"))
    n_balls = st.slider(tr("Anzahl Kugeln", "Number of balls"), 3, 10, 5, 1, key="bill_n")
    
    if st.button(tr("▶️ Billard starten", "▶️ Start billiard"), key="bill_run", use_container_width=True):
        run_billiard_simulation(n_balls)


def render_newtons_cradle_ui():
    """Newton-Wiege UI"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.markdown(tr("#### Newton-Wiege", "#### Newton's Cradle"))
    n_balls = st.slider(tr("Anzahl Kugeln", "Number of balls"), 3, 7, 5, 1, key="nc_n")
    n_start = st.slider(tr("Ausgelenkte Kugeln", "Displaced balls"), 1, n_balls - 1, 1, 1, key="nc_start")
    
    if st.button(tr("▶️ Newton-Wiege starten", "▶️ Start Newton's Cradle"), key="nc_run", use_container_width=True):
        run_newtons_cradle(n_balls, n_start)


def run_1d_collision_animation(m1, v1, m2, v2, v1_new, v2_new):
    """Animation eines 1D-Stoßes mit Plotly Frames"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    n_frames, collision_frame = 100, 40
    r1, r2 = 0.2 * np.sqrt(m1), 0.2 * np.sqrt(m2)
    x1_start, x2_start = -2.0, 2.0
    theta = np.linspace(0, 2*np.pi, 50)
    
    # Frames vorberechnen
    frames = []
    for frame in range(n_frames):
        t = frame / 30.0
        if frame < collision_frame:
            x1, x2 = x1_start + v1 * t, x2_start + v2 * t
            c1, c2 = 'blue', 'red'
        else:
            t_after = (frame - collision_frame) / 30.0
            x1, x2 = 0 + v1_new * t_after, 0 + v2_new * t_after
            c1, c2 = 'lightblue', 'salmon'
        
        frame_data = [
            go.Scatter(x=x1 + r1 * np.cos(theta), y=r1 * np.sin(theta), 
                      fill='toself', fillcolor=c1, line=dict(color='darkblue'), showlegend=False),
            go.Scatter(x=x2 + r2 * np.cos(theta), y=r2 * np.sin(theta), 
                      fill='toself', fillcolor=c2, line=dict(color='darkred'), showlegend=False)
        ]
        frames.append(go.Frame(data=frame_data, name=str(frame)))
    
    fig = go.Figure(
        data=[
            go.Scatter(x=x1_start + r1 * np.cos(theta), y=r1 * np.sin(theta),
                      fill='toself', fillcolor='blue', line=dict(color='darkblue'), name=f"m₁={m1}"),
            go.Scatter(x=x2_start + r2 * np.cos(theta), y=r2 * np.sin(theta),
                      fill='toself', fillcolor='red', line=dict(color='darkred'), name=f"m₂={m2}")
        ],
        frames=frames
    )
    
    fig.update_layout(
        title=tr("1D-Stoß Animation", "1D Collision Animation"),
        xaxis=dict(range=[-5, 5], title="x [m]"),
        yaxis=dict(range=[-1, 1], visible=False, scaleanchor="x"),
        height=280,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.2, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 30, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                dict(label="🔄 Reset", method="animate",
                     args=[["0"], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def run_2d_collision_animation(v1x, v1y, v2x, v2y, impact_param, restitution):
    """Animation eines 2D-Stoßes mit Plotly Frames"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    b1 = Body2D("Ball 1", -2.0, impact_param, v1x, v1y, 1.0, 0.15, "red")
    b2 = Body2D("Ball 2", 2.0, 0.0, v2x, v2y, 1.0, 0.15, "blue")
    
    n_frames, dt, collision_occurred = 120, 0.02, False
    theta = np.linspace(0, 2*np.pi, 30)
    
    # Simulation vorberechnen
    positions = []
    trails1_x, trails1_y = [], []
    trails2_x, trails2_y = [], []
    
    for frame in range(n_frames):
        b1.x += b1.vx * dt
        b1.y += b1.vy * dt
        b2.x += b2.vx * dt
        b2.y += b2.vy * dt
        
        trails1_x.append(b1.x)
        trails1_y.append(b1.y)
        trails2_x.append(b2.x)
        trails2_y.append(b2.y)
        
        dist = np.sqrt((b2.x - b1.x)**2 + (b2.y - b1.y)**2)
        if dist < b1.radius + b2.radius and not collision_occurred:
            b1, b2 = collision_2d(b1, b2, restitution)
            collision_occurred = True
        
        positions.append((b1.x, b1.y, b2.x, b2.y, list(trails1_x), list(trails1_y), list(trails2_x), list(trails2_y)))
    
    # Frames erstellen
    frames = []
    for i, (x1, y1, x2, y2, t1x, t1y, t2x, t2y) in enumerate(positions):
        frame_data = []
        if len(t1x) > 1:
            frame_data.append(go.Scatter(x=t1x, y=t1y, mode='lines', line=dict(color='lightcoral', width=1), showlegend=False))
        if len(t2x) > 1:
            frame_data.append(go.Scatter(x=t2x, y=t2y, mode='lines', line=dict(color='lightblue', width=1), showlegend=False))
        
        frame_data.append(go.Scatter(x=x1 + 0.15 * np.cos(theta), y=y1 + 0.15 * np.sin(theta),
                                    fill='toself', fillcolor='red', line=dict(color='black'), showlegend=False))
        frame_data.append(go.Scatter(x=x2 + 0.15 * np.cos(theta), y=y2 + 0.15 * np.sin(theta),
                                    fill='toself', fillcolor='blue', line=dict(color='black'), showlegend=False))
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    fig = go.Figure(
        data=[
            go.Scatter(x=[-2.0 + 0.15 * np.cos(t) for t in theta], y=[impact_param + 0.15 * np.sin(t) for t in theta],
                      fill='toself', fillcolor='red', line=dict(color='black'), name="Ball 1"),
            go.Scatter(x=[2.0 + 0.15 * np.cos(t) for t in theta], y=[0.15 * np.sin(t) for t in theta],
                      fill='toself', fillcolor='blue', line=dict(color='black'), name="Ball 2")
        ],
        frames=frames
    )
    
    fig.update_layout(
        title=tr(f"2D-Stoß (e={restitution})", f"2D Collision (e={restitution})"),
        xaxis=dict(range=[-4, 4], title="x", scaleanchor="y"),
        yaxis=dict(range=[-2, 2], title="y"),
        height=400, showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 25, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                dict(label="🔄 Reset", method="animate",
                     args=[["0"], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def run_billiard_simulation(n_balls: int):
    """Billard-Simulation mit Plotly Frames"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    table_w, table_h = 4.0, 2.0
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink']
    balls = []
    
    for i in range(n_balls):
        x = np.random.uniform(-table_w/2 + 0.2, table_w/2 - 0.2)
        y = np.random.uniform(-table_h/2 + 0.2, table_h/2 - 0.2)
        vx = np.random.uniform(-1.5, 1.5)
        vy = np.random.uniform(-1.5, 1.5)
        balls.append(Body2D(f"Ball {i+1}", x, y, vx, vy, 1.0, 0.12, colors[i % len(colors)]))
    
    st.info(tr("⏳ Berechne Simulation...", "⏳ Computing simulation..."))
    
    n_frames, dt = 200, 0.02
    all_positions = []
    
    for frame in range(n_frames):
        for b in balls:
            b.x += b.vx * dt
            b.y += b.vy * dt
            
            if b.x - b.radius < -table_w/2:
                b.x = -table_w/2 + b.radius
                b.vx = -b.vx * 0.95
            elif b.x + b.radius > table_w/2:
                b.x = table_w/2 - b.radius
                b.vx = -b.vx * 0.95
            if b.y - b.radius < -table_h/2:
                b.y = -table_h/2 + b.radius
                b.vy = -b.vy * 0.95
            elif b.y + b.radius > table_h/2:
                b.y = table_h/2 - b.radius
                b.vy = -b.vy * 0.95
            
            b.vx *= 0.998
            b.vy *= 0.998
        
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                dist = np.sqrt((balls[j].x - balls[i].x)**2 + (balls[j].y - balls[i].y)**2)
                if dist < balls[i].radius + balls[j].radius:
                    collision_2d(balls[i], balls[j], 0.95)
        
        all_positions.append([(b.x, b.y, b.color) for b in balls])
    
    # Frames erstellen
    theta = np.linspace(0, 2*np.pi, 20)
    frames = []
    for i, positions in enumerate(all_positions):
        frame_data = [go.Scatter(
            x=[-table_w/2, table_w/2, table_w/2, -table_w/2, -table_w/2],
            y=[-table_h/2, -table_h/2, table_h/2, table_h/2, -table_h/2],
            fill='toself', fillcolor='darkgreen', line=dict(color='brown', width=4),
            showlegend=False, hoverinfo='skip'
        )]
        for x, y, color in positions:
            frame_data.append(go.Scatter(
                x=x + 0.12 * np.cos(theta), y=y + 0.12 * np.sin(theta),
                fill='toself', fillcolor=color, line=dict(color='black', width=1), showlegend=False
            ))
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial
    init_data = [go.Scatter(
        x=[-table_w/2, table_w/2, table_w/2, -table_w/2, -table_w/2],
        y=[-table_h/2, -table_h/2, table_h/2, table_h/2, -table_h/2],
        fill='toself', fillcolor='darkgreen', line=dict(color='brown', width=4),
        showlegend=False
    )]
    for x, y, color in all_positions[0]:
        init_data.append(go.Scatter(
            x=x + 0.12 * np.cos(theta), y=y + 0.12 * np.sin(theta),
            fill='toself', fillcolor=color, line=dict(color='black', width=1), showlegend=False
        ))
    
    fig = go.Figure(data=init_data, frames=frames)
    
    fig.update_layout(
        title=tr(f"Billard ({n_balls} Kugeln)", f"Billiard ({n_balls} balls)"),
        xaxis=dict(range=[-table_w/2 - 0.2, table_w/2 + 0.2], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-table_h/2 - 0.2, table_h/2 + 0.2], visible=False),
        height=380, margin=dict(l=20, r=20, t=50, b=20),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.15, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 20, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                dict(label="🔄 Reset", method="animate",
                     args=[["0"], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def run_newtons_cradle(n_balls: int, n_start: int):
    """Newton-Wiege Animation mit Plotly Frames"""
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    L, r, spacing = 1.5, 0.1, 0.22
    theta_angles = np.zeros(n_balls)
    omega = np.zeros(n_balls)
    theta_angles[:n_start] = 0.5
    
    n_frames, dt, g = 300, 0.01, 9.81
    
    # Simulation vorberechnen
    all_positions = []
    for frame in range(n_frames):
        for i in range(n_balls):
            alpha = -(g / L) * np.sin(theta_angles[i])
            omega[i] += alpha * dt
            theta_angles[i] += omega[i] * dt
        
        for i in range(n_balls - 1):
            x_i = i * spacing + L * np.sin(theta_angles[i])
            x_j = (i + 1) * spacing + L * np.sin(theta_angles[i + 1])
            if x_j - x_i < 2 * r:
                omega[i], omega[i + 1] = omega[i + 1], omega[i]
        
        all_positions.append(theta_angles.copy())
    
    # Frames erstellen
    theta_circle = np.linspace(0, 2*np.pi, 30)
    frames = []
    
    for i, thetas in enumerate(all_positions):
        frame_data = [go.Scatter(
            x=[-spacing, (n_balls) * spacing], y=[0, 0],
            mode='lines', line=dict(color='brown', width=4), showlegend=False
        )]
        
        for j in range(n_balls):
            x_pivot = j * spacing
            x_ball = x_pivot + L * np.sin(thetas[j])
            y_ball = -L * np.cos(thetas[j])
            
            frame_data.append(go.Scatter(
                x=[x_pivot, x_ball], y=[0, y_ball],
                mode='lines', line=dict(color='gray', width=2), showlegend=False
            ))
            frame_data.append(go.Scatter(
                x=x_ball + r * np.cos(theta_circle), y=y_ball + r * np.sin(theta_circle),
                fill='toself', fillcolor='silver', line=dict(color='gray', width=1), showlegend=False
            ))
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial
    init_thetas = np.zeros(n_balls)
    init_thetas[:n_start] = 0.5
    
    init_data = [go.Scatter(
        x=[-spacing, (n_balls) * spacing], y=[0, 0],
        mode='lines', line=dict(color='brown', width=4), showlegend=False
    )]
    for j in range(n_balls):
        x_pivot = j * spacing
        x_ball = x_pivot + L * np.sin(init_thetas[j])
        y_ball = -L * np.cos(init_thetas[j])
        init_data.append(go.Scatter(x=[x_pivot, x_ball], y=[0, y_ball], mode='lines', line=dict(color='gray', width=2), showlegend=False))
        init_data.append(go.Scatter(x=x_ball + r * np.cos(theta_circle), y=y_ball + r * np.sin(theta_circle),
                                   fill='toself', fillcolor='silver', line=dict(color='gray'), showlegend=False))
    
    fig = go.Figure(data=init_data, frames=frames)
    
    fig.update_layout(
        title=tr(f"Newton-Wiege ({n_balls} Kugeln)", f"Newton's Cradle ({n_balls} balls)"),
        xaxis=dict(range=[-0.5, (n_balls - 1) * spacing + 0.5], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-L - 0.3, 0.2], visible=False),
        height=420,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {"frame": {"duration": 15, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸️ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                dict(label="🔄 Reset", method="animate",
                     args=[["0"], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)
