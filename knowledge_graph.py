# ============================================================
# knowledge_graph.py – Local HyperPhysics hub + robust Plotly graph
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Node/Edge model -------------------------------------------------------------
@dataclass
class ConceptNode:
    id: str
    label: str
    kind: str = "topic"
    tags: Sequence[str] = field(default_factory=tuple)
    action: Optional[Callable] = None

@dataclass
class ConceptEdge:
    src: str
    dst: str
    label: Optional[str] = None

class ConceptGraph:
    def __init__(self, nodes: Iterable[ConceptNode], edges: Iterable[ConceptEdge], *, title: str = "Physik Simulation", fixed_pos: Optional[Dict[str, Tuple[float,float]]] = None):
        self.nodes: Dict[str, ConceptNode] = {n.id: n for n in nodes}
        self.edges: List[ConceptEdge] = list(edges)
        self.title = title
        self.fixed_pos = dict(fixed_pos) if fixed_pos else {}

    def get(self, nid: str) -> ConceptNode:
        return self.nodes[nid]

    def resolve_action(self, nid: str):
        n = self.nodes[nid]
        return n.action

    def to_plotly(self):
        import plotly.graph_objects as go
        try:
            import networkx as nx
        except Exception as e:
            raise RuntimeError(f"NetworkX benötigt: {e}")

        G = nx.DiGraph()
        for n in self.nodes.values():
            G.add_node(n.id)
        for e in self.edges:
            if e.src in G.nodes and e.dst in G.nodes:
                G.add_edge(e.src, e.dst)

        # Positions: fixed if provided, else spring_layout
        if self.fixed_pos:
            pos = {nid: (float(x), float(y)) for nid, (x, y) in self.fixed_pos.items() if nid in G.nodes}
        else:
            pos = nx.spring_layout(G, seed=1, k=1.0)

        ids_out = list(self.nodes.keys())
        nx_list_x = [pos.get(i, (None, None))[0] for i in ids_out]
        nx_list_y = [pos.get(i, (None, None))[1] for i in ids_out]

        # Edges
        edge_x, edge_y = [], []
        for e in self.edges:
            if e.src in pos and e.dst in pos:
                x0, y0 = pos[e.src]
                x1, y1 = pos[e.dst]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="none", showlegend=False)

        # Node style by kind
        def marker_for(kind: str):
            if kind == "topic": return dict(size=18, symbol="circle", color="#f2a2a2")
            if kind == "sim": return dict(size=16, symbol="square", color="#72b7b2")
            if kind == "formula": return dict(size=14, symbol="diamond", color="#f58518")
            return dict(size=12, symbol="circle-open", color="#4c78a8")

        markers = [marker_for(self.nodes[nid].kind) for nid in ids_out]

        node_trace = go.Scatter(
            x=nx_list_x, y=nx_list_y, mode="markers+text",
            text=[self.nodes[i].label for i in ids_out],
            textposition="middle center",
            customdata=ids_out,
            marker=dict(
                size=[m["size"] for m in markers],
                symbol=[m["symbol"] for m in markers],
                color=[m["color"] for m in markers],
                line=dict(width=1, color="#222"),
                opacity=0.95
            ),
            hoverinfo="text", showlegend=False
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        # Visible region with generous padding, no fixed aspect
        xs_nodes = [x for x in nx_list_x if x is not None]
        ys_nodes = [y for y in nx_list_y if y is not None]
        xs_edges = [x for x in edge_trace.x if x is not None]
        ys_edges = [y for y in edge_trace.y if y is not None]
        X = xs_nodes + xs_edges
        Y = ys_nodes + ys_edges
        if X and Y:
            xmin, xmax = min(X), max(X)
            ymin, ymax = min(Y), max(Y)
            cx, cy = (xmin + xmax)/2.0, (ymin + ymax)/2.0
            span = max(xmax - xmin, ymax - ymin) or 1.0
            pad = 0.35 * span
            x_range = [cx - span/2 - pad, cx + span/2 + pad]
            y_range = [cy - span/2 - pad, cy + span/2 + pad]
        else:
            x_range = [-1, 1]; y_range = [-1, 1]

        fig.update_layout(
            xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, constrain="domain"),
            yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False, scaleanchor=None),
            height=720, margin=dict(l=10, r=10, t=80, b=10),
            autosize=True, hovermode="closest", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=f"Wissenslandkarte: {getattr(self, 'title', 'Physik Simulation')}", x=0.5),
            clickmode="event+select"
        )
        return fig

# --- Actions (node launchers) -------------------------------------------------
def action_mechanics_elastic():
    """Run elastic collision preset and return Plotly figure."""
    try:
        from scenarios import scenario_elastic_collision
    except Exception:
        from physics_simulator.scenarios import scenario_elastic_collision
    try:
        from core import Simulator
    except Exception:
        from physics_simulator.core import Simulator
    try:
        from plotting import plot_trajectories_3d
    except Exception:
        from physics_simulator.plotting import plot_trajectories_3d

    bodies, connections, _note = scenario_elastic_collision()
    sim = Simulator(bodies=bodies, connections=connections)
    results = sim.run()
    try:
        fig = plot_trajectories_3d(results)
        return fig
    except Exception:
        return results

def action_light_diffraction():
    """Fraunhofer Doppelspalt (analytisch) als Heatmap."""
    import numpy as np
    import plotly.graph_objects as go
    lam = 550e-9; d = 100e-6; a = 30e-6
    theta = np.linspace(-6e-3, 6e-3, 800)
    s = np.sin(theta)
    beta = np.pi * d * s / lam
    alpha = np.pi * a * s / lam
    sinc = np.where(alpha==0, 1.0, np.sin(alpha)/alpha)
    I = (np.cos(beta)**2) * (sinc**2)
    Y, X = np.meshgrid(np.linspace(0, 1, 80), theta, indexing="ij")
    Z = np.tile(I, (80,1))
    fig = go.Figure(data=go.Heatmap(z=Z, x=theta*1e3, y=Y[:,0], coloraxis="coloraxis"))
    fig.update_layout(
        title="Fraunhofer-Doppelspalt (λ=550 nm, d=100 μm, a=30 μm)",
        xaxis_title="Winkel θ [mrad]", yaxis=dict(visible=False), height=450,
        coloraxis=dict(colorscale="Viridis", showscale=False), margin=dict(l=10,r=10,t=60,b=10)
    )
    return fig

# Builders --------------------------------------------------------------------
def build_hyperphysics_hub() -> ConceptGraph:
    nodes: List[ConceptNode] = []
    edges: List[ConceptEdge] = []
    nodes.append(ConceptNode("hp_center", "HyperPhysics", "topic"))

    topics = [
        ("hp_mech", "Mechanics", 0.0),
        ("hp_em", "Electricity and Magnetism", 35.0),
        ("hp_light", "Light and Vision", 65.0),
        ("hp_sound", "Sound and Hearing", 105.0),
        ("hp_rel", "Relativity", 145.0),
        ("hp_astro", "Astrophysics", 200.0),
        ("hp_quant", "Quantum Physics", 260.0),
        ("hp_nuclear", "Nuclear Physics", 300.0),
        ("hp_cm", "Condensed Matter", 330.0),
        ("hp_heat", "Heat and Thermodynamics", 355.0),
    ]
    import math
    R = 5.0
    fixed_pos = {"hp_center": (0.0, 0.0)}
    for nid, label, deg in topics:
        rad = math.radians(deg)
        x, y = R*math.cos(rad), R*math.sin(rad)
        action = None
        if nid == "hp_mech":
            action = action_mechanics_elastic
        elif nid == "hp_light":
            action = action_light_diffraction
        nodes.append(ConceptNode(nid, label, "topic", action=action))
        fixed_pos[nid] = (x, y)
        edges.append(ConceptEdge("hp_center", nid))

    g = ConceptGraph(nodes, edges, title="HyperPhysics (lokal)", fixed_pos=fixed_pos)
    return g
