from __future__ import annotations
import numpy as np
import streamlit as st

# --- Optics raytracing core (lightweight) ---

def fan_rays(n: int = 21, height: float = 0.01, ang: float = 0.05):
    n = max(3, int(n))
    y0 = np.linspace(-height / 2.0, height / 2.0, n)
    th0 = np.linspace(-ang / 2.0, ang / 2.0, n)
    return y0, th0

def trace_system(elems, y0, th0):
    y = np.array(y0, dtype=float); th = np.array(th0, dtype=float)
    xs = [np.zeros_like(y)]; ys = [y.copy()]; x_pos = 0.0
    for kind, elem in elems:
        if kind == "space":
            x_pos += elem[0]; y = y + th * elem[0]
        elif kind == "lens":
            f = elem[0]; th = th - y / f
        elif kind == "aperture":
            radius = elem[0]; mask = np.abs(y) <= radius; y = np.where(mask, y, np.nan); th = np.where(mask, th, np.nan)
        xs.append(np.full_like(y, x_pos)); ys.append(y.copy())
    return np.vstack(xs).T, np.vstack(ys).T

def draw_system(xs, ys):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return np.zeros((300, 800, 3), dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(8, 3))
    xs = np.array(xs); ys = np.array(ys)
    for i in range(xs.shape[0]):
        ax.plot(xs[i], ys[i], color="orange", linewidth=1, alpha=0.9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True, alpha=0.2)
    fig.tight_layout();
    import io; buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120); plt.close(fig); buf.seek(0)
    img = plt.imread(buf)
    if img.dtype != np.uint8:
        img = (img*255).astype(np.uint8)
    return img

def presets():
    return {
        "Sammellinse": [("space", (0.15,)), ("lens", (0.2,)), ("space", (0.25,))],
        "Chromatik Sammellinse": [("space", (0.15,)), ("lens", (0.2,)), ("space", (0.25,))],
        "Zerstreuungslinse": [("space", (0.15,)), ("lens", (-0.2,)), ("space", (0.3,))],
        "Teleskop (zwei Linsen)": [("space", (0.1,)), ("lens", (0.5,)), ("space", (0.55,)), ("lens", (0.05,)), ("space", (0.3,))],
        "Apertur-Blende": [("space", (0.1,)), ("aperture", (0.03,)), ("space", (0.2,)), ("lens", (0.15,)), ("space", (0.2,))],
    }

# --- UI: Optics combo (wave + ray) ---
def render_optics_combo_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    st.subheader(tr("Optik — Wellenoptik & Raytracing", "Optics — Wave optics & raytracing"))
    wave_tab, ray_tab = st.tabs([tr("Wellenoptik", "Wave optics"), tr("Raytracing (klassisch)", "Raytracing (classic)")])

    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min(); A = A/(A.max()+1e-12)
        return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")

    with wave_tab:
        with st.form("wave_combo_safe"):
            N = st.selectbox(tr("Raster N", "Grid N"), [256, 384, 512], index=0, key="woN_combo")
            shape = st.selectbox(tr("Apertur", "Aperture"), ["Einzelspalt","Doppelspalt","Gitter"], index=0, key="woShape_combo")
            run = st.form_submit_button(tr("Beugungsmuster berechnen", "Compute diffraction pattern"))
        if run:
            N = int(N)
            x = np.linspace(-1,1,N,endpoint=False)
            X,_ = np.meshgrid(x,x, indexing="xy")
            if shape=="Einzelspalt":
                I = (np.sinc(6*X))**2
            elif shape=="Doppelspalt":
                I = (np.cos(25*X)**2) * (np.sinc(6*X)**2)
            else:
                I = (np.cos(60*X)**2) * (np.sinc(6*X)**2)
            st.image(to_img(I), caption=tr(f"{shape} — didaktisches Fraunhofer-Muster", f"{shape} — didactic Fraunhofer pattern"), use_container_width=True)
        else:
            st.info(tr("Parameter wählen und 'Beugungsmuster berechnen' klicken.", "Select parameters and click 'Compute diffraction pattern'."))

    with ray_tab:
        render_optics_raytracing_tab()

# --- UI: Raytracing tab ---
def render_optics_raytracing_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    st.subheader(tr("Optik — Linsen & Presets (klassisch)", "Optics — Lenses & presets (classic)"))
    preset_map = presets(); names = list(preset_map.keys())
    with st.form("rt_form_classic"):
        preset = st.selectbox(tr("Preset", "Preset"), names, index=0, key="rt_preset_cls")
        n_rays = st.selectbox(tr("Anzahl Strahlen", "Number of rays"), [11,21,31,41], index=1, key="rt_nr_cls")
        height = st.number_input(tr("Objekt-Höhe [mm]", "Object height [mm]"), min_value=1.0, max_value=50.0, value=10.0, step=1.0, key="rt_h_cls")
        ang = st.slider(tr("Winkelspreizung [°]", "Angular spread [°]"), 0.5, 10.0, 3.0, 0.5, key="rt_ang_cls")
        run = st.form_submit_button(tr("Raytracing zeichnen", "Draw raytracing"), use_container_width=True)
    if run:
        elems = preset_map[preset]
        y0, th0 = fan_rays(n=int(n_rays), height=height*1e-3, ang=np.deg2rad(ang))
        xs, ys = trace_system(elems, y0, th0)
        if "Chromatik" in preset:
            colors = [(255,120,120), (120,255,120), (120,120,255)]; shifts = [-0.05,0.0,0.05]; canvas=None
            for c,s in zip(colors, shifts):
                elems2=[]
                for kind, e in elems:
                    if kind=="lens": elems2.append(("lens", (e[0]*(1.0+s),)))
                    elif kind=="space": elems2.append(("space", e))
                    else: elems2.append(("aperture", e))
                xs2, ys2 = trace_system(elems2, y0, th0)
                img = draw_system(xs2, ys2)
                mask = (img[:,:,0]+img[:,:,1]+img[:,:,2]) > 0
                if canvas is None:
                    canvas = np.zeros_like(img)
                canvas[mask] = c
            st.image(canvas, caption=tr("Chromatische Fokusverschiebung (klassisch)", "Chromatic focus shift (classic)"), use_container_width=True)
        else:
            img = draw_system(xs, ys)
            st.image(img, caption=preset, use_container_width=True)
    else:
        st.info(tr("Preset wählen und 'Raytracing zeichnen' klicken.", "Select a preset and click 'Draw raytracing'."))
