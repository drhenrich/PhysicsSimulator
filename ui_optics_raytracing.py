
# ============================================================
# ui_optics_raytracing.py — UI wrapper for optics_raytracing.py
# Exposes: render_optics_raytracing_tab()
# ============================================================
from __future__ import annotations

def render_optics_raytracing_tab():
    import numpy as np
    import streamlit as st
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    try:
        import optics_raytracing as rt
    except Exception as e:
        st.error(tr(f"Raytracing-Modul nicht verfügbar: {e}", f"Raytracing module unavailable: {e}"))
        return

    st.subheader(tr("Optik — Linsen & Presets (klassisch)", "Optics — Lenses & presets (classic)"))

    presets = rt.presets()
    names = list(presets.keys())

    with st.form("rt_form_classic"):
        preset = st.selectbox(tr("Preset", "Preset"), names, index=0, key="rt_preset_cls")
        n_rays = st.selectbox(tr("Anzahl Strahlen", "Number of rays"), [11, 21, 31, 41], index=1, key="rt_nr_cls")
        height = st.number_input(tr("Objekt-Höhe [mm]", "Object height [mm]"), min_value=1.0, max_value=50.0, value=10.0, step=1.0, key="rt_h_cls")
        ang = st.slider(tr("Winkelspreizung [°]", "Angular spread [°]"), 0.5, 10.0, 3.0, 0.5, key="rt_ang_cls")
        run = st.form_submit_button(tr("Raytracing zeichnen", "Draw raytracing"), use_container_width=True)

    if run:
        elems = presets[preset]
        y0, th0 = rt.fan_rays(n=int(n_rays), height=height*1e-3, ang=np.deg2rad(ang))
        xs, ys = rt.trace_system(elems, y0, th0)
        # Chromatik-Preset: mische drei Farben durch ±5% f
        if "Chromatik" in preset:
            colors = [(255,120,120), (120,255,120), (120,120,255)]
            shifts = [-0.05, 0.0, +0.05]
            canvas = None
            for c, s in zip(colors, shifts):
                elems2 = []
                for kind, e in elems:
                    if kind == "lens":
                        elems2.append(("lens", rt.ThinLens(e.f*(1.0+s))))
                    elif kind == "space":
                        elems2.append(("space", rt.Space(e.L)))
                    else:
                        elems2.append(("aperture", rt.Aperture(e.radius)))
                xs2, ys2 = rt.trace_system(elems2, y0, th0)
                img = rt.draw_system(xs2, ys2)
                mask = (img[:,:,0]+img[:,:,1]+img[:,:,2]) > 0
                if canvas is None:
                    canvas = np.zeros_like(img)
                canvas[mask] = c
            st.image(canvas, caption="Chromatische Fokusverschiebung (klassisch)", use_container_width=True)
        else:
            img = rt.draw_system(xs, ys)
            st.image(img, caption=preset, use_container_width=True)
    else:
        st.info(tr("Preset wählen und 'Raytracing zeichnen' klicken.", "Select a preset and click 'Draw raytracing'."))
