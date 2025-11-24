
# ============================================================
# ui_domain_optics_combo.py — Optik (Wellenoptik & Raytracing) — Safe
# Exposes: render_optics_combo_tab()
# ============================================================
from __future__ import annotations

try:
    from i18n_utils import get_text
except Exception:
    def get_text(key, language="de"):
        return None

def render_optics_combo_tab():
    import numpy as np
    import streamlit as st
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en

    st.subheader(tr("Optik — Wellenoptik & Raytracing", "Optics — Wave Optics & Raytracing"))

    wave_tab, ray_tab = st.tabs([tr("Wellenoptik", "Wave optics"), tr("Raytracing (klassisch)", "Raytracing (classic)")])

    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min(); A = A/(A.max()+1e-12)
        return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")

    # ---------- Wellenoptik (didaktisch, bildbasiert) ----------
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

    # ---------- Raytracing (klassisch, Linsen + Presets) ----------
    with ray_tab:
        try:
            from ui_optics_raytracing import render_optics_raytracing_tab
            render_optics_raytracing_tab()
        except Exception as e:
            st.error(tr(f"Raytracing-UI nicht verfügbar: {e}", f"Raytracing UI not available: {e}"))
