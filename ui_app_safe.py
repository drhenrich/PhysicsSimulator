
# ============================================================
# ui_app_safe.py — Safe Mode + Optik-Kombi + CT (klassisch & safe)
# ============================================================
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Physics Teaching Simulator — Safe Domains", page_icon="🧩", layout="wide")
st.title("Physics Teaching Simulator — Hauptgebiete (Safe Mode)")

# Domain-Tabs importieren
try:
    from ui_domain_mech_safe import render_mech_astro_tab
except Exception as e:
    render_mech_astro_tab = None; mech_err = e
try:
    from ui_domain_optics_combo import render_optics_combo_tab
except Exception as e:
    render_optics_combo_tab = None; opt_combo_err = e
try:
    from ui_xray_ct import render_xray_ct_tab  # KLASSISCHER CT-TEIL
except Exception as e:
    render_xray_ct_tab = None; ct_classic_err = e
try:
    from ui_ct_safe import render_ct_safe_tab     # SAFE-CT
except Exception as e:
    render_ct_safe_tab = None; ct_safe_err = e
try:
    from ui_domain_mri_bloch_safe import render_mri_bloch_tab
except Exception as e:
    render_mri_bloch_tab = None; mrib_err = e
try:
    from ui_domain_em_safe import render_em_tab
except Exception as e:
    render_em_tab = None; em_err = e

tabs = st.tabs([
    "Mechanik & Astromechanik",
    "Optik (Wellenoptik + Raytracing)",
    "Röntgen/CT (klassisch)",
    "Röntgen/CT (safe)",
    "MRI & Bloch (safe)",
    "Elektrodynamik & Potential (safe)",
    "Diagnose"
])

with tabs[0]:
    if render_mech_astro_tab is None:
        st.error(f"Mechanik/Astromechanik nicht verfügbar: {mech_err}")
    else:
        render_mech_astro_tab()

with tabs[1]:
    if render_optics_combo_tab is None:
        st.error(f"Optik (Kombination) nicht verfügbar: {opt_combo_err}")
    else:
        render_optics_combo_tab()

with tabs[2]:
    if render_xray_ct_tab is None:
        st.error(f"CT (klassisch) nicht verfügbar: {ct_classic_err}")
    else:
        render_xray_ct_tab()

with tabs[3]:
    if render_ct_safe_tab is None:
        st.error(f"CT (safe) nicht verfügbar: {ct_safe_err}")
    else:
        render_ct_safe_tab()

with tabs[4]:
    if render_mri_bloch_tab is None:
        st.error(f"MRI/Bloch nicht verfügbar: {mrib_err}")
    else:
        render_mri_bloch_tab()

with tabs[5]:
    if render_em_tab is None:
        st.error(f"Elektrodynamik/Potential nicht verfügbar: {em_err}")
    else:
        render_em_tab()

with tabs[6]:
    import numpy as np, pandas as pd
    st.table({"A":[1,2], "B":[3,4]})
    st.line_chart(pd.DataFrame({"x": np.linspace(0,6,120), "sin": np.sin(np.linspace(0,6,120))}), x="x", y="sin")
    img = np.tile(np.linspace(0,1,256),(256,1)).astype("float32")
    img = (np.stack([img, img**0.5, img[:,::-1]], axis=2)*255).astype("uint8")
    st.image(img, caption="st.image Test", use_container_width=True)
