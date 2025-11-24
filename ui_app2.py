from __future__ import annotations

import streamlit as st

# Übersetzungsfunktionen aus i18n_utils.py importieren
from i18n_utils import get_text, get_language_name
from presets import PRESENTS_BLOCH, PRESENTS_OPT_WAVE, PRESENTS_CT, PRESENTS_MECH

st.set_page_config(
    page_title=get_text("title", "de"),
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "language" not in st.session_state:
    st.session_state.language = "de"

st.markdown("""
<style>
.main { padding: 2rem; }
h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
h2 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# HEADER MIT TITEL & SUBTITEL
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.write("")

with col2:
    title = get_text("title", st.session_state.language) or "Physics Teaching Simulator — Hauptgebiete (Safe Mode)"
    st.markdown(f"# {title}")
    subtitle = get_text("subtitle", st.session_state.language) or "Interaktive Simulation sicherer Physikgebiete"
    st.markdown(f"**{subtitle}**")

with col3:
    lang_choice = st.selectbox(
        "🌍", ["de", "en"],
        format_func=lambda x: get_language_name(x),
        index=0 if st.session_state.language == "de" else 1,
        key="lang_selector"
    )

if lang_choice != st.session_state.language:
    st.session_state.language = lang_choice
    st.rerun()

st.markdown("---")

# ============================================
# SIDEBAR MIT PRESETS
# ============================================
with st.sidebar:
    st.markdown(f"## {get_text('settings', st.session_state.language)}")
    st.markdown(f"{subtitle}")
    st.markdown("---")
    
    # BLOCH-PRESETS
    st.markdown("### 🧲 Bloch-Parameter")
    preset_bloch = st.selectbox("Preset wählen", [""] + list(PRESENTS_BLOCH.keys()), key="preset_bloch")
    colB1, colB2, colB3 = st.columns(3)
    
    with colB1:
        load_bloch = st.button("Laden", key="bloch-load")
    with colB2:
        reset_bloch = st.button("Reset", key="bloch-reset")
    with colB3:
        st.write("")
    
    if load_bloch and preset_bloch:
        for k, v in PRESENTS_BLOCH[preset_bloch].items():
            st.session_state[f"b{k}"] = v
        st.success(f"✅ {preset_bloch} geladen")
    
    if reset_bloch:
        for k, v in PRESENTS_BLOCH["Standard"].items():
            st.session_state[f"b{k}"] = v
        st.info("🔄 Auf Standard zurückgesetzt")
    
    T1 = st.number_input("T1 [ms]", 100.0, 3000.0, st.session_state.get("bT1", 1000.0), 50.0, key="bT1")
    T2 = st.number_input("T2 [ms]", 10.0, 500.0, st.session_state.get("bT2", 80.0), 5.0, key="bT2")
    TR = st.number_input("TR [ms]", 50.0, 4000.0, st.session_state.get("bTR", 800.0), 25.0, key="bTR")
    TE = st.number_input("TE [ms]", 5.0, 200.0, st.session_state.get("bTE", 20.0), 5.0, key="bTE")
    
    st.markdown("---")
    
    # OPTIK-WELLENOPTIK-PRESETS
    st.markdown("### 🌊 Optik-Wellenoptik")
    preset_opt = st.selectbox("Preset wählen", [""] + list(PRESENTS_OPT_WAVE.keys()), key="preset_opt_wave")
    colO1, colO2, colO3 = st.columns(3)
    
    with colO1:
        load_opt = st.button("Laden", key="opt-load")
    with colO2:
        reset_opt = st.button("Reset", key="opt-reset")
    with colO3:
        st.write("")
    
    if load_opt and preset_opt:
        for k, v in PRESENTS_OPT_WAVE[preset_opt].items():
            st.session_state[f"o{k}"] = v
        st.success(f"✅ {preset_opt} geladen")
    
    if reset_opt:
        for k, v in PRESENTS_OPT_WAVE[list(PRESENTS_OPT_WAVE.keys())[0]].items():
            st.session_state[f"o{k}"] = v
        st.info("🔄 Auf Standard zurückgesetzt")
    
    st.markdown("---")
    
    # CT-PRESETS
    st.markdown("### 🏥 CT-Parameter")
    preset_ct = st.selectbox("Preset wählen", [""] + list(PRESENTS_CT.keys()), key="preset_ct")
    colC1, colC2, colC3 = st.columns(3)
    
    with colC1:
        load_ct = st.button("Laden", key="ct-load")
    with colC2:
        reset_ct = st.button("Reset", key="ct-reset")
    with colC3:
        st.write("")
    
    if load_ct and preset_ct:
        for k, v in PRESENTS_CT[preset_ct].items():
            st.session_state[f"c{k}"] = v
        st.success(f"✅ {preset_ct} geladen")
    
    if reset_ct:
        for k, v in PRESENTS_CT[list(PRESENTS_CT.keys())[0]].items():
            st.session_state[f"c{k}"] = v
        st.info("🔄 Auf Standard zurückgesetzt")
    
    st.markdown("---")
    
    # MECHANIK-PRESETS
    st.markdown("### 🎯 Mechanik-Szenen")
    preset_mech = st.selectbox("Preset wählen", [""] + list(PRESENTS_MECH.keys()), key="preset_mech")
    colM1, colM2, colM3 = st.columns(3)
    
    with colM1:
        load_mech = st.button("Laden", key="mech-load")
    with colM2:
        reset_mech = st.button("Reset", key="mech-reset")
    with colM3:
        st.write("")
    
    if load_mech and preset_mech:
        for k, v in PRESENTS_MECH[preset_mech].items():
            st.session_state[f"m{k}"] = v
        st.success(f"✅ {preset_mech} geladen")
    
    if reset_mech:
        for k, v in PRESENTS_MECH[list(PRESENTS_MECH.keys())[0]].items():
            st.session_state[f"m{k}"] = v
        st.info("🔄 Auf Standard zurückgesetzt")
    
    st.markdown("---")
    
    if st.button(get_text("refresh", st.session_state.language), use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("ℹ️ " + get_text("about", st.session_state.language)):
        st.write("Physics Simulator v2.1")
        st.write("🌍 Languages: Deutsch | English")
        st.metric("Modules", "6")

# ============================================
# TAB-IMPORTS UND ABBRUCHSICHERUNG
# ============================================
try:
    from ui_domain_mech_safe import render_mech_astro_tab
except Exception as e:
    render_mech_astro_tab = None
    mech_err = e

try:
    from ui_domain_optics_combo import render_optics_combo_tab
except Exception as e:
    render_optics_combo_tab = None
    opt_combo_err = e

try:
    from ui_xray_ct import render_xray_ct_tab
except Exception as e:
    render_xray_ct_tab = None
    ct_classic_err = e

try:
    from ui_ct_safe import render_ct_safe_tab
except Exception as e:
    render_ct_safe_tab = None
    ct_safe_err = e

try:
    from ui_domain_mri_bloch_safe import render_mri_bloch_tab
except Exception as e:
    render_mri_bloch_tab = None
    mrib_err = e

try:
    from ui_domain_em_safe import render_em_tab
except Exception as e:
    render_em_tab = None
    em_err = e

lang = st.session_state.language

# TAB-LABELS
tabs_labels = [
    get_text("mechanics", lang) or "Mechanik & Astromechanik",
    get_text("optics", lang) or "Optik (Wellenoptik + Raytracing)",
    get_text("xray_ct_classic", lang) or "Röntgen/CT (klassisch)",
    get_text("xray_ct_safe", lang) or "Röntgen/CT (safe)",
    get_text("mri_imaging", lang) or "MRI & Bloch (safe)",
    get_text("electromagnetism", lang) or "Elektrodynamik & Potential (safe)"
]

tabs = st.tabs(tabs_labels)

with tabs[0]:
    st.markdown(f"# {tabs_labels[0]}")
    if render_mech_astro_tab is None:
        st.error(f"{tabs_labels[0]}: {mech_err}")
    else:
        render_mech_astro_tab()

with tabs[1]:
    st.markdown(f"# {tabs_labels[1]}")
    if render_optics_combo_tab is None:
        st.error(f"{tabs_labels[1]}: {opt_combo_err}")
    else:
        render_optics_combo_tab()

with tabs[2]:
    st.markdown(f"# {tabs_labels[2]}")
    if render_xray_ct_tab is None:
        st.error(f"{tabs_labels[2]}: {ct_classic_err}")
    else:
        render_xray_ct_tab()

with tabs[3]:
    st.markdown(f"# {tabs_labels[3]}")
    if render_ct_safe_tab is None:
        st.error(f"{tabs_labels[3]}: {ct_safe_err}")
    else:
        render_ct_safe_tab()

with tabs[4]:
    st.markdown(f"# {tabs_labels[4]}")
    if render_mri_bloch_tab is None:
        st.error(f"{tabs_labels[4]}: {mrib_err}")
    else:
        render_mri_bloch_tab()

with tabs[5]:
    st.markdown(f"# {tabs_labels[5]}")
    if render_em_tab is None:
        st.error(f"{tabs_labels[5]}: {em_err}")
    else:
        render_em_tab()

st.markdown("---")

footer_de = "🧩 Physik-Simulator v2.1 | Bilingual Edition | Gebaut mit Streamlit"
footer_en = "🧩 Physics Simulator v2.1 | Bilingual Edition | Built with Streamlit"
footer = footer_de if st.session_state.language == "de" else footer_en

st.markdown(footer)