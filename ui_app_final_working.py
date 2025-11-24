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
# TAB-IMPORTS UND ABBRUCHSICHERUNG
# ============================================
try:
    from ui_domain_mech_safe import render_mech_astro_tab
except Exception as e:
    render_mech_astro_tab = None
    mech_err = str(e)

try:
    from ui_domain_optics_combo import render_optics_combo_tab
except Exception as e:
    render_optics_combo_tab = None
    opt_combo_err = str(e)

try:
    from ui_xray_ct import render_xray_ct_tab
except Exception as e:
    render_xray_ct_tab = None
    ct_classic_err = str(e)

try:
    from ui_ct_safe import render_ct_safe_tab
except Exception as e:
    render_ct_safe_tab = None
    ct_safe_err = str(e)

try:
    from ui_domain_mri_bloch_safe import render_mri_bloch_tab
except Exception as e:
    render_mri_bloch_tab = None
    mrib_err = str(e)

try:
    from ui_domain_em_safe import render_em_tab
except Exception as e:
    render_em_tab = None
    em_err = str(e)

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

# ============================================
# SIDEBAR SETUP
# ============================================
sidebar_placeholder = st.sidebar.empty()

# ============================================
# TABS ERSTELLEN
# ============================================
selected_tabs = st.tabs(tabs_labels)

# ============================================
# TAB-INHALTE RENDERN MIT DYNAMISCHER SIDEBAR
# ============================================

# TAB 0: MECHANIK
with selected_tabs[0]:
    st.markdown(f"# {tabs_labels[0]}")
    
    with sidebar_placeholder.container():
        st.markdown(f"## {get_text('settings', st.session_state.language) or 'Einstellungen'}")
        st.markdown("---")
        st.markdown("### 🎯 Mechanik-Szenen")
        preset_mech = st.selectbox("Preset wählen", [""] + list(PRESENTS_MECH.keys()), key="mech_preset")
        colM1, colM2 = st.columns(2)
        
        with colM1:
            if st.button("Laden", key="mech_load_btn"):
                if preset_mech:
                    for k, v in PRESENTS_MECH[preset_mech].items():
                        st.session_state[f"m{k}"] = v
                    st.success(f"✅ {preset_mech} geladen")
        
        with colM2:
            if st.button("Reset", key="mech_reset_btn"):
                default_key = list(PRESENTS_MECH.keys())[0]
                for k, v in PRESENTS_MECH[default_key].items():
                    st.session_state[f"m{k}"] = v
                st.info("🔄 Auf Standard zurückgesetzt")
        st.markdown("---")
    
    if render_mech_astro_tab is None:
        st.error(f"{tabs_labels[0]}: {mech_err}")
    else:
        render_mech_astro_tab()

# TAB 1: OPTIK
with selected_tabs[1]:
    st.markdown(f"# {tabs_labels[1]}")
    
    with sidebar_placeholder.container():
        st.markdown(f"## {get_text('settings', st.session_state.language) or 'Einstellungen'}")
        st.markdown("---")
        st.markdown("### 🌊 Optik-Wellenoptik")
        preset_opt = st.selectbox("Preset wählen", [""] + list(PRESENTS_OPT_WAVE.keys()), key="opt_preset")
        colO1, colO2 = st.columns(2)
        
        with colO1:
            if st.button("Laden", key="opt_load_btn"):
                if preset_opt:
                    for k, v in PRESENTS_OPT_WAVE[preset_opt].items():
                        st.session_state[f"o{k}"] = v
                    st.success(f"✅ {preset_opt} geladen")
        
        with colO2:
            if st.button("Reset", key="opt_reset_btn"):
                default_key = list(PRESENTS_OPT_WAVE.keys())[0]
                for k, v in PRESENTS_OPT_WAVE[default_key].items():
                    st.session_state[f"o{k}"] = v
                st.info("🔄 Auf Standard zurückgesetzt")
        st.markdown("---")
    
    if render_optics_combo_tab is None:
        st.error(f"{tabs_labels[1]}: {opt_combo_err}")
    else:
        render_optics_combo_tab()

# TAB 2: RÖNTGEN/CT KLASSISCH
with selected_tabs[2]:
    st.markdown(f"# {tabs_labels[2]}")
    
    with sidebar_placeholder.container():
        st.markdown(f"## {get_text('settings', st.session_state.language) or 'Einstellungen'}")
        st.markdown("---")
        st.markdown("### 🏥 CT-Parameter (Klassisch)")
        preset_ct_c = st.selectbox("Preset wählen", [""] + list(PRESENTS_CT.keys()), key="ct_classic_preset")
        colC1, colC2 = st.columns(2)
        
        with colC1:
            if st.button("Laden", key="ct_classic_load_btn"):
                if preset_ct_c:
                    for k, v in PRESENTS_CT[preset_ct_c].items():
                        st.session_state[f"cc{k}"] = v
                    st.success(f"✅ {preset_ct_c} geladen")
        
        with colC2:
            if st.button("Reset", key="ct_classic_reset_btn"):
                default_key = list(PRESENTS_CT.keys())[0]
                for k, v in PRESENTS_CT[default_key].items():
                    st.session_state[f"cc{k}"] = v
                st.info("🔄 Auf Standard zurückgesetzt")
        st.markdown("---")
    
    if render_xray_ct_tab is None:
        st.error(f"{tabs_labels[2]}: {ct_classic_err}")
    else:
        render_xray_ct_tab()

# TAB 3: RÖNTGEN/CT SAFE
with selected_tabs[3]:
    st.markdown(f"# {tabs_labels[3]}")
    
    with sidebar_placeholder.container():
        st.markdown(f"## {get_text('settings', st.session_state.language) or 'Einstellungen'}")
        st.markdown("---")
        st.markdown("### 🏥 CT-Parameter (Safe)")
        preset_ct_s = st.selectbox("Preset wählen", [""] + list(PRESENTS_CT.keys()), key="ct_safe_preset")
        colC1, colC2 = st.columns(2)
        
        with colC1:
            if st.button("Laden", key="ct_safe_load_btn"):
                if preset_ct_s:
                    for k, v in PRESENTS_CT[preset_ct_s].items():
                        st.session_state[f"cs{k}"] = v
                    st.success(f"✅ {preset_ct_s} geladen")
        
        with colC2:
            if st.button("Reset", key="ct_safe_reset_btn"):
                default_key = list(PRESENTS_CT.keys())[0]
                for k, v in PRESENTS_CT[default_key].items():
                    st.session_state[f"cs{k}"] = v
                st.info("🔄 Auf Standard zurückgesetzt")
        st.markdown("---")
    
    if render_ct_safe_tab is None:
        st.error(f"{tabs_labels[3]}: {ct_safe_err}")
    else:
        render_ct_safe_tab()

# TAB 4: MRI BLOCH
with selected_tabs[4]:
    st.markdown(f"# {tabs_labels[4]}")
    
    with sidebar_placeholder.container():
        st.markdown(f"## {get_text('settings', st.session_state.language) or 'Einstellungen'}")
        st.markdown("---")
        st.markdown("### 🧲 Bloch-Parameter")
        preset_bloch = st.selectbox("Preset wählen", [""] + list(PRESENTS_BLOCH.keys()), key="bloch_preset")
        colB1, colB2 = st.columns(2)
        
        with colB1:
            if st.button("Laden", key="bloch_load_btn"):
                if preset_bloch:
                    for k, v in PRESENTS_BLOCH[preset_bloch].items():
                        st.session_state[f"b{k}"] = v
                    st.success(f"✅ {preset_bloch} geladen")
        
        with colB2:
            if st.button("Reset", key="bloch_reset_btn"):
                for k, v in PRESENTS_BLOCH["Standard"].items():
                    st.session_state[f"b{k}"] = v
                st.info("🔄 Auf Standard zurückgesetzt")
        st.markdown("---")
    
    if render_mri_bloch_tab is None:
        st.error(f"{tabs_labels[4]}: {mrib_err}")
    else:
        render_mri_bloch_tab()

# TAB 5: ELEKTRODYNAMIK
with selected_tabs[5]:
    st.markdown(f"# {tabs_labels[5]}")
    
    with sidebar_placeholder.container():
        st.markdown(f"## {get_text('settings', st.session_state.language) or 'Einstellungen'}")
        st.markdown("---")
        st.info("ℹ️ Dieses Modul hat keine vordefinierten Presets")
        st.markdown("---")
    
    if render_em_tab is None:
        st.error(f"{tabs_labels[5]}: {em_err}")
    else:
        render_em_tab()

st.markdown("---")

footer_de = "🧩 Physik-Simulator v2.1 | Bilingual Edition | Gebaut mit Streamlit"
footer_en = "🧩 Physics Simulator v2.1 | Bilingual Edition | Built with Streamlit"
footer = footer_de if st.session_state.language == "de" else footer_en

st.markdown(footer)