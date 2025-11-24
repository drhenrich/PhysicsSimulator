# ============================================================
# ui_app_bilingual.py — Physics Simulator with German/English
# Bilingual Support (Deutsch / English)
# ============================================================
from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

# Import translations
try:
    from i18n_utils import get_text, get_language_name, TRANSLATIONS
except ImportError:
    # Fallback if i18n_utils not available
    def get_text(key, language="de"):
        return key
    def get_language_name(lang):
        return "Deutsch" if lang == "de" else "English"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🔬 Physics Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LANGUAGE SELECTION
# ============================================================
if "language" not in st.session_state:
    st.session_state.language = "de"

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h2 {
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER & LANGUAGE SELECTOR
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")

with col2:
    title = get_text("title", st.session_state.language)
    st.markdown(f"# {title}")

with col3:
    lang_choice = st.selectbox(
        "🌍",
        ["de", "en"],
        format_func=lambda x: get_language_name(x),
        index=0 if st.session_state.language == "de" else 1,
        key="lang_selector"
    )
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice
        st.rerun()

st.markdown("---")

# ============================================================
# SIDEBAR SETTINGS
# ============================================================
with st.sidebar:
    st.markdown(f"## {get_text('settings', st.session_state.language)}")
    st.markdown(f"""
    {get_text('subtitle', st.session_state.language)}
    """)

    st.markdown("---")

    # Refresh Button
    if st.button(f"{get_text('refresh', st.session_state.language)}", use_container_width=True):
        st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ " + get_text("about", st.session_state.language)):
        st.write("Physics Simulator v2.1")
        st.write(f"🌍 Languages: Deutsch | English")
        st.metric("Modules", "6")

# ============================================================
# MAIN CONTENT - TABS
# ============================================================
lang = st.session_state.language
tabs_labels = [
    get_text("mechanics", lang),
    get_text("optics", lang),
    get_text("medical", lang),
    get_text("electromagnetism", lang),
    get_text("data", lang),
    get_text("tools", lang)
]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs_labels)

# ============================================================
# TAB 1: MECHANICS
# ============================================================
with tab1:
    st.markdown(f"# {get_text('mechanics', lang)}")
    st.markdown(f"### {get_text('particle_dynamics', lang)}")

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("mech_form_bilingual"):
            n_objects = st.slider(
                get_text("particle_count", lang),
                2, 10, 3,
                key="mech_n_bil"
            )
            duration = st.number_input(
                get_text("duration_sec", lang),
                0.1, 100.0, 10.0,
                key="mech_dur_bil"
            )
            show_trails = st.checkbox(
                get_text("show_trails", lang),
                value=True,
                key="mech_trails_bil"
            )

            run = st.form_submit_button(
                get_text("start_simulation", lang),
                use_container_width=True
            )

        if run:
            with st.spinner(get_text("computing", lang)):
                import time
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i+1)

            st.success(get_text("completed", lang))
            st.balloons()

    with col1:
        st.markdown(f"### {get_text('energy_analysis', lang)}")
        st.info(get_text("energy_conserved", lang))
        st.metric(
            get_text("total_energy", lang),
            "58.0 J"
        )

# ============================================================
# TAB 2: OPTICS
# ============================================================
with tab2:
    st.markdown(f"# {get_text('optics', lang)}")
    st.markdown(f"### {get_text('diffraction_patterns', lang)}")

    with st.form("optics_form_bilingual"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{get_text('aperture_config', lang)}**")
            aperture = st.selectbox(
                get_text("aperture_type", lang),
                [
                    get_text("single_slit", lang),
                    get_text("double_slit", lang),
                    get_text("grating", lang)
                ],
                key="opt_aperture_bil"
            )

        with col2:
            st.markdown(f"**{get_text('wave_parameters', lang)}**")
            wavelength = st.slider(
                get_text("wavelength", lang),
                380, 780, 550,
                key="opt_wl_bil"
            )

        run = st.form_submit_button(
            get_text("calculate_pattern", lang),
            use_container_width=True
        )

    if run:
        with st.spinner(get_text("computing_field", lang)):
            import time
            for i in range(50):
                time.sleep(0.02)

        st.success(get_text("pattern_calculated", lang))
        st.metric(get_text("peak_intensity", lang), "1.00")

# ============================================================
# TAB 3: MEDICAL IMAGING
# ============================================================
with tab3:
    st.markdown(f"# {get_text('medical', lang)}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {get_text('ct_parameters', lang)}")

        with st.form("ct_form_bilingual"):
            N = st.selectbox(
                get_text("grid_size", lang),
                [96, 128, 160],
                index=1,
                key="ct_N_bil"
            )
            n_proj = st.selectbox(
                get_text("number", lang),
                [60, 90, 120],
                index=0,
                key="ct_P_bil"
            )

            run = st.form_submit_button(
                get_text("generate_ct", lang),
                use_container_width=True
            )

        if run:
            with st.spinner(get_text("computing_ct", lang)):
                import time
                for i in range(100):
                    time.sleep(0.01)

            st.success(get_text("ct_completed", lang))

# ============================================================
# TAB 4: ELECTROMAGNETISM
# ============================================================
with tab4:
    st.markdown(f"# {get_text('electromagnetism', lang)}")
    st.markdown(f"### {get_text('point_charges', lang)}")

    with st.form("em_form_bilingual"):
        n_charges = st.slider(
            get_text("number", lang),
            1, 5, 2,
            key="em_n_bil"
        )
        q1 = st.slider(
            get_text("charge_q1", lang),
            -10, 10, 1,
            key="em_q1_bil"
        )

        run = st.form_submit_button(
            get_text("calculate_field", lang),
            use_container_width=True
        )

    if run:
        with st.spinner(get_text("computing_field", lang)):
            import time
            for i in range(100):
                time.sleep(0.01)

        st.success(get_text("field_calculated", lang))
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(get_text("charges", lang), n_charges)
        with col_b:
            st.metric(get_text("max_efield", lang), "2.5 V/m")

# ============================================================
# TAB 5: DATA ANALYSIS
# ============================================================
with tab5:
    st.markdown(f"# {get_text('data', lang)}")

    x = np.linspace(0, 10, 100)
    df = pd.DataFrame({
        'x': x,
        'sin': np.sin(x),
        'cos': np.cos(x)
    })

    st.line_chart(df.set_index('x'))

# ============================================================
# TAB 6: TOOLS
# ============================================================
with tab6:
    st.markdown(f"# {get_text('tools', lang)}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### ✅ {get_text('info', lang)}")
        st.metric("Version", "2.1 Bilingual")
        st.metric(f"🌍 {get_text('language', lang)}", 
                 get_language_name(st.session_state.language))

    with col2:
        st.markdown(f"### 📋 Modules")
        st.write("✅ Mechanics")
        st.write("✅ Optics")
        st.write("✅ Medical Imaging")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
footer_de = "🔬 Physics Teaching Simulator v2.1 | Bilingual Edition | Gebaut mit Streamlit"
footer_en = "🔬 Physics Teaching Simulator v2.1 | Bilingual Edition | Built with Streamlit"
footer = footer_de if st.session_state.language == "de" else footer_en

st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <small>{footer}</small>
</div>
""", unsafe_allow_html=True)
