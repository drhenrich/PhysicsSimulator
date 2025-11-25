from __future__ import annotations
import streamlit as st
from i18n_bundle import get_text, get_language_name
from sim_core_bundle import Simulator, PRESETS_ENHANCED, PRESENTS_BLOCH, PRESENTS_OPT_WAVE, PRESENTS_CT, PRESENTS_MECH
from sim_core_bundle import plot_trajectory_3d, plot_conservation_laws, plot_collision_analysis
from ui_mech_bundle import render_mech_astro_tab
from ui_optics_bundle import render_optics_combo_tab
from ui_med_bundle import render_ct_safe_tab, render_xray_ct_tab, render_mri_bloch_tab, render_em_tab
from ui_thermo_bundle import render_thermo_tab
from ui_atom_bundle import render_atom_tab
from ui_ultrasound import render_ultrasound_tab

st.set_page_config(page_title=get_text("title", "de"), page_icon="🧩", layout="wide", initial_sidebar_state="expanded")

if "language" not in st.session_state:
    st.session_state.language = "de"

st.markdown("""
<style>
.main { padding: 2rem; }
h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
h2 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col1: st.write("")
with col2:
    title = get_text("title", st.session_state.language) or "Physics Teaching Simulator"
    st.markdown(f"# {title}")
    subtitle = get_text("subtitle", st.session_state.language) or "Interaktive Simulation"
    st.markdown(f"**{subtitle}**")
with col3:
    lang_choice = st.selectbox("🌍", ["de", "en"], format_func=lambda x: get_language_name(x), index=0 if st.session_state.language=="de" else 1, key="lang_selector")
if lang_choice != st.session_state.language:
    st.session_state.language = lang_choice
    st.rerun()

st.markdown("---")
lang = st.session_state.language

tabs_labels = [
    get_text("mechanics", lang) or "Mechanik & Himmelsmechanik",
    get_text("thermodynamics", lang) or "Thermodynamik",
    get_text("atom_physics", lang) or "Atomphysik",
    get_text("optics", lang) or "Optik",
    get_text("xray_ct_classic", lang) or "Xray/CT",
    get_text("mri_imaging", lang) or "MRI & Bloch",
    get_text("ultrasound", lang) or "Ultraschall",
    get_text("electromagnetism", lang) or "Elektrodynamik"
]
selected_tabs = st.tabs(tabs_labels)

# Tab 0: Mechanics
with selected_tabs[0]:
    st.markdown(f"# {tabs_labels[0]}")
    render_mech_astro_tab()
    st.markdown(f"### {get_text('adv_mech_title', lang)}")
    preset_adv = st.selectbox(get_text("adv_preset_select", lang), [""] + list(PRESETS_ENHANCED.keys()), key="mech_adv_preset")
    colE1, colE2 = st.columns([1,1])
    with colE1:
        restitution = st.slider(get_text("restitution", lang), 0.0, 1.0, 1.0, 0.05, key="mech_adv_restitution")
    with colE2:
        drag = st.slider(get_text("drag", lang), 0.0, 1.0, 0.0, 0.01, key="mech_adv_drag")
    if st.button(get_text("adv_run", lang), key="mech_adv_run", use_container_width=True):
        if not preset_adv:
            st.warning(get_text("choose_preset_warning", lang))
        else:
            bodies, connections, note = PRESETS_ENHANCED[preset_adv]()
            sim = Simulator(bodies=bodies, connections=connections, restitution=restitution, drag=drag)
            results = sim.run()
            st.success(get_text("adv_success", lang).format(preset=preset_adv, note=note))
            fig_traj = plot_trajectory_3d(bodies, results)
            if fig_traj is not None: st.plotly_chart(fig_traj, use_container_width=True)
            fig_energy = plot_conservation_laws(bodies, results)
            if fig_energy is not None: st.plotly_chart(fig_energy, use_container_width=True)
            if sim.collision_events:
                fig_col = plot_collision_analysis(sim.collision_events)
                if fig_col is not None: st.plotly_chart(fig_col, use_container_width=True)

# Tab 1: Thermodynamics (NEU!)
with selected_tabs[1]:
    st.markdown(f"# {tabs_labels[1]}")
    render_thermo_tab()

# Tab 2: Atom Physics (NEU!)
with selected_tabs[2]:
    st.markdown(f"# {tabs_labels[2]}")
    render_atom_tab()

# Tab 3: Optics
with selected_tabs[3]:
    st.markdown(f"# {tabs_labels[3]}")
    render_optics_combo_tab()

# Tab 4: Xray/CT combined
with selected_tabs[4]:
    st.markdown(f"# {tabs_labels[4]}")
    subtab_classic, subtab_safe = st.tabs(["Klassisch", "Reduziert"])
    with subtab_classic:
        render_xray_ct_tab()
    with subtab_safe:
        render_ct_safe_tab()

# Tab 5: MRI & Bloch
with selected_tabs[5]:
    st.markdown(f"# {tabs_labels[5]}")
    render_mri_bloch_tab()

# Tab 6: Ultrasound (NEU!)
with selected_tabs[6]:
    st.markdown(f"# {tabs_labels[6]}")
    render_ultrasound_tab()

# Tab 7: Electrodynamics
with selected_tabs[7]:
    st.markdown(f"# {tabs_labels[7]}")
    render_em_tab()

st.markdown("---")
st.markdown("🧩 Physics Simulator | Bilingual Edition | v4.0 mit Mechanik, Thermodynamik, Atomphysik & Ultraschall")
