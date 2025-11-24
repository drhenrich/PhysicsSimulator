from __future__ import annotations

import streamlit as st

# Übersetzungsfunktionen aus i18n_utils.py importieren
from i18n_utils import get_text, get_language_name
from presets import PRESENTS_BLOCH, PRESENTS_OPT_WAVE, PRESENTS_CT, PRESENTS_MECH
from core import Simulator
from plotting import plot_trajectory_3d, plot_conservation_laws, plot_collision_analysis
from scenarios_enhanced import PRESETS_ENHANCED

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
    title = get_text("title", st.session_state.language) or "Physics Teaching Simulator — Hauptgebiete"
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
    get_text("mechanics", lang) or "Mechanik & Himmelsmechanik",
    get_text("optics", lang) or "Optik (Wellenoptik + Raytracing)",
    get_text("xray_ct_classic", lang) or "Xray/CT",
    get_text("mri_imaging", lang) or "MRI & Bloch",
    get_text("electromagnetism", lang) or "Elektrodynamik & Potential"
]


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
    
    if render_mech_astro_tab is None:
        st.error(f"{tabs_labels[0]}: {mech_err}")
    else:
        render_mech_astro_tab()

    # Erweiterte Presets (nutzt core.Simulator)
    st.markdown(f"### {get_text('adv_mech_title', st.session_state.language)}")
    preset_adv = st.selectbox(get_text("adv_preset_select", st.session_state.language), [""] + list(PRESETS_ENHANCED.keys()), key="mech_adv_preset")
    colE1, colE2 = st.columns([1,1])
    with colE1:
        restitution = st.slider(get_text("restitution", st.session_state.language), 0.0, 1.0, 1.0, 0.05, key="mech_adv_restitution")
    with colE2:
        drag = st.slider(get_text("drag", st.session_state.language), 0.0, 1.0, 0.0, 0.01, key="mech_adv_drag")
    if st.button(get_text("adv_run", st.session_state.language), key="mech_adv_run", use_container_width=True):
        if not preset_adv:
            st.warning(get_text("choose_preset_warning", st.session_state.language))
        else:
            bodies, connections, note = PRESETS_ENHANCED[preset_adv]()
            sim = Simulator(bodies=bodies, connections=connections, restitution=restitution, drag=drag)
            results = sim.run()
            st.success(get_text("adv_success", st.session_state.language).format(preset=preset_adv, note=note))
            # Trajektorie
            try:
                fig_traj = plot_trajectory_3d(bodies, results)
                st.plotly_chart(fig_traj, use_container_width=True)
            except Exception:
                st.write("Trajektorien:", results.get("positions"))
            # Energie
            try:
                fig_energy = plot_conservation_laws(bodies, results)
                st.plotly_chart(fig_energy, use_container_width=True)
            except Exception:
                st.write("Energie:", results.get("energies"))
            # Kollisionen
            if sim.collision_events:
                try:
                    fig_col = plot_collision_analysis(sim.collision_events)
                    st.plotly_chart(fig_col, use_container_width=True)
                except Exception:
                    st.write("Kollisionen:", sim.collision_events)

# TAB 1: OPTIK
with selected_tabs[1]:
    st.markdown(f"# {tabs_labels[1]}")
    
    if render_optics_combo_tab is None:
        st.error(f"{tabs_labels[1]}: {opt_combo_err}")
    else:
        render_optics_combo_tab()

# TAB 2: RÖNTGEN/CT (kombiniert)
with selected_tabs[2]:
    st.markdown(f"# {tabs_labels[2]}")
    subtab_classic, subtab_safe = st.tabs(["Klassisch", "Reduziert"])
    with subtab_classic:
        if render_xray_ct_tab is None:
            st.error(f"{tabs_labels[2]}: {ct_classic_err}")
        else:
            render_xray_ct_tab()
    with subtab_safe:
        if render_ct_safe_tab is None:
            st.error(f"{tabs_labels[2]}: {ct_safe_err}")
        else:
            render_ct_safe_tab()

# TAB 3: MRI BLOCH
with selected_tabs[3]:
    st.markdown(f"# {tabs_labels[3]}")
    
    if render_mri_bloch_tab is None:
        st.error(f"{tabs_labels[4]}: {mrib_err}")
    else:
        render_mri_bloch_tab()

# TAB 4: ELEKTRODYNAMIK
with selected_tabs[4]:
    st.markdown(f"# {tabs_labels[4]}")
    if render_em_tab is None:
        st.error(f"{tabs_labels[4]}: {em_err}")
    else:
        render_em_tab()

st.markdown("---")

footer_de = "🧩 Physik-Simulator v2.1 | Bilingual Edition | Gebaut mit Streamlit"
footer_en = "🧩 Physics Simulator v2.1 | Bilingual Edition | Built with Streamlit"
footer = footer_de if st.session_state.language == "de" else footer_en

st.markdown(footer)
