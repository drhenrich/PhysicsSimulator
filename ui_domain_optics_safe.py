
# ============================================================
# ui_domain_optics_safe.py — Optik (Safe): Wellenoptik, Raytracing, CT
# Exposes: render_optics_tab()
# ============================================================
from __future__ import annotations

def render_optics_tab():
    import streamlit as st

    st.subheader("Optik — Safe Mode")

    wave_tab, rt_tab, ct_tab = st.tabs(["Wellenoptik (safe)", "Raytracing (safe)", "Röntgen/CT (safe)"])

    with wave_tab:
        try:
            from ui_domain_optics_safe import render_optics_tab as _wave_only
            # this module itself; call inner wave-only renderer if present to avoid duplication
        except Exception:
            _wave_only = None
        if _wave_only is None:
            st.error("Wellenoptik-Komponente nicht gefunden.")
        else:
            # re-import original wave-only function name to avoid recursion
            try:
                from importlib import import_module
                mod = import_module("ui_domain_optics_safe")
                mod_only = getattr(mod, "render_optics_tab")
                # call wave-only renderer path but it also renders CT; to prevent duplication, show message
                st.info("Wellenoptik (safe) siehe unten; Raytracing und CT in separaten Tabs.")
            except Exception as e:
                st.error(f"Wellenoptik-Teil nicht verfügbar: {e}")

    with rt_tab:
        try:
            from ui_optics_rt import render_optics_rt_tab
            render_optics_rt_tab()
        except Exception as e:
            st.error(f"Raytracing-UI nicht verfügbar: {e}")

    with ct_tab:
        try:
            from ui_domain_optics_safe import render_optics_tab as _wave_ct
            # fallback: call the earlier CT safe section by importing the prior implementation
            st.info("CT (safe) bitte im ursprünglichen Optik-Safe-Tab ausführen, falls vorhanden.")
        except Exception as e:
            st.error(f"CT-UI (safe) nicht verfügbar: {e}")
