
# ============================================================
# ui_mri.py — Streamlit UI for MRI (didactic core)
# Exposes: render_mri_tab()
# ============================================================
from __future__ import annotations

def render_mri_tab():
    import numpy as np
    import streamlit as st
    try:
        import plotly.graph_objects as go
    except Exception:
        go = None

    try:
        from mri_core import shepp_phantom, t1t2_maps, steady_state, acquire_cartesian, reconstruct
    except Exception as e:
        st.error(f"MRI-Core nicht verfügbar: {e}")
        return

    st.subheader("MRI — Sequenz & Kontrast (didaktisch)")

    tab_kontrast, tab_seq = st.tabs(["Kontrast-Explorer", "Sequenz-Designer"])

    with tab_kontrast:
        with st.form("mri_contrast"):
            N = st.selectbox("Matrix N", [128, 192, 256], index=0, key="mri_N")
            TR = st.number_input("TR [ms]", min_value=50.0, max_value=4000.0, value=1000.0, step=50.0, key="mri_TR")
            TE = st.number_input("TE [ms]", min_value=5.0, max_value=200.0, value=40.0, step=5.0, key="mri_TE")
            seq = st.selectbox("Sequenz", ["Spin-Echo","Gradient-Echo"], index=0, key="mri_seq")
            noise = st.slider("k-Space Rauschen", 0.0, 0.1, 0.0, 0.005, key="mri_noise")
            run = st.form_submit_button("Simulieren", use_container_width=True)
        if run:
            try:
                N = int(N)
                rho = shepp_phantom(N)
                T1, T2 = t1t2_maps(N)
                S = steady_state(SE=(seq.startswith("Spin")), TR_ms=TR, TE_ms=TE, T1=T1, T2=T2, rho=rho)
                k = acquire_cartesian(S, noise=noise)
                im = reconstruct(k)
                im = im/ (im.max()+1e-12)

                # Render robust via st.image; show k-space magnitude as well
                im_rgb = (np.stack([im, im, im], axis=2)*255).astype("uint8")
                kmag = np.log10(1 + np.abs(k)); kmag = kmag / (kmag.max()+1e-12)
                kmag_rgb = (np.stack([kmag, kmag, kmag], axis=2)*255).astype("uint8")
                st.image(im_rgb, caption="Rekonstruktion |ρ_eff| (normiert)", use_container_width=True)
                st.image(kmag_rgb, caption="k-Space |K| (log)", use_container_width=True)
            except Exception as e:
                st.exception(e)
        else:
            st.info("Parameter wählen und 'Simulieren' klicken.")

    with tab_seq:
        st.write("Vereinfachter Sequenz-Designer (didaktisch). Erweiterbar um echte Puls-/Gradienten-Timelines.")
        with st.form("mri_seqform"):
            TR = st.number_input("TR [ms] (Seq)", min_value=50.0, max_value=4000.0, value=800.0, step=25.0, key="mri_TR2")
            TE = st.number_input("TE [ms] (Seq)", min_value=5.0, max_value=200.0, value=20.0, step=5.0, key="mri_TE2")
            seq = st.selectbox("Sequenz (Seq)", ["Spin-Echo","Gradient-Echo"], index=0, key="mri_seq2")
            run2 = st.form_submit_button("Signal berechnen", use_container_width=True)
        if run2:
            try:
                rho = shepp_phantom(64)
                T1, T2 = t1t2_maps(64)
                S = steady_state(SE=(seq.startswith("Spin")), TR_ms=TR, TE_ms=TE, T1=T1, T2=T2, rho=rho)
                st.write(f"Mittelwert |Signal|: {float(np.mean(np.abs(S))):.3f}")
            except Exception as e:
                st.exception(e)
        else:
            st.info("TR/TE einstellen und 'Signal berechnen' klicken.")
