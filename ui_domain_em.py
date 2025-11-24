
# ============================================================
# ui_domain_em.py — Elektrodynamik (didaktisch, 2D Elektrostatik)
# Exposes: render_em_tab()
# ============================================================
from __future__ import annotations

def render_em_tab():
    import numpy as np
    import streamlit as st

    st.subheader("Elektrodynamik — 2D Elektrostatik (didaktisch)")

    def render(I, caption):
        I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
        I = I - I.min(); I = I/(I.max()+1e-12)
        img = (np.repeat(I[...,None],3,axis=2)*255).astype("uint8")
        st.image(img, caption=caption, use_container_width=True)

    with st.form("em_form"):
        N = st.selectbox("Raster N", [96, 128, 192, 256], index=1, key="em_N")
        nQ = st.selectbox("Anzahl Punktladungen", [1,2,3], index=1, key="em_nq")
        q1 = st.slider("q1 [a.u.]", -2.0, 2.0, 1.0, 0.1, key="em_q1")
        q2 = st.slider("q2 [a.u.]", -2.0, 2.0, -1.0, 0.1, key="em_q2") if nQ>=2 else 0.0
        q3 = st.slider("q3 [a.u.]", -2.0, 2.0, 1.0, 0.1, key="em_q3") if nQ>=3 else 0.0
        run = st.form_submit_button("Feld berechnen", use_container_width=True)

    if run:
        N=int(N)
        Y,X = np.indices((N,N))
        x = (X - N/2)/(N/2); y = (Y - N/2)/(N/2)
        eps = 1e-3
        charges = [(0.3,0.0,q1)]
        if nQ>=2: charges.append((-0.3,0.0,q2))
        if nQ>=3: charges.append((0.0,0.3,q3))
        phi = np.zeros((N,N), dtype=float)
        Ex = np.zeros_like(phi); Ey = np.zeros_like(phi)
        for cx, cy, q in charges:
            rx = x - cx; ry = y - cy; r2 = rx*rx + ry*ry + eps**2; r = np.sqrt(r2)
            phi += q / r
            Ex += q * rx / (r2*r)
            Ey += q * ry / (r2*r)
        render(phi, "Potential (normiert)")
        Emag = np.sqrt(Ex**2 + Ey**2)
        render(Emag, "|E| (normiert)")
    else:
        st.info("Parameter wählen und 'Feld berechnen' klicken.")
