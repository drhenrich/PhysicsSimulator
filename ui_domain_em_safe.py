
# ============================================================
# ui_domain_em_safe.py — Elektrodynamik & Potential (100% st.image, no deps)
# Exposes: render_em_tab()
# ============================================================
from __future__ import annotations

def render_em_tab():
    import numpy as np
    import streamlit as st
    try:
        from i18n_utils import get_text
        t = lambda key, de, en: get_text(key, st.session_state.get("language", "de"))
    except Exception:
        t = lambda key, de, en: de if st.session_state.get("language", "de") == "de" else en
    tr = lambda de, en: de if st.session_state.get("language", "de") == "de" else en

    st.subheader(tr("Elektrodynamik & Potential", "Electrodynamics & Potential"))

    es_tab, pot_tab, preset_tab = st.tabs([
        tr("Elektrostatik (Punktladungen)", "Electrostatics (point charges)"),
        tr("Potentialfeld (Poisson)", "Potential field (Poisson)"),
        tr("Presets & Simulation", "Presets & Simulation")
    ])

    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min(); A = A/(A.max()+1e-12)
        return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")

    # ---------- Elektrostatik ----------
    def compute_field(N_val, nQ_val, q1v, q2v, q3v):
        Y, X = np.indices((N_val, N_val), dtype=float)
        x = (X - N_val/2) / (N_val/2)
        y = (Y - N_val/2) / (N_val/2)
        eps = 1e-3
        charges = [(0.3,0.0,q1v)]
        if nQ_val >= 2: charges.append((-0.3,0.0,q2v))
        if nQ_val >= 3: charges.append((0.0,0.3,q3v))
        phi = np.zeros_like(x)
        Ex = np.zeros_like(phi); Ey = np.zeros_like(phi)
        for cx, cy, q in charges:
            rx = x - cx; ry = y - cy; r2 = rx*rx + ry*ry + eps**2; r = np.sqrt(r2)
            phi += q / r
            Ex += q * rx / (r2*r)
            Ey += q * ry / (r2*r)
        Emag = np.sqrt(Ex**2 + Ey**2)
        return phi, Emag

    with es_tab:
        with st.form("em_es"):
            N = st.selectbox(tr("Raster N", "Grid N"), [128, 192, 256], index=1, key="esN_form")
            nQ = st.selectbox(tr("Anzahl Punktladungen", "Number of point charges"), [1,2,3], index=2, key="esnQ_form")
            q1 = st.slider("q1", -2.0, 2.0, 1.0, 0.1, key="esq1_form")
            q2 = st.slider("q2", -2.0, 2.0, -1.0, 0.1, key="esq2_form") if nQ>=2 else 0.0
            q3 = st.slider("q3", -2.0, 2.0, 1.0, 0.1, key="esq3_form") if nQ>=3 else 0.0
            run = st.form_submit_button(tr("Feld berechnen", "Compute field"))
        if run:
            phi, Emag = compute_field(int(N), int(nQ), q1, q2 if nQ>=2 else 0.0, q3 if nQ>=3 else 0.0)
            st.image(to_img(phi), caption=tr("Potential (normiert)", "Potential (normalized)"), use_container_width=True)
            st.image(to_img(Emag), caption=tr("|E| (normiert)", "|E| (normalized)"), use_container_width=True)
        else:
            st.info(tr("Parameter wählen und 'Feld berechnen' klicken.", "Choose parameters and click 'Compute field'."))

    # ---------- Potentialfeld (Poisson) ----------
    with pot_tab:
        with st.form("em_pot"):
            N = st.selectbox(tr("Raster N (Poisson)", "Grid N (Poisson)"), [96, 128, 160], index=1, key="potN")
            iters = st.selectbox(tr("Iterationsschritte", "Iterations"), [100, 250, 500, 1000], index=2, key="potIter")
            runP = st.form_submit_button(tr("Poisson lösen", "Solve Poisson"))
        if runP:
            N=int(N); iters=int(iters)
            rho = np.zeros((N,N), dtype=float)
            # Zwei entgegengesetzte Ladungsinseln als Quelle
            rho[N//3, N//3] = +1.0
            rho[2*N//3, 2*N//3] = -1.0
            phi = np.zeros_like(rho)
            # Jacobi-Iteration mit Dirichlet-Rand phi=0
            for _ in range(iters):
                phi = 0.25*(np.roll(phi,1,0)+np.roll(phi,-1,0)+np.roll(phi,1,1)+np.roll(phi,-1,1) - rho)
                phi[0,:]=phi[-1,:]=phi[:,0]=phi[:,-1]=0.0
            st.image(to_img(phi), caption=tr(f"Poisson-Lösung nach {iters} Iterationen (normiert)", f"Poisson solution after {iters} iterations (normalized)"), use_container_width=True)
        else:
            st.info(tr("Parameter wählen und 'Poisson lösen' klicken.", "Choose parameters and click 'Solve Poisson'."))

    # ---------- Presets & Simulation ----------
    with preset_tab:
        st.markdown(tr("Wähle ein Elektrodynamik-Preset und starte direkt.", "Choose an electrodynamics preset and run."))
        em_presets = {
            "Einzelladung": {"N": 192, "nQ": 1, "q1": 1.0, "q2": 0.0, "q3": 0.0},
            "Dipol": {"N": 192, "nQ": 2, "q1": 1.5, "q2": -1.5, "q3": 0.0},
            "Quadrupol": {"N": 192, "nQ": 3, "q1": 1.5, "q2": -1.5, "q3": 1.5},
        }
        preset_name = st.selectbox(tr("Preset wählen", "Select preset"), [""] + list(em_presets.keys()), key="em_preset_tab")
        if st.button(tr("Simulation starten", "Start simulation"), key="em_preset_run"):
            if not preset_name:
                st.warning(tr("Bitte ein Preset auswählen.", "Please select a preset."))
            else:
                cfg = em_presets[preset_name]
                phi, Emag = compute_field(int(cfg["N"]), int(cfg["nQ"]), cfg["q1"], cfg["q2"], cfg["q3"])
                st.success(tr(f"{preset_name} simuliert.", f"{preset_name} simulated."))
                st.image(to_img(phi), caption=tr("Potential (normiert)", "Potential (normalized)"), use_container_width=True)
                st.image(to_img(Emag), caption=tr("|E| (normiert)", "|E| (normalized)"), use_container_width=True)
