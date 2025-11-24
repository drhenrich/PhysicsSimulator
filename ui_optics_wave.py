# ============================================================
# ui_optics_wave.py — Streamlit UI for Wave Optics
# Robust image rendering (st.image) + core-less fallback if optics_wave is missing
# Exposes: render_optics_wave_tab()
# ============================================================
from __future__ import annotations

def render_optics_wave_tab():
    import numpy as np
    import streamlit as st

    # ------------------ Try core import; allow fallback if it fails ------------------
    _core_ok = True
    try:
        from optics_wave import Field2D, Aperture, Propagator, Interferometer
    except Exception as e:
        _core_ok = False
        _core_err = e

    st.subheader("Wellenoptik & Fourier-Optik (robustes Rendering)")

    # ------------------ Helper: grayscale image from intensity ------------------
    def as_gray_image(I: np.ndarray) -> np.ndarray:
        I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
        I = np.maximum(I, 0.0)
        I = I / (I.max() + 1e-12)
        img = (255.0 * I).astype(np.uint8)
        return np.repeat(img[..., None], 3, axis=2)  # RGB

    # ------------------ Fallback physics (Fraunhofer, didactic 1D -> 2D) ------------------
    def fallback_spaltlabor(N, lam_nm, z, method, shape, w_mm, p_mm, incoh):
        # Use simple Fraunhofer patterns (sinc^2, cos^2*sinc^2, comb for grating); 2D via outer
        N = int(N)
        lam = lam_nm * 1e-9
        w = w_mm * 1e-3
        p = p_mm * 1e-3
        # 1D frequency coordinate (screen x), normalized domain
        x = np.linspace(-1.0, 1.0, N, endpoint=False)
        # scale argument ~ (pi*w*x/(lam*z)); we keep proportionality for didactics
        arg = np.pi * (w / (lam * max(z,1e-9))) * x
        def sinc(x): 
            y = np.ones_like(x); m = np.abs(x)>1e-12; y[m] = np.sin(x[m]) / x[m]; return y
        if shape == "Einzelspalt":
            I1 = sinc(arg)**2
        elif shape == "Doppelspalt":
            # envelope * interference with period set by p
            arg_p = np.pi * (p / (lam * max(z,1e-9))) * x
            I1 = (np.cos(arg_p))**2 * (sinc(arg))**2
        elif shape == "Gitter":
            # multiple-slit: sharp peaks via cos^2 with higher spatial frequency
            arg_p = 3.0 * np.pi * (p / (lam * max(z,1e-9))) * x
            I1 = (np.cos(arg_p))**2 * (sinc(arg))**2
        elif shape == "Rechteck":
            I1 = sinc(arg)**2
        else:  # Kreis -> airy-like 1D stand-in
            arg_r = 1.22 * np.pi * (w / (lam * max(z,1e-9))) * np.abs(x)
            I1 = (np.sinc(arg_r/np.pi))**2
        # partial coherence: blend with blurred version
        I = np.outer(I1, I1)
        if incoh < 1.0:
            try:
                from scipy.ndimage import gaussian_filter
                I = incoh*I + (1.0-incoh)*gaussian_filter(I, sigma=max(1.0, (1.0-incoh)*0.1*N))
            except Exception:
                # fallback: simple box blur
                k = max(1, int((1.0-incoh)*0.03*N))
                if k > 1:
                    K = np.ones((k,k), dtype=float); K /= K.sum()
                    # manual 2D conv (valid)
                    from numpy.lib.stride_tricks import as_strided
                    H,W = I.shape
                    outH, outW = H-k+1, W-k+1
                    s0, s1 = I.strides
                    blocks = as_strided(I, shape=(outH, outW, k, k), strides=(s0, s1, s0, s1))
                    I = np.tensordot(blocks, K, axes=((2,3),(0,1)))
        return I

    def fallback_interferometer(N, lam_nm, dL_mm, kind):
        N = int(N)
        lam = lam_nm * 1e-9
        dL = dL_mm * 1e-3
        x = np.linspace(-1.0, 1.0, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="xy")
        # create fringes along x; phase shift by dL
        phi = 2.0 * np.pi * dL / max(lam, 1e-12)
        if kind.lower().startswith("m"):
            # Michelson: two beams, small tilt for fringes
            tilt = 2.0 * np.pi * 12.0 * X
            U = 1.0 + np.cos(tilt + phi)
        else:
            # MZI: two opposed tilts
            tilt1 = 2.0 * np.pi * 10.0 * X
            tilt2 = -2.0 * np.pi * 10.0 * X
            U = (np.cos(tilt1) + np.cos(tilt2 + phi))**2
        I = U.astype(np.float64)
        I = I / (I.max() + 1e-12)
        return I

    # Tabs
    tab_lab, tab_if = st.tabs(["Spaltlabor", "Interferometer-Explorer"])

    # ---------------- Spaltlabor ----------------
    with tab_lab:
        with st.form("wave_form"):
            N = st.selectbox("Raster N", [256, 384, 512], index=0, key="wo_N")
            lam = st.number_input("Wellenlänge λ [nm]", min_value=300.0, max_value=800.0, value=532.0, step=5.0, key="wo_lam")
            z = st.number_input("Abstand z [m]", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="wo_z")
            method = st.selectbox("Propagation", ["Fraunhofer","Fresnel","ASM"], index=0, key="wo_method")
            shape = st.selectbox("Apertur", ["Einzelspalt","Doppelspalt","Gitter","Rechteck","Kreis"], index=0, key="wo_shape")
            w = st.number_input("Breite/Radius w [mm]", min_value=0.01, max_value=2.0, value=0.1, step=0.01, key="wo_w")
            p = st.number_input("Periode p [mm] (Gitter/Doppel)", min_value=0.02, max_value=2.0, value=0.2, step=0.01, key="wo_p")
            incoh = st.slider("Teilkohärenz μ (0..1)", 0.0, 1.0, 1.0, 0.05, key="wo_mu")
            submitted = st.form_submit_button("Berechnen", use_container_width=True)

        if submitted:
            try:
                if _core_ok:
                    # compute with core
                    lam_m = lam*1e-9
                    F = Field2D.grid(N=int(N), FOV=(3e-3,3e-3), lam=lam_m)
                    if shape == "Einzelspalt":
                        Ap = Aperture.slit(width=w*1e-3)
                    elif shape == "Doppelspalt":
                        Ap = Aperture.double_slit(width=w*1e-3, period=p*1e-3)
                    elif shape == "Gitter":
                        Ap = Aperture.grating(period=p*1e-3, duty=0.5)
                    elif shape == "Rechteck":
                        Ap = Aperture.rect(width=w*1e-3, height=w*1e-3)
                    else:
                        Ap = Aperture.circle(radius=w*1e-3/2.0)
                    U0 = F.plane_wave() * Ap.mask(F.X, F.Y)
                    prop = Propagator(method=method.lower(), lam=lam_m, z=float(z))
                    U1 = prop.propagate(F, U0, mu=float(incoh))
                    I = np.abs(U1)**2
                else:
                    # fallback (no core)
                    I = fallback_spaltlabor(N, lam, z, method, shape, w, p, incoh)
                img = as_gray_image(I)
                st.image(img, caption="Intensität (normiert)", use_container_width=True, channels="RGB", output_format="PNG", clamp=True)
                if not _core_ok:
                    st.warning(f"Core-Fallback aktiv: optics_wave konnte nicht importiert werden ({_core_err}).")
            except Exception as e:
                st.exception(e)
        else:
            st.info("Parameter einstellen und 'Berechnen' klicken.")

    # ---------------- Interferometer ----------------
    with tab_if:
        with st.form("if_form"):
            N2 = st.selectbox("Raster N (IF)", [256, 384, 512], index=0, key="if_N")
            lam2 = st.number_input("λ [nm] (IF)", min_value=300.0, max_value=800.0, value=532.0, step=5.0, key="if_lam")
            arm = st.number_input("Arm-ΔL [mm]", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="if_dL")
            kind = st.selectbox("Interferometer", ["Michelson","MZI"], index=0, key="if_kind")
            submitted2 = st.form_submit_button("Interferenz berechnen", use_container_width=True)

        if submitted2:
            try:
                if _core_ok:
                    lam_m = lam2*1e-9
                    F = Field2D.grid(N=int(N2), FOV=(3e-3,3e-3), lam=lam_m)
                    If = Interferometer(kind=kind, lam=lam_m, dL=arm*1e-3)
                    U = If.interfere(F)
                    I = np.abs(U)**2
                else:
                    I = fallback_interferometer(N2, lam2, arm, kind)
                img = as_gray_image(I)
                st.image(img, caption=f"{kind}-Interferenz", use_container_width=True, channels="RGB", output_format="PNG", clamp=True)
                if not _core_ok:
                    st.warning(f"Core-Fallback aktiv: optics_wave konnte nicht importiert werden ({_core_err}).")
            except Exception as e:
                st.exception(e)
        else:
            st.info("Parameter wählen und 'Interferenz berechnen' klicken.")
