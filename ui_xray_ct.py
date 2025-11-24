
# ============================================================
# ui_xray_ct.py — Fast & robust X-ray / CT UI
# - Image-based rendering (st.image)
# - Capped grid sizes and projections
# - Progress bar + soft time budget
# Exposes: render_xray_ct_tab()
# ============================================================
from __future__ import annotations

def render_xray_ct_tab():
    import time
    import numpy as np
    import streamlit as st

    try:
        from xray_ct import (
            Spectra, MaterialDB, Projector, Reconstructor,
            shepp_logan, cylinders_phantom, forward_intensity, to_hu
        )
    except Exception as e:
        st.error(f"Xray/CT-Core nicht verfügbar: {e}")
        return

    st.subheader("Röntgen / CT-Physik (Schnellmodus)")

    def to_gray_img(A: np.ndarray, cmap: str = "gray", invert: bool = False):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min()
        A = A / (A.max() + 1e-12)
        if invert:
            A = 1.0 - A
        img = (A * 255.0).astype("uint8")
        return np.repeat(img[..., None], 3, axis=2)

    with st.form("ct_form_fast"):
        col1, col2 = st.columns([2,1], gap="large")
        with col2:
            phantom = st.selectbox("Phantom", ["Shepp-Logan", "Zylinder (Wasser/Knochen/Luft)"], index=0, key="ct_phantom")
            N = st.selectbox("Bildgröße N", [128, 192, 256], index=0, key="ct_N")  # default 128 for speed
            geom = st.selectbox("Geometrie", ["Parallel"], index=0, key="ct_geom")  # limit to parallel in fast mode
            n_det = st.number_input("Detektorelemente", min_value=90, max_value=720, value=180, step=10, key="ct_ndet")
            n_proj = st.number_input("Projektionen", min_value=30, max_value=720, value=120, step=30, key="ct_nproj")
            kVp = st.number_input("kVp", min_value=40.0, max_value=140.0, value=80.0, step=2.0, key="ct_kvp")
            filt = st.number_input("Filtration [mm Al]", min_value=0.0, max_value=10.0, value=2.5, step=0.5, key="ct_filt")
            poly = st.checkbox("Polychromatisch", value=False, key="ct_poly")  # off by default for speed
            noise = st.slider("Detektor-Rauschen (σ)", min_value=0.0, max_value=0.05, value=0.0, step=0.005, key="ct_noise")
            budget = st.slider("Zeitbudget [s]", min_value=1, max_value=10, value=4, step=1, key="ct_budget")
            run = st.form_submit_button("Simulieren (schnell)", use_container_width=True)

    if run:
        start_t = time.time()
        try:
            with st.spinner("Simuliere CT (Schnellmodus)..."):
                N = int(N)
                # Cap sizes hard to guarantee completion
                n_det = min(int(n_det), 256)
                n_proj = min(int(n_proj), 180)

                mu_true = shepp_logan(N) if phantom.startswith("Shepp") else cylinders_phantom(N)
                st.image(to_gray_img(mu_true, invert=True), caption="Phantom (μ, normiert)", use_container_width=True)

                proj = Projector(geometry="parallel")
                db = MaterialDB()
                spec = Spectra(kVp=float(kVp), filtration_mm_Al=float(filt))

                labels = None
                if (not phantom.startswith("Shepp")) and poly:
                    labels = {"air": 0.001, "water": 0.20, "bone": 0.45}

                # Progressive forward model: compute in chunks to allow progress + time budget
                I = np.zeros((n_proj, n_det), dtype=np.float32)
                thetas = np.linspace(0.0, np.pi, n_proj, endpoint=False)
                dets = np.linspace(-1.0, 1.0, n_det, endpoint=False)
                pb = st.progress(0.0, text="Erzeuge Sinogramm...")
                for ti, th in enumerate(thetas):
                    if time.time() - start_t > budget:
                        st.warning(f"Zeitbudget erreicht nach {ti}/{n_proj} Projektionen.")
                        thetas = thetas[:ti]  # truncate angles to computed part
                        I = I[:ti]
                        break
                    # one-row forward by calling the core but limiting energies
                    # We call the full forward_intensity only once per batch would be heavy;
                    # here we emulate a per-angle call by slicing after full calc if possible.
                    # For simplicity and robustness: compute one angle via small wrapper:
                    # --- Small inline line-integral (parallel) ---
                    d = np.array([np.cos(th), np.sin(th)], dtype=float)
                    n = np.array([-np.sin(th), np.cos(th)], dtype=float)
                    # energy handling
                    if poly:
                        Es, w = spec.sample_spectrum(3)
                        mu_scale = db.mu_water(Es); mu_scale = mu_scale/(mu_scale.max()+1e-12)
                    else:
                        Es, w = [spec.effective_energy()], [1.0]
                        mu_scale = np.array([1.0])
                    row = np.zeros(n_det, dtype=np.float32)
                    # Lo-res steps to stay fast
                    steps = max(64, N//2)
                    for si, s in enumerate(dets):
                        p0 = -np.sqrt(2.0)*d + s*n
                        p1 =  np.sqrt(2.0)*d + s*n
                        val = 0.0
                        for k in range(len(Es)):
                            # inline ray integral
                            t = np.linspace(0.0, 1.0, steps)
                            x = p0[0] + (p1[0] - p0[0])*t
                            y = p0[1] + (p1[1] - p0[1])*t
                            # bilinear sample
                            gx = (x*0.5 + 0.5)*(N-1)
                            gy = (y*0.5 + 0.5)*(N-1)
                            x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
                            x1 = np.clip(x0+1, 0, N-1); y1 = np.clip(y0+1, 0, N-1)
                            x0 = np.clip(x0, 0, N-1); y0 = np.clip(y0, 0, N-1)
                            wx = gx - x0; wy = gy - y0
                            Ia = mu_true[y0, x0]; Ib = mu_true[y0, x1]; Ic = mu_true[y1, x0]; Id = mu_true[y1, x1]
                            mus = (1-wx)*(1-wy)*Ia + wx*(1-wy)*Ib + (1-wx)*wy*Ic + wx*wy*Id
                            line = mus.mean() * (2.0*np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
                            val += w[k] * np.exp(-line*mu_scale[k])
                        row[si] = val
                    I[ti] = row
                    pb.progress((ti+1)/n_proj, text=f"Erzeuge Sinogramm... {ti+1}/{n_proj}")

                st.image(to_gray_img(I), caption="Sinogramm (−ln(I/I0) vor Filter, skaliert)", use_container_width=True)

                if I.shape[0] >= 3:
                    I0 = max(1e-3, float(np.max(I)))
                    sino_log = -np.log(np.clip(I/I0, 1e-6, 1.0))
                    # Reconstruction (parallel FBP) with hard cap on size
                    R = Reconstructor(method="fbp")
                    rec = R.fbp(sino_log, thetas, (N, N))
                    st.image(to_gray_img(rec, invert=False), caption="Rekonstruktion μ (normiert)", use_container_width=True)

                    # HU (effective energy)
                    mu_w = float(np.mean(MaterialDB().mu_water(spec.effective_energy())))
                    hu = to_hu(rec, mu_w)
                    # normalize HU to display
                    hu_disp = (hu - np.percentile(hu, 5)) / (np.percentile(hu, 95)-np.percentile(hu, 5)+1e-12)
                    hu_disp = np.clip(hu_disp, 0, 1)
                    st.image(to_gray_img(hu_disp), caption="HU-Karte (skaliert für Anzeige)", use_container_width=True)
                else:
                    st.info("Zu wenig Projektionen für Rekonstruktion (Zeitbudget).")

        except Exception as e:
            st.exception(e)
    else:
        st.info("Parameter einstellen und 'Simulieren (schnell)' klicken.")
