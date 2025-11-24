from __future__ import annotations
import numpy as np
import streamlit as st
from sim_core_bundle import PRESENTS_BLOCH  # for consistency if needed

# --- CT reduced ---
def render_ct_safe_tab():
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    st.subheader(tr("Röntgen/CT", "Xray/CT"))
    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min(); A = A/(A.max()+1e-12)
        return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")
    with st.form("ct_safe_form"):
        N = st.selectbox(tr("Phantom N", "Phantom N"), [96, 128, 160], index=1, key="ctN_safe")
        n_proj = st.selectbox(tr("Projektionen", "Projections"), [60, 90, 120], index=0, key="ctP_safe")
        reconstruct = st.checkbox(tr("Rekonstruktion (FBP)", "Reconstruction (FBP)"), value=True, key="ctFBP_safe")
        run = st.form_submit_button(tr("CT simulieren", "Simulate CT"))
    if run:
        N=int(N); n_proj=int(n_proj); n_det = 2*N
        Y,X = np.indices((N,N)); x=(X-N/2)/(N/2); y=(Y-N/2)/(N/2); r=np.sqrt(x**2+y**2)
        mu = np.zeros((N,N), dtype=float); mu[r<0.85]=0.2; mu[r<0.2]=0.001; ring=(r>0.55)&(r<0.75); mu[ring]=0.45
        st.image(to_img(mu), caption=tr("Phantom (normiert)", "Phantom (normalized)"), use_container_width=True)
        dets = np.linspace(-1,1,n_det,endpoint=False); thetas = np.linspace(0,np.pi,n_proj,endpoint=False)
        sino = np.zeros((n_proj, n_det), dtype=float)
        for ti, th in enumerate(thetas):
            d = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
            for si, s in enumerate(dets):
                p0 = -np.sqrt(2)*d + s*n; p1 = np.sqrt(2)*d + s*n; t = np.linspace(0,1,48)
                xx = p0[0] + (p1[0]-p0[0])*t; yy = p0[1] + (p1[1]-p0[1])*t
                gx = (xx*0.5+0.5)*(N-1); gy=(yy*0.5+0.5)*(N-1)
                x0 = np.clip(np.floor(gx).astype(int),0,N-1); y0=np.clip(np.floor(gy).astype(int),0,N-1)
                line = mu[y0, x0].mean()*2*np.sqrt(2)
                sino[ti, si] = np.exp(-line)
        st.image(to_img(sino), caption=tr("Sinogramm (I/I0)", "Sinogram (I/I0)"), use_container_width=True)
        if reconstruct:
            sino_log = -np.log(np.clip(sino/np.max(sino), 1e-6, 1.0)); n_proj, n_det = sino_log.shape; f = np.fft.rfftfreq(n_det); H = np.abs(f)
            rec = np.zeros((N,N), dtype=float); yy, xx = np.indices((N,N)); x = (xx - N/2)/(N/2); y = (yy - N/2)/(N/2)
            for th in thetas:
                P = np.fft.irfft(np.fft.rfft(sino_log[int(th/np.pi*len(thetas))]) * H, n=n_det)
                s = x*np.cos(th) + y*np.sin(th); u = (s*0.5 + 0.5) * (n_det - 1); u0 = np.clip(np.floor(u).astype(int), 0, n_det-1); u1 = np.clip(u0+1, 0, n_det-1); w = u - u0; val = (1-w)*P[u0] + w*P[u1]; rec += val
            rec = rec * (np.pi / n_proj); rec = rec - rec.min(); rec = rec/(rec.max()+1e-12)
            st.image(to_img(rec), caption=tr("Rekonstruktion (FBP, normiert)", "Reconstruction (FBP, normalized)"), use_container_width=True)
    else:
        st.info(tr("Parameter wählen und 'CT simulieren' klicken.", "Choose parameters and click 'Simulate CT'."))

# --- Xray/CT fast ---
def render_xray_ct_tab():
    import time
    try:
        from xray_ct import Spectra, MaterialDB, Projector, Reconstructor, shepp_logan, cylinders_phantom, to_hu
    except Exception as e:
        st.error(f"Xray/CT core not available: {e}"); return
    lang = st.session_state.get("language", "de"); tr = lambda de, en: de if lang == "de" else en
    st.subheader(tr("Röntgen / CT-Physik (Schnellmodus)", "Xray/CT physics (fast)"))
    def to_gray_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0); A = A - A.min(); A = A / (A.max() + 1e-12); img = (A * 255.0).astype("uint8"); return np.repeat(img[..., None], 3, axis=2)
    with st.form("ct_form_fast"):
        col1, col2 = st.columns([2,1], gap="large")
        with col2:
            phantom = st.selectbox("Phantom", ["Shepp-Logan", "Zylinder (Wasser/Knochen/Luft)"], index=0, key="ct_phantom")
            N = st.selectbox(tr("Bildgröße N", "Image size N"), [128, 192, 256], index=0, key="ct_N")
            geom = st.selectbox(tr("Geometrie", "Geometry"), ["Parallel"], index=0, key="ct_geom")
            n_det = st.number_input(tr("Detektorelemente", "Detector elements"), min_value=90, max_value=720, value=180, step=10, key="ct_ndet")
            n_proj = st.number_input(tr("Projektionen", "Projections"), min_value=30, max_value=720, value=120, step=30, key="ct_nproj")
            kVp = st.number_input("kVp", min_value=40.0, max_value=140.0, value=80.0, step=2.0, key="ct_kvp")
            filt = st.number_input(tr("Filtration [mm Al]", "Filtration [mm Al]"), min_value=0.0, max_value=10.0, value=2.5, step=0.5, key="ct_filt")
            poly = st.checkbox(tr("Polychromatisch", "Polychromatic"), value=False, key="ct_poly")
            noise = st.slider(tr("Detektor-Rauschen (σ)", "Detector noise (σ)"), min_value=0.0, max_value=0.05, value=0.0, step=0.005, key="ct_noise")
            budget = st.slider(tr("Zeitbudget [s]", "Time budget [s]"), min_value=1, max_value=10, value=4, step=1, key="ct_budget")
            run = st.form_submit_button(tr("Simulieren (schnell)", "Simulate (fast)"), use_container_width=True)
    if run:
        start_t = time.time(); N = int(N); n_det = min(int(n_det), 256); n_proj = min(int(n_proj), 180)
        mu_true = shepp_logan(N) if phantom.startswith("Shepp") else cylinders_phantom(N)
        st.image(to_gray_img(mu_true), caption=tr("Phantom (μ, normiert)", "Phantom (μ, normalized)"), use_container_width=True)
        proj = Projector(geometry="parallel"); db = MaterialDB(); spec = Spectra(kVp=float(kVp), filtration_mm_Al=float(filt))
        I = np.zeros((n_proj, n_det), dtype=np.float32); thetas = np.linspace(0.0, np.pi, n_proj, endpoint=False); dets = np.linspace(-1.0, 1.0, n_det, endpoint=False)
        pb = st.progress(0.0, text=tr("Erzeuge Sinogramm...", "Generating sinogram..."))
        for ti, th in enumerate(thetas):
            if time.time() - start_t > budget:
                st.warning(tr(f"Zeitbudget erreicht nach {ti}/{n_proj} Projektionen.", f"Time budget reached after {ti}/{n_proj} projections."))
                thetas = thetas[:ti]; I = I[:ti]; break
            d = np.array([np.cos(th), np.sin(th)], dtype=float); n = np.array([-np.sin(th), np.cos(th)], dtype=float)
            if poly:
                Es, w = spec.sample_spectrum(3); mu_scale = db.mu_water(Es); mu_scale = mu_scale/(mu_scale.max()+1e-12)
            else:
                Es, w = [spec.effective_energy()], [1.0]; mu_scale = np.array([1.0])
            row = np.zeros(n_det, dtype=np.float32); steps = max(64, N//2)
            for si, s in enumerate(dets):
                p0 = -np.sqrt(2.0)*d + s*n; p1 =  np.sqrt(2.0)*d + s*n; val = 0.0
                for k in range(len(Es)):
                    t = np.linspace(0.0, 1.0, steps)
                    x = p0[0] + (p1[0] - p0[0])*t; y = p0[1] + (p1[1] - p0[1])*t
                    gx = (x*0.5 + 0.5)*(N-1); gy = (y*0.5 + 0.5)*(N-1)
                    x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int); x1 = np.clip(x0+1, 0, N-1); y1 = np.clip(y0+1, 0, N-1)
                    x0 = np.clip(x0, 0, N-1); y0 = np.clip(y0, 0, N-1)
                    wx = gx - x0; wy = gy - y0
                    Ia = mu_true[y0, x0]; Ib = mu_true[y0, x1]; Ic = mu_true[y1, x0]; Id = mu_true[y1, x1]
                    mus = (1-wx)*(1-wy)*Ia + wx*(1-wy)*Ib + (1-wx)*wy*Ic + wx*wy*Id
                    line = mus.mean() * (2.0*np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
                    val += w[k] * np.exp(-line*mu_scale[k])
                row[si] = val
            I[ti] = row; pb.progress((ti+1)/n_proj, text=tr(f"Erzeuge Sinogramm... {ti+1}/{n_proj}", f"Generating sinogram... {ti+1}/{n_proj}"))
        st.image(to_gray_img(I), caption=tr("Sinogramm (−ln(I/I0) vor Filter, skaliert)", "Sinogram (−ln(I/I0) pre-filter, scaled)"), use_container_width=True)
        if I.shape[0] >= 3:
            I0 = max(1e-3, float(np.max(I))); sino_log = -np.log(np.clip(I/I0, 1e-6, 1.0))
            R = Reconstructor(method="fbp"); rec = R.fbp(sino_log, thetas, (N, N)); st.image(to_gray_img(rec, invert=False), caption=tr("Rekonstruktion μ (normiert)", "Reconstruction μ (normalized)"), use_container_width=True)
            mu_w = float(np.mean(MaterialDB().mu_water(spec.effective_energy()))); hu = to_hu(rec, mu_w)
            hu_disp = (hu - np.percentile(hu, 5)) / (np.percentile(hu, 95)-np.percentile(hu, 5)+1e-12); hu_disp = np.clip(hu_disp, 0, 1)
            st.image(to_gray_img(hu_disp), caption=tr("HU-Karte (skaliert für Anzeige)", "HU map (scaled for display)"), use_container_width=True)
        else:
            st.info(tr("Zu wenig Projektionen für Rekonstruktion (Zeitbudget).", "Not enough projections for reconstruction (time budget)."))
    else:
        st.info(tr("Parameter einstellen und 'Simulieren (schnell)' klicken.", "Set parameters and click 'Simulate (fast)'."))

# --- MRI / Bloch ---
def render_mri_bloch_tab():
    import numpy as np
    import matplotlib.pyplot as plt
    lang = st.session_state.get("language", "de"); tr = lambda de, en: de if lang == "de" else en
    st.subheader("MRI & Bloch")
    mri_tab, bloch_tab = st.tabs(["MRI", "Bloch"])
    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0); A = A - A.min(); A = A/(A.max()+1e-12); return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")
    with mri_tab:
        with st.form("mri_safe"):
            N = st.selectbox("Matrix N", [96, 128, 160], index=1, key="mriN"); noise = st.slider(tr("k-Space Rauschen", "k-space noise"), 0.0, 0.1, 0.0, 0.005, key="mriNoise")
            run = st.form_submit_button(tr("Simulieren", "Simulate"))
        if run:
            N=int(N); img = np.zeros((N,N), dtype=float); Y,X = np.indices((N,N)); x=(X-N/2)/(N/2); y=(Y-N/2)/(N/2); r=np.sqrt(x**2+y**2)
            img[r<0.9] = 1.0; img[(x+0.3)**2+(y+0.2)**2<0.1**2]=0.5; img[(x-0.25)**2+(y-0.25)**2<0.15**2]=1.5
            k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))); k = k + (np.random.normal(0,noise,k.shape) + 1j*np.random.normal(0,noise,k.shape))/np.sqrt(2) if noise>0 else k
            rec = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k))); rec = np.abs(rec); rec/=rec.max()+1e-12
            st.image(to_img(rec), caption=tr("MRI Rekonstruktion", "MRI reconstruction"), use_container_width=True)
            kmag = np.log10(1+np.abs(k)); kmag/=kmag.max()+1e-12; st.image(to_img(kmag), caption="k-Space |K| (log)", use_container_width=True)
        else:
            st.info(tr("Parameter wählen und 'Simulieren' klicken.", "Choose parameters and click 'Simulate'."))
    with bloch_tab:
        with st.form("bloch_safe"):
            T1 = st.number_input("T1 [ms]", min_value=100.0, max_value=3000.0, value=1000.0, step=50.0, key="bT1")
            T2 = st.number_input("T2 [ms]", min_value=10.0, max_value=500.0, value=80.0, step=5.0, key="bT2")
            TR = st.number_input("TR [ms]", min_value=50.0, max_value=4000.0, value=800.0, step=25.0, key="bTR")
            TE = st.number_input("TE [ms]", min_value=5.0, max_value=200.0, value=20.0, step=5.0, key="bTE")
            runB = st.form_submit_button(tr("Bloch berechnen", "Compute Bloch"))
        if runB:
            t = np.linspace(0, TR/1000.0, 400); M0 = 1.0; Mz = M0*(1-np.exp(-t/(T1/1000.0))); Mxy = M0*(1-np.exp(-TR/1000.0/(T1/1000.0))) * np.exp(-TE/1000.0/(T2/1000.0))
            fig = plt.figure(figsize=(5,2.5)); ax=fig.add_subplot(111); ax.plot(t*1000.0, Mz); ax.set_xlabel("t [ms]"); ax.set_ylabel(tr("Mz (norm.)","Mz (norm.)")); ax.set_title(tr("Mz(t)", "Mz(t)")); st.pyplot(fig); st.metric("Mxy @ TE", f"{float(Mxy):.3f}")
        else:
            st.info(tr("Parameter wählen und 'Bloch berechnen' klicken.", "Choose parameters and click 'Compute Bloch'."))

# --- Electrodynamics ---
def render_em_tab():
    lang = st.session_state.get("language", "de"); tr = lambda de, en: de if lang == "de" else en
    st.subheader(tr("Elektrodynamik & Potential", "Electrodynamics & Potential"))
    es_tab, pot_tab, preset_tab = st.tabs([tr("Elektrostatik (Punktladungen)", "Electrostatics (point charges)"), tr("Potentialfeld (Poisson)", "Potential field (Poisson)"), tr("Presets & Simulation", "Presets & Simulation")])
    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0); A = A - A.min(); A = A/(A.max()+1e-12); return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")
    def compute_field(N_val, nQ_val, q1v, q2v, q3v):
        Y, X = np.indices((N_val, N_val), dtype=float)
        x = (X - N_val/2) / (N_val/2); y = (Y - N_val/2) / (N_val/2); eps = 1e-3
        charges = [(0.3,0.0,q1v)];
        if nQ_val >= 2: charges.append((-0.3,0.0,q2v))
        if nQ_val >= 3: charges.append((0.0,0.3,q3v))
        phi = np.zeros_like(x); Ex = np.zeros_like(phi); Ey = np.zeros_like(phi)
        for cx, cy, q in charges:
            rx = x - cx; ry = y - cy; r2 = rx*rx + ry*ry + eps**2; r = np.sqrt(r2)
            phi += q / r; Ex += q * rx / (r2*r); Ey += q * ry / (r2*r)
        Emag = np.sqrt(Ex**2 + Ey**2); return phi, Emag
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
    with pot_tab:
        with st.form("em_pot"):
            N = st.selectbox(tr("Raster N (Poisson)", "Grid N (Poisson)"), [96, 128, 160], index=1, key="potN")
            iters = st.selectbox(tr("Iterationsschritte", "Iterations"), [100, 250, 500, 1000], index=2, key="potIter")
            runP = st.form_submit_button(tr("Poisson lösen", "Solve Poisson"))
        if runP:
            N=int(N); iters=int(iters); rho = np.zeros((N,N), dtype=float); rho[N//3, N//3] = +1.0; rho[2*N//3, 2*N//3] = -1.0; phi = np.zeros_like(rho)
            for _ in range(iters):
                phi = 0.25*(np.roll(phi,1,0)+np.roll(phi,-1,0)+np.roll(phi,1,1)+np.roll(phi,-1,1) - rho); phi[0,:]=phi[-1,:]=phi[:,0]=phi[:,-1]=0.0
            st.image(to_img(phi), caption=tr(f"Poisson-Lösung nach {iters} Iterationen (normiert)", f"Poisson solution after {iters} iterations (normalized)"), use_container_width=True)
        else:
            st.info(tr("Parameter wählen und 'Poisson lösen' klicken.", "Choose parameters and click 'Solve Poisson'."))
    with preset_tab:
        st.markdown(tr("Wähle ein Elektrodynamik-Preset und starte direkt.", "Choose an electrodynamics preset and run."))
        em_presets = {"Einzelladung": {"N": 192, "nQ": 1, "q1": 1.0, "q2": 0.0, "q3": 0.0}, "Dipol": {"N": 192, "nQ": 2, "q1": 1.5, "q2": -1.5, "q3": 0.0}, "Quadrupol": {"N": 192, "nQ": 3, "q1": 1.5, "q2": -1.5, "q3": 1.5}}
        preset_name = st.selectbox(tr("Preset wählen", "Select preset"), [""] + list(em_presets.keys()), key="em_preset_tab")
        if st.button(tr("Simulation starten", "Start simulation"), key="em_preset_run"):
            if not preset_name:
                st.warning(tr("Bitte ein Preset auswählen.", "Please select a preset."))
            else:
                cfg = em_presets[preset_name]; phi, Emag = compute_field(int(cfg["N"]), int(cfg["nQ"]), cfg["q1"], cfg["q2"], cfg["q3"])
                st.success(tr(f"{preset_name} simuliert.", f"{preset_name} simulated."))
                st.image(to_img(phi), caption=tr("Potential (normiert)", "Potential (normalized)"), use_container_width=True)
                st.image(to_img(Emag), caption=tr("|E| (normiert)", "|E| (normalized)"), use_container_width=True)
