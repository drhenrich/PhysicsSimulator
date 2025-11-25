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
    def to_gray_img(A: np.ndarray, invert: bool = False):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0); A = A - A.min(); A = A / (A.max() + 1e-12)
        if invert: A = 1.0 - A
        img = (A * 255.0).astype("uint8"); return np.repeat(img[..., None], 3, axis=2)
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
            R = Reconstructor(method="fbp"); rec = R.fbp(sino_log, thetas, (N, N)); st.image(to_gray_img(rec), caption=tr("Rekonstruktion μ (normiert)", "Reconstruction μ (normalized)"), use_container_width=True)
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en
    
    st.subheader(tr("Elektrodynamik & Potential", "Electrodynamics & Potential"))
    
    es_tab, pot_tab, preset_tab = st.tabs([
        tr("Elektrostatik (Punktladungen)", "Electrostatics (point charges)"),
        tr("Potentialfeld (Poisson)", "Potential field (Poisson)"),
        tr("Presets & Simulation", "Presets & Simulation")
    ])
    
    # Physikalische Konstante
    k_e = 8.99e9  # Coulomb-Konstante [N·m²/C²]
    
    def compute_field_advanced(x_range, y_range, N_grid, charges):
        """
        Berechnet E-Feld und Potential für gegebene Punktladungen.
        
        Args:
            x_range: (x_min, x_max) in Metern
            y_range: (y_min, y_max) in Metern
            N_grid: Anzahl Gitterpunkte pro Dimension
            charges: Liste von (x, y, q) Tupeln - Position in m, Ladung in C
        
        Returns:
            X, Y: Koordinatenmatrizen
            phi: Potential in V
            Ex, Ey: E-Feld Komponenten in V/m (N/C)
            Emag: Betrag des E-Feldes
        """
        x = np.linspace(x_range[0], x_range[1], N_grid)
        y = np.linspace(y_range[0], y_range[1], N_grid)
        X, Y = np.meshgrid(x, y)
        
        phi = np.zeros_like(X)
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(X)
        
        eps = 0.05  # Regularisierung nahe Ladungen (in m)
        
        for cx, cy, q in charges:
            rx = X - cx
            ry = Y - cy
            r2 = rx**2 + ry**2 + eps**2
            r = np.sqrt(r2)
            
            # Potential: φ = k * q / r
            phi += k_e * q / r
            
            # E-Feld: E = k * q * r_vec / r³
            Ex += k_e * q * rx / (r2 * r)
            Ey += k_e * q * ry / (r2 * r)
        
        Emag = np.sqrt(Ex**2 + Ey**2)
        
        return X, Y, phi, Ex, Ey, Emag
    
    def compute_field_lines(X, Y, Ex, Ey, charges, n_lines=16):
        """
        Berechnet Feldlinien-Startpunkte um positive Ladungen.
        """
        lines_x = []
        lines_y = []
        
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        
        for cx, cy, q in charges:
            if q > 0:  # Feldlinien starten bei positiven Ladungen
                n = max(8, int(n_lines * abs(q) / max(abs(c[2]) for c in charges)))
                for i in range(n):
                    angle = 2 * np.pi * i / n
                    start_r = 0.15  # Startabstand in m
                    
                    # Startpunkt
                    px = cx + start_r * np.cos(angle)
                    py = cy + start_r * np.sin(angle)
                    
                    line_x = [px]
                    line_y = [py]
                    
                    # Feldlinie verfolgen
                    step = 0.03  # Schrittweite in m
                    for _ in range(200):
                        # Interpoliere E-Feld an aktueller Position
                        ix = int((px - X[0, 0]) / dx)
                        iy = int((py - Y[0, 0]) / dy)
                        
                        if ix < 0 or ix >= X.shape[1] - 1 or iy < 0 or iy >= Y.shape[0] - 1:
                            break
                        
                        ex = Ex[iy, ix]
                        ey = Ey[iy, ix]
                        e_mag = np.sqrt(ex**2 + ey**2) + 1e-10
                        
                        # Normierter Schritt
                        px += step * ex / e_mag
                        py += step * ey / e_mag
                        
                        line_x.append(px)
                        line_y.append(py)
                        
                        # Stopp bei negativer Ladung oder Rand
                        for ncx, ncy, nq in charges:
                            if nq < 0 and (px - ncx)**2 + (py - ncy)**2 < 0.1**2:
                                break
                        else:
                            continue
                        break
                    
                    lines_x.append(line_x)
                    lines_y.append(line_y)
        
        return lines_x, lines_y
    
    def plot_electrostatics(X, Y, phi, Ex, Ey, Emag, charges, show_field_lines=True):
        """
        Erstellt interaktive Plotly-Darstellung wie im Referenzbild.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                tr("Elektrische Feldlinien", "Electric Field Lines"),
                tr("Elektrisches Potential", "Electric Potential")
            ),
            horizontal_spacing=0.12
        )
        
        # Linkes Bild: Feldstärke mit Feldlinien
        # Clipping für bessere Darstellung
        Emag_clipped = np.clip(Emag, 0, np.percentile(Emag, 98))
        
        fig.add_trace(
            go.Heatmap(
                x=X[0, :],
                y=Y[:, 0],
                z=Emag_clipped,
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(text=tr("Feldstärke [N/C]", "Field strength [N/C]"), side="right"),
                    x=0.45,
                    len=0.9
                ),
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>|E|=%{z:.0f} N/C<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Feldlinien hinzufügen
        if show_field_lines:
            lines_x, lines_y = compute_field_lines(X, Y, Ex, Ey, charges)
            for lx, ly in zip(lines_x, lines_y):
                fig.add_trace(
                    go.Scatter(
                        x=lx, y=ly,
                        mode='lines',
                        line=dict(color='white', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
        
        # Ladungen als Symbole
        for cx, cy, q in charges:
            color = 'red' if q > 0 else 'blue'
            symbol = '+' if q > 0 else '−'
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy],
                    mode='markers+text',
                    marker=dict(size=20, color=color, line=dict(color='white', width=2)),
                    text=[symbol],
                    textfont=dict(size=16, color='white'),
                    textposition='middle center',
                    showlegend=False,
                    hovertemplate=f"q={q:.2e} C<br>x={cx:.2f} m<br>y={cy:.2f} m<extra></extra>"
                ),
                row=1, col=1
            )
            # Annotation mit Ladungswert
            fig.add_annotation(
                x=cx, y=cy + 0.4,
                text=f"q={q:.1e}<br>C",
                showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                row=1, col=1
            )
        
        # Rechtes Bild: Potential mit Konturlinien
        # Clipping für Potential
        phi_clipped = np.clip(phi, np.percentile(phi, 2), np.percentile(phi, 98))
        
        fig.add_trace(
            go.Heatmap(
                x=X[0, :],
                y=Y[:, 0],
                z=phi_clipped,
                colorscale='RdBu_r',  # Rot-Blau divergent
                zmid=0,  # Null in der Mitte
                colorbar=dict(
                    title=dict(text=tr("Potential [V]", "Potential [V]"), side="right"),
                    x=1.02,
                    len=0.9
                ),
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>φ=%{z:.0f} V<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Konturlinien für Potential
        n_contours = 15
        fig.add_trace(
            go.Contour(
                x=X[0, :],
                y=Y[:, 0],
                z=phi_clipped,
                contours=dict(
                    coloring='none',
                    showlabels=False,
                    start=np.percentile(phi_clipped, 5),
                    end=np.percentile(phi_clipped, 95),
                    size=(np.percentile(phi_clipped, 95) - np.percentile(phi_clipped, 5)) / n_contours
                ),
                line=dict(color='white', width=0.5),
                showscale=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        # Ladungen auch im rechten Bild
        for cx, cy, q in charges:
            color = 'red' if q > 0 else 'blue'
            symbol = '+' if q > 0 else '−'
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy],
                    mode='markers+text',
                    marker=dict(size=20, color=color, line=dict(color='white', width=2)),
                    text=[symbol],
                    textfont=dict(size=16, color='white'),
                    textposition='middle center',
                    showlegend=False,
                    hovertemplate=f"q={q:.2e} C<br>x={cx:.2f} m<br>y={cy:.2f} m<extra></extra>"
                ),
                row=1, col=2
            )
        
        # Layout
        fig.update_layout(
            height=500,
            margin=dict(l=60, r=100, t=60, b=60),
            showlegend=False
        )
        
        fig.update_xaxes(title_text="x [m]", scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="y [m]", row=1, col=1)
        fig.update_xaxes(title_text="x [m]", scaleanchor="y", scaleratio=1, row=1, col=2)
        fig.update_yaxes(title_text="y [m]", row=1, col=2)
        
        return fig
    
    # --- Elektrostatik Tab ---
    with es_tab:
        st.markdown(tr(
            "**Punktladungen im Vakuum** — Berechnung von E-Feld und Potential",
            "**Point charges in vacuum** — Calculation of E-field and potential"
        ))
        
        col_params, col_plot = st.columns([1, 3])
        
        with col_params:
            st.markdown(f"**{tr('Parameter', 'Parameters')}**")
            
            N_grid = st.selectbox(
                tr("Auflösung", "Resolution"),
                [64, 100, 150, 200],
                index=1,
                key="es_grid"
            )
            
            x_span = st.slider(
                tr("x-Bereich [m]", "x-range [m]"),
                min_value=1.0, max_value=10.0, value=4.0, step=0.5,
                key="es_xspan"
            )
            
            n_charges = st.selectbox(
                tr("Anzahl Ladungen", "Number of charges"),
                [1, 2, 3, 4],
                index=1,
                key="es_ncharges"
            )
            
            st.markdown("---")
            st.markdown(f"**{tr('Ladungen', 'Charges')}**")
            
            charges = []
            
            # Ladung 1
            q1 = st.number_input(
                "q₁ [µC]", min_value=-10.0, max_value=10.0, value=1.0, step=0.5,
                key="es_q1"
            ) * 1e-6
            x1 = st.number_input(
                "x₁ [m]", min_value=-x_span, max_value=x_span, value=-1.5, step=0.25,
                key="es_x1"
            )
            charges.append((x1, 0.0, q1))
            
            if n_charges >= 2:
                q2 = st.number_input(
                    "q₂ [µC]", min_value=-10.0, max_value=10.0, value=-4.0, step=0.5,
                    key="es_q2"
                ) * 1e-6
                x2 = st.number_input(
                    "x₂ [m]", min_value=-x_span, max_value=x_span, value=2.0, step=0.25,
                    key="es_x2"
                )
                charges.append((x2, 0.0, q2))
            
            if n_charges >= 3:
                q3 = st.number_input(
                    "q₃ [µC]", min_value=-10.0, max_value=10.0, value=2.0, step=0.5,
                    key="es_q3"
                ) * 1e-6
                x3 = st.number_input(
                    "x₃ [m]", min_value=-x_span, max_value=x_span, value=0.0, step=0.25,
                    key="es_x3"
                )
                y3 = st.number_input(
                    "y₃ [m]", min_value=-x_span/2, max_value=x_span/2, value=1.5, step=0.25,
                    key="es_y3"
                )
                charges.append((x3, y3, q3))
            
            if n_charges >= 4:
                q4 = st.number_input(
                    "q₄ [µC]", min_value=-10.0, max_value=10.0, value=-2.0, step=0.5,
                    key="es_q4"
                ) * 1e-6
                x4 = st.number_input(
                    "x₄ [m]", min_value=-x_span, max_value=x_span, value=0.0, step=0.25,
                    key="es_x4"
                )
                y4 = st.number_input(
                    "y₄ [m]", min_value=-x_span/2, max_value=x_span/2, value=-1.5, step=0.25,
                    key="es_y4"
                )
                charges.append((x4, y4, q4))
            
            st.markdown("---")
            show_lines = st.checkbox(
                tr("Feldlinien zeigen", "Show field lines"),
                value=True,
                key="es_lines"
            )
            
            run_es = st.button(
                tr("▶️ Berechnen", "▶️ Calculate"),
                key="es_run",
                use_container_width=True
            )
        
        with col_plot:
            if run_es:
                with st.spinner(tr("Berechne Feld...", "Computing field...")):
                    X, Y, phi, Ex, Ey, Emag = compute_field_advanced(
                        (-x_span, x_span),
                        (-x_span/2, x_span/2),
                        N_grid,
                        charges
                    )
                    
                    fig = plot_electrostatics(X, Y, phi, Ex, Ey, Emag, charges, show_lines)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Zusammenfassung
                    st.markdown(f"**{tr('Zusammenfassung', 'Summary')}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_q = sum(q for _, _, q in charges)
                        st.metric(tr("Gesamtladung", "Total charge"), f"{total_q*1e6:.2f} µC")
                    with col2:
                        st.metric(tr("Max. Feldstärke", "Max. field strength"), f"{np.max(Emag):.0f} N/C")
                    with col3:
                        st.metric(tr("Potentialbereich", "Potential range"), 
                                  f"{np.min(phi)/1000:.1f} ... {np.max(phi)/1000:.1f} kV")
            else:
                st.info(tr(
                    "Parameter links einstellen und 'Berechnen' klicken.",
                    "Set parameters on the left and click 'Calculate'."
                ))
    
    # --- Poisson Tab ---
    with pot_tab:
        st.markdown(tr(
            "**Poisson-Gleichung** — Numerische Lösung mit Relaxation",
            "**Poisson equation** — Numerical solution with relaxation"
        ))
        
        with st.form("em_pot"):
            col1, col2 = st.columns(2)
            with col1:
                N_pot = st.selectbox(
                    tr("Gitter N", "Grid N"),
                    [64, 96, 128, 160],
                    index=2,
                    key="potN"
                )
            with col2:
                iters = st.selectbox(
                    tr("Iterationen", "Iterations"),
                    [100, 250, 500, 1000, 2000],
                    index=2,
                    key="potIter"
                )
            
            runP = st.form_submit_button(
                tr("Poisson lösen", "Solve Poisson"),
                use_container_width=True
            )
        
        if runP:
            N = int(N_pot)
            iters = int(iters)
            
            # Ladungsverteilung
            rho = np.zeros((N, N), dtype=float)
            rho[N//3, N//3] = +1.0
            rho[2*N//3, 2*N//3] = -1.0
            
            # Jacobi-Relaxation
            phi = np.zeros_like(rho)
            progress = st.progress(0)
            
            for i in range(iters):
                phi = 0.25 * (
                    np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
                    np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - rho
                )
                phi[0, :] = phi[-1, :] = phi[:, 0] = phi[:, -1] = 0.0
                
                if i % (iters // 20) == 0:
                    progress.progress((i + 1) / iters)
            
            progress.progress(1.0)
            
            # Visualisierung
            x = np.linspace(-1, 1, N)
            y = np.linspace(-1, 1, N)
            
            fig = go.Figure()
            
            fig.add_trace(go.Heatmap(
                x=x, y=y, z=phi,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title=tr("Potential φ", "Potential φ"))
            ))
            
            # Konturlinien
            fig.add_trace(go.Contour(
                x=x, y=y, z=phi,
                contours=dict(coloring='none', showlabels=True),
                line=dict(color='black', width=1),
                showscale=False
            ))
            
            # Ladungspositionen
            fig.add_trace(go.Scatter(
                x=[-1 + 2*N//3/N], y=[-1 + 2*N//3/N],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                text=['+'],
                textfont=dict(size=14, color='white'),
                textposition='middle center',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[-1 + 2*(2*N//3)/N], y=[-1 + 2*(2*N//3)/N],
                mode='markers+text',
                marker=dict(size=15, color='blue'),
                text=['−'],
                textfont=dict(size=14, color='white'),
                textposition='middle center',
                showlegend=False
            ))
            
            fig.update_layout(
                title=tr(f"Poisson-Lösung nach {iters} Iterationen", 
                        f"Poisson solution after {iters} iterations"),
                xaxis_title="x",
                yaxis_title="y",
                height=500,
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(tr(
                "Parameter wählen und 'Poisson lösen' klicken.",
                "Choose parameters and click 'Solve Poisson'."
            ))
    
    # --- Presets Tab ---
    with preset_tab:
        st.markdown(tr(
            "**Schnellstart mit vordefinierten Konfigurationen**",
            "**Quick start with predefined configurations**"
        ))
        
        presets = {
            tr("Einzelladung (positiv)", "Single charge (positive)"): {
                "charges": [(0.0, 0.0, 1e-6)],
                "x_span": 3.0
            },
            tr("Dipol", "Dipole"): {
                "charges": [(-1.0, 0.0, 1e-6), (1.0, 0.0, -1e-6)],
                "x_span": 4.0
            },
            tr("Dipol (ungleich)", "Dipole (unequal)"): {
                "charges": [(-1.5, 0.0, 1e-6), (2.0, 0.0, -4e-6)],
                "x_span": 4.0
            },
            tr("Quadrupol", "Quadrupole"): {
                "charges": [
                    (-1.0, -1.0, 1e-6), (1.0, -1.0, -1e-6),
                    (-1.0, 1.0, -1e-6), (1.0, 1.0, 1e-6)
                ],
                "x_span": 3.0
            },
            tr("Drei Ladungen (linear)", "Three charges (linear)"): {
                "charges": [(-2.0, 0.0, 2e-6), (0.0, 0.0, -1e-6), (2.0, 0.0, 2e-6)],
                "x_span": 4.0
            },
        }
        
        preset_name = st.selectbox(
            tr("Preset wählen", "Select preset"),
            list(presets.keys()),
            key="em_preset_select"
        )
        
        if st.button(tr("▶️ Preset simulieren", "▶️ Simulate preset"), 
                     key="em_preset_run", use_container_width=True):
            cfg = presets[preset_name]
            charges = cfg["charges"]
            x_span = cfg["x_span"]
            
            with st.spinner(tr("Berechne...", "Computing...")):
                X, Y, phi, Ex, Ey, Emag = compute_field_advanced(
                    (-x_span, x_span),
                    (-x_span * 0.6, x_span * 0.6),
                    120,
                    charges
                )
                
                fig = plot_electrostatics(X, Y, phi, Ex, Ey, Emag, charges, show_field_lines=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Ladungstabelle
                st.markdown(f"**{tr('Ladungskonfiguration', 'Charge configuration')}**")
                charge_data = []
                for i, (cx, cy, q) in enumerate(charges):
                    charge_data.append({
                        tr("Ladung", "Charge"): f"q{i+1}",
                        "x [m]": f"{cx:.2f}",
                        "y [m]": f"{cy:.2f}",
                        "q [µC]": f"{q*1e6:.2f}",
                        tr("Typ", "Type"): "+" if q > 0 else "−"
                    })
                st.table(charge_data)
