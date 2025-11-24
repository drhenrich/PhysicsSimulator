
# ============================================================
# ui_ct_safe.py — Röntgen/CT — Phantom, Sinogramm, FBP
# Exposes: render_ct_safe_tab()
# ============================================================
from __future__ import annotations

def render_ct_safe_tab():
    import numpy as np
    import streamlit as st
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
        # Phantom
        Y,X = np.indices((N,N))
        x=(X-N/2)/(N/2); y=(Y-N/2)/(N/2); r=np.sqrt(x**2+y**2)
        mu = np.zeros((N,N), dtype=float); mu[r<0.85]=0.2; mu[r<0.2]=0.001; ring=(r>0.55)&(r<0.75); mu[ring]=0.45
        st.image(to_img(mu), caption=tr("Phantom (normiert)", "Phantom (normalized)"), use_container_width=True)

        # Sinogramm (parallel)
        dets = np.linspace(-1,1,n_det,endpoint=False)
        thetas = np.linspace(0,np.pi,n_proj,endpoint=False)
        sino = np.zeros((n_proj, n_det), dtype=float)
        for ti, th in enumerate(thetas):
            d = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
            for si, s in enumerate(dets):
                p0 = -np.sqrt(2)*d + s*n; p1 = np.sqrt(2)*d + s*n
                t = np.linspace(0,1,48)
                x = p0[0] + (p1[0]-p0[0])*t; y = p0[1] + (p1[1]-p0[1])*t
                gx = (x*0.5+0.5)*(N-1); gy=(y*0.5+0.5)*(N-1)
                x0 = np.clip(np.floor(gx).astype(int),0,N-1); y0=np.clip(np.floor(gy).astype(int),0,N-1)
                line = mu[y0, x0].mean()*2*np.sqrt(2)
                sino[ti, si] = np.exp(-line)
        st.image(to_img(sino), caption=tr("Sinogramm (I/I0)", "Sinogram (I/I0)"), use_container_width=True)

        if reconstruct:
            # Log, Ramp-Filter (1D), einfache FBP
            sino_log = -np.log(np.clip(sino/np.max(sino), 1e-6, 1.0))
            n_proj, n_det = sino_log.shape
            # Precompute ramp in freq
            import numpy as np
            f = np.fft.rfftfreq(n_det)
            H = np.abs(f)
            # allocate image
            rec = np.zeros((N,N), dtype=float)
            yy, xx = np.indices((N,N))
            x = (xx - N/2)/(N/2); y = (yy - N/2)/(N/2)
            for ti, th in enumerate(thetas):
                # filter one projection
                P = np.fft.irfft(np.fft.rfft(sino_log[ti]) * H, n=n_det)
                s = x*np.cos(th) + y*np.sin(th)
                u = (s*0.5 + 0.5) * (n_det - 1)
                u0 = np.clip(np.floor(u).astype(int), 0, n_det-1)
                u1 = np.clip(u0+1, 0, n_det-1)
                w = u - u0
                val = (1-w)*P[u0] + w*P[u1]
                rec += val
            rec = rec * (np.pi / n_proj)
            rec = rec - rec.min(); rec = rec/(rec.max()+1e-12)
            st.image(to_img(rec), caption=tr("Rekonstruktion (FBP, normiert)", "Reconstruction (FBP, normalized)"), use_container_width=True)
    else:
        st.info(tr("Parameter wählen und 'CT simulieren' klicken.", "Choose parameters and click 'Simulate CT'."))
