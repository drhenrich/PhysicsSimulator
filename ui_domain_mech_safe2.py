
# ============================================================
# ui_domain_mech_safe.py — Mechanik & Astromechanik (100% st.image, no deps)
# Exposes: render_mech_astro_tab()
# ============================================================
from __future__ import annotations

def render_mech_astro_tab():
    import numpy as np
    import streamlit as st

    st.subheader("Mechanik & Astromechanik — Safe Mode")

    mech_tab, astro_tab = st.tabs(["Mechanik (2D)", "Astromechanik (2D)"])

    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min(); A = A/(A.max()+1e-12)
        return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")

    # --------- Mechanik: einfache geradlinige Bewegung + elastische Wände ---------
    with mech_tab:
        with st.form("mech_safe"):
            N = st.selectbox("Raster N (Anzeige)", [256, 384, 512], index=0, key="mechN")
            n = st.number_input("Anzahl Objekte", 1, 10, 3, 1, key="mech_n")
            t_end = st.number_input("t_end [s]", min_value=0.1, max_value=20.0, value=5.0, step=0.1, key="mech_t_end")
            dt = st.number_input("dt [s]", min_value=0.001, max_value=0.5, value=0.02, step=0.001, key="mech_dt")
            run = st.form_submit_button("Simulieren (safe)")
        if run:
            N=int(N); n=int(n)
            steps = int(t_end/dt)
            rng = np.random.default_rng(1)
            # Partikel in [-1,1]^2
            x = rng.uniform(-0.8,0.8,size=(n,)); y = rng.uniform(-0.8,0.8,size=(n,))
            vx = rng.uniform(-0.5,0.5,size=(n,)); vy = rng.uniform(-0.5,0.5,size=(n,))
            traj = np.zeros((steps, n, 2), dtype=float)
            for t in range(steps):
                x += vx*dt; y += vy*dt
                hitx = (x<-1)|(x>1); hity=(y<-1)|(y>1)
                vx[hitx]*=-1; vy[hity]*=-1
                x = np.clip(x,-1,1); y=np.clip(y,-1,1)
                traj[t,:,0]=x; traj[t,:,1]=y
            # Render letzte Position + Spuren
            img = np.zeros((N,N), dtype=float)
            for i in range(n):
                px = (traj[:,i,0]*0.5+0.5)*(N-1)
                py = (traj[:,i,1]*0.5+0.5)*(N-1)
                px = np.clip(px.astype(int),0,N-1)
                py = np.clip(py.astype(int),0,N-1)
                img[py, px] = 1.0
            st.image(to_img(img), caption="Mechanik (Spuren)", use_container_width=True)
        else:
            st.info("Parameter einstellen und 'Simulieren (safe)' klicken.")

    # --------- Astromechanik: Kepler-Bahn (Ellipse) ---------
    with astro_tab:
        with st.form("astro_safe"):
            N = st.selectbox("Raster N (Anzeige)", [256, 384, 512], index=0, key="astroN")
            e = st.slider("Exzentrizität e", 0.0, 0.9, 0.5, 0.01, key="astro_e")
            a = st.number_input("Große Halbachse a [a.u.]", min_value=0.5, max_value=3.0, value=1.0, step=0.1, key="astro_a")
            runA = st.form_submit_button("Bahn zeichnen (safe)")
        if runA:
            import numpy as np
            N=int(N)
            th = np.linspace(0, 2*np.pi, 1500, endpoint=False)
            r = a*(1-e**2)/(1+e*np.cos(th))
            x = r*np.cos(th); y = r*np.sin(th)
            # render
            img = np.zeros((N,N), dtype=float)
            px = (x/(1.2*a)*0.5+0.5)*(N-1); py=(y/(1.2*a)*0.5+0.5)*(N-1)
            px = np.clip(px.astype(int),0,N-1); py=np.clip(py.astype(int),0,N-1)
            img[py, px] = 1.0
            # Zentralmasse
            cx = int(0.5*(N-1)); cy=int(0.5*(N-1)); img[cy-2:cy+3, cx-2:cx+3]=1.0
            st.image(to_img(img), caption="Kepler-Ellipse (safe)", use_container_width=True)
        else:
            st.info("Parameter einstellen und 'Bahn zeichnen (safe)' klicken.")
