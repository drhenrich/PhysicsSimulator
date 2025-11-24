
# ============================================================
# ui_domain_mri_bloch_safe.py — MRI & Bloch (100% st.image, no deps)
# Exposes: render_mri_bloch_tab()
# ============================================================
from __future__ import annotations

try:
    from i18n_utils import get_text
except Exception:
    def get_text(key, language="de"):
        return None

def render_mri_bloch_tab():
    import numpy as np
    import streamlit as st
    lang = st.session_state.get("language", "de")
    tr = lambda de, en: de if lang == "de" else en

    st.subheader(tr("MRI & Bloch", "MRI & Bloch"))

    mri_tab, bloch_tab = st.tabs([tr("MRI", "MRI"), tr("Bloch", "Bloch")])

    def to_img(A: np.ndarray):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A - A.min(); A = A/(A.max()+1e-12)
        return (np.repeat(A[...,None],3,axis=2)*255).astype("uint8")

    with mri_tab:
        with st.form("mri_safe"):
            N = st.selectbox(tr("Matrix N", "Matrix N"), [96, 128, 160], index=1, key="mriN")
            noise = st.slider(tr("k-Space Rauschen", "k-space noise"), 0.0, 0.1, 0.0, 0.005, key="mriNoise")
            run = st.form_submit_button(tr("Simulieren", "Simulate"))
        if run:
            N=int(N)
            img = np.zeros((N,N), dtype=float)
            Y,X = np.indices((N,N)); x=(X-N/2)/(N/2); y=(Y-N/2)/(N/2); r=np.sqrt(x**2+y**2)
            img[r<0.9] = 1.0; img[(x+0.3)**2+(y+0.2)**2<0.1**2]=0.5; img[(x-0.25)**2+(y-0.25)**2<0.15**2]=1.5
            k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
            if noise>0:
                k = k + (np.random.normal(0,noise,k.shape) + 1j*np.random.normal(0,noise,k.shape))/np.sqrt(2)
            rec = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k))); rec = np.abs(rec); rec/=rec.max()+1e-12
            st.image(to_img(rec), caption=tr("MRI Rekonstruktion", "MRI reconstruction"), use_container_width=True)
            kmag = np.log10(1+np.abs(k)); kmag/=kmag.max()+1e-12
            st.image(to_img(kmag), caption="k-Space |K| (log)", use_container_width=True)
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
            t = np.linspace(0, TR/1000.0, 400)
            M0 = 1.0
            Mz = M0*(1-np.exp(-t/(T1/1000.0)))
            Mxy = M0*(1-np.exp(-TR/1000.0/(T1/1000.0))) * np.exp(-TE/1000.0/(T2/1000.0))
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(5,2.5)); ax=fig.add_subplot(111)
            ax.plot(t*1000.0, Mz); ax.set_xlabel("t [ms]"); ax.set_ylabel(tr("Mz (norm.)","Mz (norm.)")); ax.set_title(tr("Mz(t)", "Mz(t)"))
            st.pyplot(fig)
            st.metric("Mxy @ TE", f"{float(Mxy):.3f}")
        else:
            st.info(tr("Parameter wählen und 'Bloch berechnen' klicken.", "Choose parameters and click 'Compute Bloch'."))
