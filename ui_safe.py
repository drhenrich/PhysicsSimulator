
# ============================================================
# ui_safe.py — Minimal safe-mode app: guaranteed outputs with tiny workloads
# Run: streamlit run ui_safe.py
# ============================================================
from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Physics Simulator — Safe Mode", layout="wide")
st.title("Physics Simulator — Safe Mode")

def to_gray_img(A: np.ndarray):
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = A - A.min(); A = A / (A.max() + 1e-12)
    img = (A*255).astype("uint8")
    return np.repeat(img[..., None], 3, axis=2)

tab_wave, tab_ct, tab_mri = st.tabs(["Wellenoptik", "CT", "MRI"])

with tab_wave:
    N=128
    x = np.linspace(-1,1,N,endpoint=False)
    X,Y = np.meshgrid(x,x, indexing="xy")
    # simple double-slit interference surrogate
    I = (np.cos(20*X)**2) * (np.sinc(5*X/np.pi)**2)
    st.image(to_gray_img(I), caption="Wellenoptik (didaktisch, klein)", use_container_width=True)

with tab_ct:
    N=96
    Y,X = np.indices((N,N))
    r = np.sqrt((X-N/2)**2 + (Y-N/2)**2)/(N/2)
    mu = np.zeros((N,N), dtype=float); mu[r<0.8]=0.2; mu[r<0.2]=0.001; ring=(r>0.55)&(r<0.7); mu[ring]=0.45
    st.image(to_gray_img(mu), caption="Phantom", use_container_width=True)
    # quick sinogram approx
    n_proj, n_det = 60, 120
    thetas = np.linspace(0, np.pi, n_proj, endpoint=False)
    dets = np.linspace(-1,1,n_det, endpoint=False)
    sino = np.zeros((n_proj, n_det), dtype=float)
    for ti, th in enumerate(thetas):
        d = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
        for si, s in enumerate(dets):
            p0 = -np.sqrt(2)*d + s*n; p1 = np.sqrt(2)*d + s*n
            t = np.linspace(0,1,48)
            x = p0[0] + (p1[0]-p0[0])*t; y = p0[1] + (p1[1]-p0[1])*t
            gx = (x*0.5+0.5)*(N-1); gy=(y*0.5+0.5)*(N-1)
            x0 = np.clip(np.floor(gx).astype(int),0,N-1); y0=np.clip(np.floor(gy).astype(int),0,N-1)
            sino[ti, si] = np.exp(-mu[y0, x0].mean()*2*np.sqrt(2))
    st.image(to_gray_img(sino), caption="Sinogramm (klein)", use_container_width=True)

with tab_mri:
    N=96
    img = np.zeros((N,N), dtype=float)
    Y,X = np.indices((N,N)); x=(X-N/2)/(N/2); y=(Y-N/2)/(N/2); r=np.sqrt(x**2+y**2)
    img[r<0.9] = 1.0; img[(x+0.3)**2+(y+0.2)**2<0.1**2]=0.5; img[(x-0.25)**2+(y-0.25)**2<0.15**2]=1.5
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    rec = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k))); rec = np.abs(rec); rec/=rec.max()+1e-12
    st.image(to_gray_img(rec), caption="MRI Rekonstruktion (klein)", use_container_width=True)
