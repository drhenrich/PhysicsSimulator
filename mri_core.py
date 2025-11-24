
# ============================================================
# mri_core.py — Minimal didactic MRI core
# ============================================================
from __future__ import annotations
import numpy as np

def shepp_phantom(N=128):
    # simple magnitude phantom (spin density)
    img = np.zeros((N,N), dtype=float)
    Y,X = np.indices((N,N))
    x = (X - N/2)/(N/2); y = (Y - N/2)/(N/2)
    r = np.sqrt(x**2 + y**2)
    img[r<0.9] = 1.0
    img[(x+0.3)**2 + (y+0.2)**2 < 0.1**2] = 0.5
    img[(x-0.25)**2 + (y-0.25)**2 < 0.15**2] = 1.5
    return img

def t1t2_maps(N=128):
    T1 = np.ones((N,N))*1000.0  # ms
    T2 = np.ones((N,N))*80.0    # ms
    Y,X = np.indices((N,N))
    x = (X - N/2)/(N/2); y = (Y - N/2)/(N/2)
    T1[(x+0.3)**2 + (y+0.2)**2 < 0.1**2] = 600.0
    T2[(x-0.25)**2 + (y-0.25)**2 < 0.15**2] = 40.0
    return T1, T2

def steady_state(SE: bool, TR_ms: float, TE_ms: float, T1, T2, rho):
    E1 = np.exp(-TR_ms/np.maximum(T1,1e-9))
    E2 = np.exp(-TE_ms/np.maximum(T2,1e-9))
    if SE:
        S = rho * (1.0 - E1) * E2
    else:
        # GRE approx (ignoring flip, use T2*~T2)
        S = rho * (1.0 - E1) * E2
    return S

def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

def acquire_cartesian(S, noise=0.0):
    k = fft2c(S)
    if noise>0:
        k = k + (np.random.normal(0,noise,k.shape) + 1j*np.random.normal(0,noise,k.shape))/np.sqrt(2)
    return k

def reconstruct(k):
    im = ifft2c(k)
    return np.abs(im)
