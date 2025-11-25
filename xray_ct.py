# ============================================================
# xray_ct.py — Minimal didactic X-ray / CT core
# API used by ui_xray_ct.py:
#   Spectra, MaterialDB, Projector, Reconstructor,
#   shepp_logan, cylinders_phantom, forward_intensity, to_hu
# ============================================================
from __future__ import annotations
import numpy as np

# ---------------- Spectra ----------------
class Spectra:
    def __init__(self, kVp: float = 80.0, filtration_mm_Al: float = 2.5):
        self.kVp = float(kVp)
        self.filtration = float(filtration_mm_Al)

    def effective_energy(self) -> float:
        # crude rule of thumb: E_eff ≈ 0.35–0.5 kVp, increase with filtration
        base = 0.38 + 0.02*np.tanh((self.filtration-2.0)/5.0)
        return float(base * self.kVp)

    def sample_spectrum(self, nE: int = 5):
        # normalized discrete spectrum [keV, weight]; simple triangular weights
        Eeff = self.effective_energy()
        Emax = self.kVp
        Emin = max(15.0, 0.2*Eeff)
        Es = np.linspace(Emin, Emax, nE)
        w = np.maximum(0.0, 1.0 - np.abs(Es - Eeff)/(Emax-Emin+1e-9))
        w = w / (w.sum()+1e-12)
        return Es, w

# ---------------- Materials ----------------
class MaterialDB:
    # very rough attenuation [1/cm] toy model vs energy (keV)
    def mu_water(self, E_keV):
        E = np.asarray(E_keV, dtype=float)
        return 0.20 * (40.0/E)**3  # decreases with energy
    def mu_bone(self, E_keV):
        E = np.asarray(E_keV, dtype=float)
        return 0.45 * (40.0/E)**3 + 0.05
    def mu_air(self, E_keV):
        E = np.asarray(E_keV, dtype=float)
        return 0.001 * np.ones_like(E)

# ---------------- Phantoms ----------------
def _ellipse(img, xc, yc, a, b, angle_rad, value):
    H, W = img.shape
    Y, X = np.indices((H, W))
    x = (X - W/2.0) / (W/2.0)
    y = (Y - H/2.0) / (H/2.0)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    xr =  (x - xc)*ca + (y - yc)*sa
    yr = -(x - xc)*sa + (y - yc)*ca
    mask = (xr**2 / (a**2 + 1e-12) + yr**2 / (b**2 + 1e-12)) <= 1.0
    img[mask] = value
    return img

def shepp_logan(N: int = 256):
    mu = np.zeros((N, N), dtype=np.float32)
    # simple scaled variant (values are relative attenuation)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0.0, 0.20),
        (0.0, -0.0184, 0.6624, 0.8740, 0.0, -0.18),
        (0.22, 0.0, 0.11, 0.31, -0.4, -0.16),
        (-0.22, 0.0, 0.16, 0.41, 0.4, -0.16),
        (0.0, 0.35, 0.21, 0.25, 0.0, 0.10),
        (0.0, 0.1, 0.046, 0.046, 0.0, 0.18),
        (0.0, -0.1, 0.046, 0.046, 0.0, 0.18),
        (-0.08, -0.605, 0.046, 0.023, 0.0, 0.18),
        (0.0, -0.605, 0.023, 0.023, 0.0, 0.18),
        (0.06, -0.605, 0.046, 0.023, 0.0, 0.18),
    ]
    for xc, yc, a, b, ang, val in ellipses:
        _ellipse(mu, xc, yc, a, b, ang, val)
    mu = (mu - mu.min()) / (mu.max() - mu.min() + 1e-12)
    return mu

def cylinders_phantom(N: int = 256):
    mu = np.zeros((N, N), dtype=np.float32)
    H, W = mu.shape
    Y, X = np.indices((H, W))
    x = (X - W/2.0) / (W/2.0)
    y = (Y - H/2.0) / (H/2.0)
    r = np.sqrt(x**2 + y**2)
    # background water-ish
    mu[:] = 0.20
    # central air hole
    mu[r < 0.15] = 0.001
    # outer bone ring
    ring = (r > 0.55) & (r < 0.75)
    mu[ring] = 0.45
    return mu

# ---------------- Projector ----------------
class Projector:
    def __init__(self, geometry: str = "parallel"):
        self.geometry = geometry

# ---------------- Forward model ----------------
def _bilinear(img, x, y):
    # x,y in normalized [-1,1]; img shape N x N
    N = img.shape[0]
    gx = (x * 0.5 + 0.5) * (N - 1)
    gy = (y * 0.5 + 0.5) * (N - 1)
    x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
    x1 = np.clip(x0 + 1, 0, N-1); y1 = np.clip(y0 + 1, 0, N-1)
    x0 = np.clip(x0, 0, N-1); y0 = np.clip(y0, 0, N-1)
    wx = gx - x0; wy = gy - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return (1-wx)*(1-wy)*Ia + wx*(1-wy)*Ib + (1-wx)*wy*Ic + wx*wy*Id

def _ray_integral(mu_img, x0, y0, x1, y1, steps=256):
    # integrate mu along straight line from (x0,y0) to (x1,y1) in normalized coords [-1,1]
    t = np.linspace(0.0, 1.0, steps)
    x = x0 + (x1 - x0)*t
    y = y0 + (y1 - y0)*t
    mu_s = _bilinear(mu_img, x, y)
    # path length scale: total length ~ 2*sqrt(2) for full cross
    L = np.hypot(x1-x0, y1-y0)
    return float(mu_s.mean() * L * 2.0)  # crude scaling to avoid underflow

def forward_intensity(mu_true, projector: Projector, spectra: Spectra, db: MaterialDB,
                      materials_map=None, poly=True, n_det=360, n_proj=360):
    # Build sinogram of intensities I(theta, s) with crude line integration
    N = mu_true.shape[0]
    thetas = np.linspace(0.0, np.pi, int(n_proj), endpoint=False)
    det = np.linspace(-1.0, 1.0, int(n_det), endpoint=False)
    I = np.zeros((int(n_proj), int(n_det)), dtype=np.float32)

    # If materials_map provided, remap mu_true to mixture of materials (toy)
    # otherwise use mu_true as absolute attenuation proxy.
    if materials_map is not None:
        # map approximate scalar to materials; here we just clamp to given representative values
        mu_img = np.clip(mu_true, 0.001, 0.45).astype(np.float32)
    else:
        mu_img = mu_true.astype(np.float32)

    if poly:
        Es, w = spectra.sample_spectrum(5)
        mu_scale = db.mu_water(Es)  # use water as baseline scaling
        mu_scale = mu_scale / (mu_scale.max() + 1e-12)
    else:
        Es = [spectra.effective_energy()]; w = [1.0]
        mu_scale = np.array([1.0])

    for ti, th in enumerate(thetas):
        d = np.array([np.cos(th), np.sin(th)], dtype=float)
        n = np.array([-np.sin(th), np.cos(th)], dtype=float)
        for si, s in enumerate(det):
            p0 = -np.sqrt(2.0)*d + s*n
            p1 =  np.sqrt(2.0)*d + s*n
            # energy average
            val = 0.0
            for k in range(len(Es)):
                line = _ray_integral(mu_img * mu_scale[k], p0[0], p0[1], p1[0], p1[1], steps=max(64, N))
                val += w[k] * np.exp(-line)
            I[ti, si] = val
    return I, thetas, det

# ---------------- Reconstruction ----------------
def _ramp_filter(sino_row):
    # Ram-Lak filter in freq domain (1D)
    n = len(sino_row)
    f = np.fft.rfftfreq(n)
    H = np.abs(f)
    return np.fft.irfft(np.fft.rfft(sino_row)*H, n=n)

class Reconstructor:
    def __init__(self, method="fbp", n_iter=30, relax=0.5):
        self.method = method
        self.n_iter = int(n_iter)
        self.relax = float(relax)

    def fbp(self, sino_log, thetas, out_shape):
        # Parallel-beam filtered backprojection
        n_proj, n_det = sino_log.shape
        N = int(out_shape[0])
        rec = np.zeros((N, N), dtype=np.float32)
        # coords
        y, x = np.indices((N, N))
        x = (x - N/2.0) / (N/2.0)
        y = (y - N/2.0) / (N/2.0)
        for ti, th in enumerate(thetas):
            s = x*np.cos(th) + y*np.sin(th)  # detector coord in [-1,1]
            # map s to detector index
            u = (s*0.5 + 0.5) * (n_det - 1)
            u0 = np.clip(np.floor(u).astype(int), 0, n_det-1)
            u1 = np.clip(u0+1, 0, n_det-1)
            w = u - u0
            # filter this projection
            p = _ramp_filter(sino_log[ti])
            val = (1-w)*p[u0] + w*p[u1]
            rec += val
        rec = rec * (np.pi / n_proj)
        rec = rec - rec.min()
        rec = rec / (rec.max() + 1e-12)
        return rec

    def sirt(self, sino_log, thetas, projector: Projector, out_shape):
        # Very simple unfiltered backprojection as placeholder for fan/SIRT
        return self.fbp(sino_log, thetas, out_shape)

# ---------------- HU ----------------
def to_hu(mu_img, mu_water_eff):
    mu = np.asarray(mu_img, dtype=float)
    return 1000.0 * (mu - mu_water_eff) / (mu_water_eff + 1e-12)
