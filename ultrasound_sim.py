
# ============================================================
# ultrasound_sim.py — Educational Ultrasound Simulation Toolkit
# ============================================================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

# ---------------------------
# Utilities
# ---------------------------

def analytic_signal(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Hilbert-transform based analytic signal (no SciPy)."""
    x = np.asarray(x)
    N = x.shape[axis]
    Xf = np.fft.fft(x, axis=axis)
    H = np.zeros(N, dtype=np.float64)
    if N % 2 == 0:
        H[0] = 1.0
        H[N//2] = 1.0
        H[1:N//2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N+1)//2] = 2.0
    # reshape H to broadcast along axis
    shp = [1]*x.ndim
    shp[axis] = N
    H = H.reshape(shp)
    xa = np.fft.ifft(Xf * H, axis=axis)
    return xa

def envelope_db(x: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Return log-compressed envelope (dB), normalized to max=0 dB."""
    env = np.abs(analytic_signal(x, axis=-1))
    env = env / (np.max(env) + 1e-12)
    db = 20.0 * np.log10(env + 1e-12)
    db = np.maximum(db, floor_db)
    return db

def gaussian_pulse(f0: float, fs: float, cycles: float = 2.0, frac_bw: float = 0.6) -> np.ndarray:
    """Generate a bandlimited pulse centered at f0 using a Gaussian envelope.
    cycles: approx number of cycles at -6 dB
    frac_bw: fractional bandwidth at -6 dB (approximate)
    """
    T = cycles / float(f0)
    t = np.arange(int(np.ceil(T*fs))) / fs
    sigma = T / (2.0*np.sqrt(2.0*np.log(2.0)))  # relate to -6 dB width
    g = np.exp(-0.5*((t - T/2)/sigma)**2)
    s = g * np.cos(2.0*np.pi*f0*(t - T/2))
    s -= np.mean(s)
    return s.astype(np.float32)

def db_atten_amp(alpha_db_per_mhz_cm: float, f0_hz: float, path_m: float) -> float:
    """Amplitude attenuation for power-law alpha [dB/(MHz·cm)], frequency f0, path length (two-way) in meters.
    A_amp = 10^(- alpha * f_MHz * L_cm / 20)
    """
    f_mhz = f0_hz * 1e-6
    L_cm = path_m * 100.0
    att_db = alpha_db_per_mhz_cm * f_mhz * L_cm
    return float(10.0**(-att_db / 20.0))

def lin_interp(signal: np.ndarray, t: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolation along last axis of signal sampled at times t; returns samples at t_query.
    signal: (..., Nt)
    t, t_query: (Nt,), (K,)
    returns: (..., K)
    """
    Nt = signal.shape[-1]
    dt = float(t[1]-t[0]) if Nt>1 else 1.0
    idx = (t_query - t[0]) / dt
    i0 = np.clip(np.floor(idx).astype(np.int64), 0, Nt-2)
    w = (idx - i0).astype(np.float64)
    s0 = np.take(signal, i0, axis=-1)
    s1 = np.take(signal, i0+1, axis=-1)
    return (1.0 - w) * s0 + w * s1

# ---------------------------
# Core Data Classes
# ---------------------------

@dataclass
class Medium:
    c: float = 1540.0            # speed of sound [m/s]
    rho: float = 1000.0          # density [kg/m^3]
    alpha_db_cm_mhz: float = 0.5 # attenuation coefficient [dB/(MHz·cm)]
    alpha_power: float = 1.0     # power law exponent y (for extensions)

@dataclass
class Transducer:
    kind: str = "linear"         # linear | phased
    f0: float = 5e6              # center frequency [Hz]
    pitch: float = 0.0003        # element spacing [m] (0.3 mm)
    N: int = 64                  # number of elements
    frac_bw: float = 0.6         # fractional bandwidth ~ -6 dB
    tx_cycles: float = 2.0       # transmit pulse cycles

    def positions(self) -> np.ndarray:
        x = (np.arange(self.N) - (self.N-1)/2.0) * self.pitch
        return x.astype(np.float64)

@dataclass
class Beamformer:
    method: str = "das"          # das | capon (placeholder)
    apod: str = "hanning"        # none | hanning | hamming
    tx_focus_z: float = 0.03     # [m]
    tx_focus_x: float = 0.0      # [m]

    def apod_weights(self, N: int) -> np.ndarray:
        if self.apod == "hanning":
            w = np.hanning(N)
        elif self.apod == "hamming":
            w = np.hamming(N)
        else:
            w = np.ones(N, dtype=np.float64)
        return (w / np.max(w)).astype(np.float64)

@dataclass
class ScanConverter:
    x_span: Tuple[float, float] = (-0.015, 0.015)  # +/- 15 mm
    z_span: Tuple[float, float] = (0.005, 0.06)    # 5-60 mm
    nx: int = 128
    nz: int = 512

    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.x_span[0], self.x_span[1], self.nx)
        z = np.linspace(self.z_span[0], self.z_span[1], self.nz)
        X, Z = np.meshgrid(x, z, indexing="xy")
        return X, Z

# ---------------------------
# Channel Simulation (RF)
# ---------------------------

@dataclass
class Scatterer:
    x: float
    z: float
    amp: float = 1.0  # reflectivity
    vx: float = 0.0   # lateral velocity [m/s]
    vz: float = 0.0   # axial velocity [m/s]

def simulate_rf_channels(
    medium: Medium,
    tx: Transducer,
    beam: Beamformer,
    scatterers: List[Scatterer],
    fs: float = 40e6,
    t_max: float = 80e-6,
    ensembles: int = 1,
    pri: float = 1e-4,  # pulse repetition interval [s] for Doppler
) -> Dict[str, np.ndarray]:
    """Simulate raw RF channel data for a linear array using a simple point-scatterer model.
    Returns dict with: t, rf (ensembles x Nch x Nt), tx_pulse
    """
    x_e = tx.positions()                  # (N,)
    Nch = tx.N
    Nt = int(np.ceil(t_max*fs))
    t = np.arange(Nt)/fs
    # transmit pulse
    tx_pulse = gaussian_pulse(tx.f0, fs, cycles=tx.tx_cycles, frac_bw=tx.frac_bw)
    Nt_p = len(tx_pulse)

    rf = np.zeros((ensembles, Nch, Nt), dtype=np.float32)
    for m in range(ensembles):
        # scatterer positions for this ensemble (Doppler update)
        scas = []
        for s in scatterers:
            x_m = s.x + m*pri*s.vx
            z_m = s.z + m*pri*s.vz
            scas.append((x_m, z_m, s.amp))

        for ch in range(Nch):
            xe = x_e[ch]
            # one-way receive distance from scatterer to element
            for (xs, zs, amp) in scas:
                r_rx = np.sqrt((xe - xs)**2 + zs**2)
                r_tx = np.sqrt((0.0 - xs)**2 + zs**2)  # from array center (approx focused TX)
                tau = (r_tx + r_rx)/medium.c
                idx0 = int(np.round(tau*fs))
                if idx0 < Nt:
                    # attenuation (two-way path length)
                    A = db_atten_amp(medium.alpha_db_cm_mhz, tx.f0, r_tx + r_rx)
                    amp_eff = amp * A / max(r_tx + r_rx, 1e-6)  # spherical spreading ~ 1/r
                    i_start = idx0
                    i_end = min(idx0 + Nt_p, Nt)
                    seg_len = i_end - i_start
                    rf[m, ch, i_start:i_end] += amp_eff * tx_pulse[:seg_len]

    return {"t": t, "rf": rf, "tx_pulse": tx_pulse, "fs": fs}

# ---------------------------
# Delay-and-Sum Beamforming
# ---------------------------

def das_beamform(
    rf: np.ndarray, t: np.ndarray, fs: float,
    x_elems: np.ndarray,
    medium: Medium,
    Ximg: np.ndarray, Zimg: np.ndarray,
    apod: Optional[np.ndarray] = None
) -> np.ndarray:
    """Dynamic receive focusing delay-and-sum to form image on (Ximg,Zimg)."""
    ensembles, Nch, Nt = rf.shape
    if apod is None:
        apod = np.ones(Nch, dtype=np.float64)
    apod_flat = apod.flatten()  # ensure 1D array of length Nch

    nz = Zimg.shape[0]
    nx = Ximg.shape[1]
    
    # output envelope (per ensemble), then average later if needed
    img_env = np.zeros((ensembles, nx, nz), dtype=np.float32)  # (ens, nx, nz)

    dt = float(t[1] - t[0]) if Nt > 1 else 1.0
    t0 = float(t[0])

    for e in range(ensembles):
        rf_e = rf[e]  # (Nch, Nt)
        for ix in range(nx):
            x = Ximg[0, ix]
            z_col = Zimg[:, ix]  # (nz,)
            
            # distances from center (TX) to each depth point
            r_tx = np.sqrt(x**2 + z_col**2)  # (nz,)
            
            col_sum = np.zeros(nz, dtype=np.float64)
            
            for ich in range(Nch):
                xe = x_elems[ich]
                # distance from element to each depth point
                r_rx = np.sqrt((xe - x)**2 + z_col**2)  # (nz,)
                
                # total travel time
                tau = (r_tx + r_rx) / medium.c  # (nz,)
                
                # convert to sample indices
                idx_float = (tau - t0) / dt
                idx0 = np.clip(np.floor(idx_float).astype(np.int64), 0, Nt - 2)
                w = idx_float - idx0
                w = np.clip(w, 0.0, 1.0)
                
                # linear interpolation
                s0 = rf_e[ich, idx0]
                s1 = rf_e[ich, np.minimum(idx0 + 1, Nt - 1)]
                samples = (1.0 - w) * s0 + w * s1
                
                col_sum += apod_flat[ich] * samples
            
            # envelope of beamformed column
            env_col = np.abs(analytic_signal(col_sum))
            img_env[e, ix, :] = env_col.astype(np.float32)

    # combine ensembles (e.g., magnitude average)
    img = np.mean(img_env, axis=0)  # (nx, nz)
    return img.T  # -> (nz, nx)

# ---------------------------
# Doppler (Autocorrelation Estimator, Kasai)
# ---------------------------

def doppler_autocorr(rf: np.ndarray, lag: int = 1) -> np.ndarray:
    """Autocorrelation-based Doppler estimate (Kasai).
    rf: (ensembles, Nch, Nt) - we first build a channel-averaged signal.
    Returns: complex R(lag) over slow-time for each fast-time sample (Nt,).
    """
    y = np.mean(rf, axis=1)  # (ensembles, Nt)
    y = y - np.mean(y, axis=0, keepdims=True)
    R = np.sum(y[:-lag, :] * np.conj(y[lag:, :]), axis=0)  # (Nt,)
    return R

def doppler_color_map(
    rf: np.ndarray, t: np.ndarray, fs: float, f0: float, c: float,
    ensembles: int, pri: float, dynamic_range_db: float = 40.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vel_map, power_map) on slow-time collapsed to depth-time (Nt,)."""
    R1 = doppler_autocorr(rf, lag=1)  # (Nt,)
    fd = np.angle(R1) / (2.0*np.pi*pri)  # mean Doppler [Hz]
    v = (fd * c) / (2.0 * f0)
    pwr = 20.0*np.log10(np.abs(R1) / (np.max(np.abs(R1))+1e-12) + 1e-12)
    pwr = np.maximum(pwr, -dynamic_range_db)
    return v, pwr

# ---------------------------
# High-level Convenience
# ---------------------------

def simulate_bmode(
    medium: Medium, tx: Transducer, beam: Beamformer, sc: ScanConverter,
    scatterers: List[Scatterer], fs: float = 40e6, t_max: float = 80e-6,
    ensembles: int = 1, pri: float = 1e-4, dr_db: float = 60.0
) -> Dict[str, np.ndarray]:
    """Simulate RF channels and form a B-mode image with DAS."""
    ch = simulate_rf_channels(medium, tx, beam, scatterers, fs=fs, t_max=t_max, ensembles=ensembles, pri=pri)
    t = ch["t"]; rf = ch["rf"]
    X, Z = sc.grid()
    img_lin = das_beamform(rf, t, ch["fs"], tx.positions(), medium, X, Z, apod=Beamformer(beam.method, beam.apod).apod_weights(tx.N))
    env = img_lin / (np.max(img_lin)+1e-12)
    bmode = 20.0*np.log10(env + 1e-12)
    bmode = np.clip(bmode, -dr_db, 0.0)
    return {"bmode_db": bmode, "X": X, "Z": Z, "t": t, "rf": rf, "fs": ch["fs"]}

# ---------------------------
# Phantoms
# ---------------------------

@dataclass
class Scatterer2D:
    x: float
    z: float
    amp: float = 1.0

def point_scatterer_phantom(x: float=0.0, z: float=0.03, amp: float=1.0) -> List[Scatterer]:
    return [Scatterer(x=x, z=z, amp=amp)]

def carotid_phantom(
    width: float = 0.006, depth: float = 0.025, length: float = 0.03,
    flow_vel: float = 0.3,  # m/s (peak)
    rho_scatter: float = 1e5, seed: int = 0
) -> List[Scatterer]:
    """Random speckle + cylindrical vessel (laminar profile)."""
    rng = np.random.default_rng(seed)
    scatters: List[Scatterer] = []
    # tissue speckle around region z in [10, 50] mm, x in [-15, 15] mm
    Nx = 2000
    xs = rng.uniform(-0.015, 0.015, size=Nx)
    zs = rng.uniform(0.01, 0.05, size=Nx)
    amps = rng.normal(0.0, 1.0, size=Nx) * 0.3
    for i in range(Nx):
        scatters.append(Scatterer(x=float(xs[i]), z=float(zs[i]), amp=float(amps[i])))
    # vessel (cylinder axis along y): center at (0, depth), radius = width/2
    r0 = width/2.0
    Nv = 2000
    xs = rng.uniform(-0.015, 0.015, size=Nv)
    zs = rng.uniform(depth - r0, depth + r0, size=Nv)
    for i in range(Nv):
        r = np.sqrt(xs[i]**2 + (zs[i]-depth)**2)
        if r <= r0:
            vz = flow_vel * (1.0 - (r/r0)**2)  # laminar parabolic, axial along z
            scatters.append(Scatterer(x=float(xs[i]), z=float(zs[i]), amp=0.5, vz=float(vz)))
    return scatters
