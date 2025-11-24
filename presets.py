PRESENTS_BLOCH = {
    "Standard": {"T1": 1000.0, "T2": 80.0, "TR": 800.0, "TE": 20.0},
    "Lang T1": {"T1": 2400.0, "T2": 120.0, "TR": 3000.0, "TE": 80.0},
}

PRESENTS_OPT_WAVE = {
    "Einzelspalt 532nm": {
        "N": 256, "lam": 532.0, "z": 1.0, "method": "Fraunhofer", "shape": "Einzelspalt", "w": 0.1, "p": 0.2, "incoh": 1.0
    },
    "Doppelspalt": {
        "N": 256, "lam": 532.0, "z": 1.0, "method": "Fraunhofer", "shape": "Doppelspalt", "w": 0.1, "p": 0.3, "incoh": 1.0
    }
}

PRESENTS_CT = {
    "Shepp-Logan Schnell": {
        "phantom": "Shepp-Logan", "N": 128, "geom": "Parallel", "ndet": 180, "nproj": 120, "kVp": 80.0, "filt": 2.5,
        "poly": False, "noise": 0.0, "budget": 4
    },
    "Knochen/Luft Scan": {
        "phantom": "Zylinder (Wasser/Knochen/Luft)", "N": 192, "geom": "Parallel", "ndet": 256, "nproj": 180, "kVp": 120.0,
        "filt": 5.0, "poly": True, "noise": 0.01, "budget": 6
    }
}

PRESENTS_MECH = {
    "2 Teilchenstoß": {
        "n": 2,
        "positions": [[-0.5, 0.2], [0.5, -0.2]],
        "velocities": [[0.5, 0.2], [-0.6, 0]],
        "t_end": 10.0,
        "dt": 0.02,
    },
    "Kreisbewegung": {
        "n": 1,
        "positions": [[0.0, 1.0]],
        "velocities": [[-1.0, 0.0]],
        "t_end": 6.0,
        "dt": 0.01,
    },
}
