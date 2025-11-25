
# ============================================================
# ui_ultrasound.py — Streamlit UI for the Ultrasound module
# Exposes: render_ultrasound_tab()
# ============================================================
from __future__ import annotations

def render_ultrasound_tab():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    try:
        from ultrasound_sim import (
            Medium, Transducer, Beamformer, ScanConverter,
            Scatterer, point_scatterer_phantom, carotid_phantom,
            simulate_rf_channels, simulate_bmode, envelope_db
        )
    except Exception as e:
        st.error(f"Ultraschall-Core nicht verfügbar: {e}")
        return

    st.subheader("Akustik / Ultraschall")

    left, right = st.columns([3,1], gap="large")

    with right:
        st.markdown("**Medium**")
        c = st.number_input("Schallgeschwindigkeit c [m/s]", min_value=1000.0, max_value=2000.0, value=1540.0, step=1.0, key="us_c")
        rho = st.number_input("Dichte ρ [kg/m³]", min_value=500.0, max_value=1500.0, value=1000.0, step=1.0, key="us_rho")
        alpha = st.number_input("Dämpfung α [dB/(MHz·cm)]", min_value=0.0, max_value=2.0, value=0.5, step=0.1, format="%.2f", key="us_alpha")

        st.markdown("---")
        st.markdown("**Schallkopf (Lineararray)**")
        f0 = st.number_input("Zentralfrequenz f0 [MHz]", min_value=1.0, max_value=20.0, value=7.0, step=0.5, key="us_f0") * 1e6
        pitch = st.number_input("Pitch [mm]", min_value=0.1, max_value=1.0, value=0.3, step=0.05, key="us_pitch") * 1e-3
        Ne = st.number_input("Elemente N", min_value=16, max_value=128, value=64, step=1, key="us_N")
        frac_bw = st.slider("Bandbreite (fraktional)", min_value=0.2, max_value=1.0, value=0.6, step=0.05, key="us_bw")
        tx_cycles = st.slider("Pulse (Zyklen)", min_value=1, max_value=5, value=2, step=1, key="us_cycles")

        st.markdown("---")
        st.markdown("**Beamforming**")
        focus = st.number_input("TX Fokus z [mm]", min_value=5.0, max_value=80.0, value=30.0, step=1.0, key="us_focus") * 1e-3
        apod = st.selectbox("Apodisation", ["hanning", "hamming", "none"], index=0, key="us_apod")

        st.markdown("---")
        st.markdown("**Erfassung**")
        fs = st.number_input("Abtastrate fs [MHz]", min_value=10.0, max_value=100.0, value=40.0, step=5.0, key="us_fs") * 1e6
        tmax = st.number_input("t_max [µs]", min_value=20.0, max_value=200.0, value=80.0, step=5.0, key="us_tmax") * 1e-6
        ens = st.number_input("Ensembles (Doppler)", min_value=1, max_value=64, value=8, step=1, key="us_ensembles")
        pri = st.number_input("PRI [µs]", min_value=50.0, max_value=1000.0, value=200.0, step=10.0, key="us_pri") * 1e-6
        dr = st.slider("Dynamikbereich B-Mode [dB]", min_value=40, max_value=80, value=60, step=5, key="us_dr")

        st.markdown("---")
        st.markdown("**Bildraster**")
        xspan = st.slider("x-Span [mm]", min_value=10, max_value=40, value=30, step=1, key="us_xspan") * 1e-3
        zmin = st.number_input("z_min [mm]", min_value=5.0, max_value=30.0, value=10.0, step=1.0, key="us_zmin") * 1e-3
        zmax = st.number_input("z_max [mm]", min_value=30.0, max_value=100.0, value=60.0, step=1.0, key="us_zmax") * 1e-3
        nx = st.slider("nx", min_value=64, max_value=256, value=128, step=32, key="us_nx")
        nz = st.slider("nz", min_value=256, max_value=1024, value=512, step=128, key="us_nz")

        st.markdown("---")
        scenario = st.selectbox("Szenario", ["Punktstreuer-PSF", "Carotis-Phantom", "Array-Steuerung"], index=1, key="us_scenario")
        run = st.button("Simulieren", use_container_width=True, key="us_run")

    with left:
        if run:
            medium = Medium(c=float(c), rho=float(rho), alpha_db_cm_mhz=float(alpha))
            txd = Transducer(kind="linear", f0=float(f0), pitch=float(pitch), N=int(Ne), frac_bw=float(frac_bw), tx_cycles=int(tx_cycles))
            beam = Beamformer(method="das", apod=str(apod), tx_focus_z=float(focus), tx_focus_x=0.0)
            sc = ScanConverter(x_span=(-xspan/2, xspan/2), z_span=(float(zmin), float(zmax)), nx=int(nx), nz=int(nz))

            if scenario == "Punktstreuer-PSF":
                scas = [Scatterer(x=0.0, z=(zmin+zmax)/2.0, amp=1.0)]
            elif scenario == "Carotis-Phantom":
                scas = carotid_phantom(width=0.006, depth=0.025, length=0.03, flow_vel=0.3, rho_scatter=1e5, seed=1)
            else:  # Array-Steuerung
                scas = [Scatterer(x=-0.004, z=0.02, amp=1.0),
                        Scatterer(x= 0.000, z=0.03, amp=1.0),
                        Scatterer(x= 0.004, z=0.045, amp=1.0)]

            out = simulate_bmode(medium, txd, beam, sc, scas, fs=float(fs), t_max=float(tmax), ensembles=int(ens), pri=float(pri), dr_db=float(dr))
            bmode = out["bmode_db"]; X = out["X"]; Z = out["Z"]

            fig = go.Figure(data=go.Heatmap(
                z=bmode, x=X[0, :]*1e3, y=Z[:, 0]*1e3,
                colorscale="Greys", reversescale=True, zmin=-float(dr), zmax=0.0,
                coloraxis="coloraxis", showscale=True,
                hovertemplate="x=%{x:.1f} mm<br>z=%{y:.1f} mm<br>B=%{z:.1f} dB<extra></extra>"
            ))
            fig.update_layout(
                xaxis_title="x [mm]", yaxis_title="z [mm]",
                height=720, margin=dict(l=10, r=10, t=40, b=10),
                title="B-Mode (DAS)"
            )
            st.plotly_chart(fig, use_container_width=True, key="us_bmode")

            if scenario == "Carotis-Phantom" and int(ens) >= 4:
                rf = out["rf"]
                tarr = out["t"]; fs_val = float(out["fs"])
                y = np.mean(rf, axis=1)  # (ens, Nt)
                R1 = np.sum(y[:-1, :] * np.conj(y[1:, :]), axis=0)
                fd = np.angle(R1) / (2.0*np.pi*float(pri))
                v = (fd * float(c)) / (2.0 * float(f0))

                z_axis = (tarr * float(c) / 2.0) * 1e3
                import pandas as pd
                df = pd.DataFrame({"z_mm": z_axis, "v_mps": v})
                df = df[(df["z_mm"] >= Z[0,0]*1e3) & (df["z_mm"] <= Z[-1,0]*1e3)]
                fig2 = go.Figure(data=go.Scatter(x=df["v_mps"], y=df["z_mm"], mode="lines"))
                fig2.update_layout(xaxis_title="v [m/s]", yaxis_title="z [mm]", height=480, title="Farbdoppler (Axialprofil, grob)")
                st.plotly_chart(fig2, use_container_width=True, key="us_doppler")

            if st.button("RF als NPZ herunterladen", key="us_dl_rf"):
                import io, numpy as _np
                buf = io.BytesIO()
                _np.savez_compressed(buf, rf=out["rf"], t=out["t"], fs=out["fs"])
                st.download_button("Download rf_data.npz", buf.getvalue(), file_name="rf_data.npz", mime="application/octet-stream", key="us_dlbtn")

        else:
            st.info("Parameter rechts wählen und 'Simulieren' starten.")
