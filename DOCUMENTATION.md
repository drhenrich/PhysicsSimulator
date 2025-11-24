# Developer Documentation (Bundled Version)

## Architecture

The bundled app splits into five modules under `bundle/` plus a main entrypoint:

- `physics_sim.py`: Streamlit app wiring tabs together.
- `i18n_bundle.py`: Translation helper `get_text`/`get_language_name`.
- `sim_core_bundle.py`: Physics core (Body, Simulator), plotting stubs, enhanced scenarios, simplified presets.
- `ui_mech_bundle.py`: Mechanics UI and Sun–Earth orbit helper.
- `ui_optics_bundle.py`: Wave optics + paraxial raytracing UI/core.
- `ui_med_bundle.py`: CT reduced, self-contained Xray/CT fast, MRI/Bloch, Electrodynamics UIs.

The original, more granular modules remain in the repo root for reference. The bundle is a minimal, dependency-light subset designed to run without missing external cores.

## Running

```bash
pip install streamlit numpy matplotlib plotly
streamlit run bundle/physics_sim.py
```

Language defaults to German; change via the header selector.

## Key Components

- **sim_core_bundle**: Provides `Body`, `Simulator` (Euler), plotting utilities (Plotly/Matplotlib fallback), enhanced scenarios (charged pair, collisions, springs, planetary), and presets for Bloch, optics, CT, mechanics.
- **ui_mech_bundle**: Renders mechanics tab (basic info) and a Sun–Earth orbit demo using a symplectic Euler integrator.
- **ui_optics_bundle**: Wave optics (simple Fraunhofer patterns) and raytracing using paraxial ABCD-style propagation; presets cover common lens setups.
- **ui_med_bundle**: 
  - `render_ct_safe_tab`: simple phantom + sinogram + FBP.
  - `render_xray_ct_tab`: self-contained fast sinogram + FBP, no external `xray_ct`.
  - `render_mri_bloch_tab`: simple MRI image reconstruction + Bloch plotting.
  - `render_em_tab`: electrostatics, Poisson solver, electrodynamics presets.

## Localization

`i18n_bundle.py` stores DE/EN strings for high-level labels. Some embedded text is still localized via inline lambdas (`tr`) inside UIs. Add more keys to `i18n_bundle.py` to centralize translations.

## Extending

- To add presets: extend `PRESETS_ENHANCED` or domain-specific preset dicts in `sim_core_bundle.py` or UI modules.
- To change physics integrators: update `Simulator` in `sim_core_bundle.py` or swap in your own core and adjust imports.
- To restore richer features from the original codebase, refer to the root modules and port functionality into the bundled modules.

## Testing

Run `python -m py_compile bundle/*.py` to check syntax. The app itself runs via Streamlit; most functions rely on user interaction rather than unit tests.
