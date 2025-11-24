# Physics Simulator (Bundled)

A trimmed, self-contained Streamlit app for teaching physics concepts: mechanics, optics, Xray/CT, MRI/Bloch, and electrodynamics. This bundle reduces the project to a few modules under `bundle/` so it runs without external dependencies beyond standard scientific Python and Streamlit.

## Quick Start

```bash
pip install streamlit numpy matplotlib plotly
streamlit run bundle/physics_sim.py
```

The app is bilingual (DE/EN). Switch language via the selector in the header.

## Files

- `physics_sim.py` — main Streamlit entrypoint (tabs for Mechanics, Optics, Xray/CT, MRI/Bloch, Electrodynamics)
- `i18n_bundle.py` — minimal translation helper (DE/EN)
- `sim_core_bundle.py` — core physics (Body/Simulator), plotting stubs, enhanced scenarios, and simplified presets
- `ui_mech_bundle.py` — mechanics UI + simple Sun–Earth orbit
- `ui_optics_bundle.py` — wave optics + lightweight raytracing UI/core
- `ui_med_bundle.py` — CT reduced, self-contained Xray/CT fast, MRI/Bloch, electrodynamics UIs

## Features

- **Mechanics**: basic 2D mechanics tab + advanced 3D presets (elastic/inelastic collisions, charged pairs, spring system, scaled planetary)
- **Optics**: Fraunhofer wave patterns and paraxial raytracing presets
- **Xray/CT**: self-contained fast sinogram + FBP demo and a reduced CT form
- **MRI/Bloch**: simple image-space MRI reconstruction and Bloch parameter plots
- **Electrodynamics**: point-charge field visualization, Poisson solver, presets (single, dipole, quadrupole)

## Notes

- All data is generated on the fly; no external cores are required in the bundle.
- Plotly is used when available; otherwise, Matplotlib fallbacks are provided.
- The original unbundled files remain in the project root; the bundle is a clean subset for easy execution.

## License

Specify your preferred license here.
