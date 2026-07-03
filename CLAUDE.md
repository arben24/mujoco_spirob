# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands use **uv** exclusively — never pip, virtualenv, or python -m.

```bash
# Setup
uv venv && uv pip install -e .

# Run an app
uv run apps/<script>.py

# Test
uv run pytest

# Single test file
uv run pytest tests/test_math_spirob.py

# Docs
cd docs && uv run make html

# Headless video recording
PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video
```

## Key Apps

| Script | Purpose |
|---|---|
| `apps/generate_2d_spirob.py` | Generate MuJoCo XML from spiral parameters; output is the `.xml` used by Simulink |
| `apps/spirob_metrics.py` | Batch parameter sweep — edit `VARIABLE_PARAMS` / `GEOM_SCENARIOS` at top of file to configure runs |
| `apps/spirob_analyse.py [--enable-position-estimation]` | Compute metrics for all `build/experiments/Run_XXX` dirs |
| `apps/sys_id/spirob_sysid.py [--mode uniform\|per-joint] [--method ...]` | System identification via `scipy.optimize`; results saved to `build/sysid_params.json` |
| `apps/spirob_plot_gui.py` | Interactive Qt GUI for filtering and plotting runs |
| `apps/spirob_digital_twin.py` | Passive MuJoCo viewer with optional hardware (pyserial/customtkinter) |

`simulation_configs.json` is a **generated artifact** written by `spir_sim.save_configs_to_json()`, not a configuration input.

## Architecture & Data Flow

```
VARIABLE_PARAMS + GEOM_SCENARIOS / SysIdConfig
    → generate_xml_string()          # SpiralCalculator (bisect) → XMLBuilder + SensorRegistry
    → mj.MjModel.from_xml_string()
    → run_simulation_with_data_collection()   # ControllerFunc callback → Polars DataFrame
    → exporter.save_experiment(df, record)    # atomic: data.parquet + meta.json per run
    → analyzer.load_experiment("Run_XXX")     # → (ExperimentRecord, LazyFrame)
    → meta_analyzer / plots.py               # lazy aggregation, Matplotlib/Seaborn output
```

**Library entry point**: always import from `src/math_spirob/__init__.py`, not submodules directly.

## Critical Files

| File | Role |
|---|---|
| `src/math_spirob/__init__.py` | Public API |
| `src/math_spirob/data_schema.py` | All Pydantic models (`ExperimentConfig`, `ExperimentRecord`, `SensorMeta`, `DataGroup`) |
| `src/math_spirob/spirob_simulate.py` | Simulation loop, controllers, contact force extraction |
| `src/math_spirob/spirob_generator.py` | `SpiralCalculator`, `XMLBuilder`, `SensorRegistry` |
| `src/math_spirob/exporter.py` | `save_experiment()`, `generate_sensor_meta()` |

## Sensor Naming Convention

Column names in `data.parquet` must match exactly:

| Pattern | Example |
|---|---|
| 1D sensor | `tendon_frc_0`, `tendon_pos_0` |
| 3D sensor (uppercase axes) | `acc_0_X`, `gyro_0_Z`, `body_spiral_0_contact_force_Y` |
| 3D geom/estimate (lowercase) | `geom_pos_0_x`, `pos_estimate_0_z` |
| 4D vel estimate | `vel_estimate_0_x/y/z/_norm` |
| 4D quaternion | `quat_estimate_0_w/x/y/z` |

Sensors are registered in `generate_xml_string()` via `SensorRegistry.register("acc", "accelerometer")` **before** `XMLBuilder` is called. To add a sensor type: register it there and add a `DataGroup` entry in `data_schema.py`.

## Controller Interface

```python
ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]
# Tendon actuators: ctrlrange=[-50, 0] — pull only; positive ctrl values are no-ops
# data.ctrl[0] = -ramp * 5.0  # negative = tension
# Built-ins: static_controller, ramped_controller, sine_controller, PIDController (class)
```

## Key Patterns

```python
# XML generation
xml = sg.generate_xml_string(L_target=0.3, base_d=0.05, tip_d=0.01, Delta_theta_deg=30)

# Simulation + data collection
df, record = spir_sim.run_simulation_with_data_collection(
    model, controller=ramped_controller, sim_time=2.0, include_geom_pos=True)

# Load experiment (lazy — don't .collect() early)
record, lf = analyzer.load_experiment("Run_001")

# Sensor metadata (auto-infer from DataFrame columns — don't construct SensorMeta manually)
sensor_metas = exporter.generate_sensor_meta(df)

# Metadata filtering
meta_df = plots.load_run_metadata()
run_ids = plots.filter_runs(meta_df, {"controller_info": "Ramped", "L_target_min": 0.3})

# Discover all runs
records = meta_analyzer.crawl_experiments()
```

## Contact Force Convention

`extract_body_contact_forces()` returns world-frame forces per body. `mj_contactForce` gives force in contact frame; `contact.frame` (3×3 rotation matrix) transforms it to world frame. `body1 += F_world`, `body2 -= F_world` (Newton's 3rd law).

## Experiment Storage

Each run writes to `build/experiments/Run_XXX_<descriptor>/`:
- `data.parquet` — time-series sensor data (Polars)
- `meta.json` — Pydantic-validated `ExperimentRecord`

## MATLAB/Simulink Co-Simulation

Generate the XML first with `uv run apps/generate_2d_spirob.py`, then insert it into the MuJoCo Plant block in `apps/Matlab/Spirob_Simulink.slx`. On Linux, launch MATLAB from `/usr/local/MATLAB/R2025b/bin/matlab` and save pathdef with `savepath ~/Documents/MATLAB/pathdef.m`.
