# Mujoco Spirob - AI Agent Instructions

## Project Overview
Mathematical and simulation tools for logarithmic spiral structures (SpiRob) using MuJoCo. Core components:
- `src/math_spirob/`: Library — math functions, XML generation, simulation, analysis, plotting
- `apps/`: Executable scripts (always run via `uv run apps/<script>.py`)
- `build/experiments/Run_XXX/`: Per-run outputs — `meta.json` (Pydantic-validated) + `data.parquet`

## Essential Workflows
- **Setup**: `uv venv && uv pip install -e .` — **uv only**, never pip/virtualenv/python -m
- **Run Apps** (`uv run apps/<script>.py`):
  - `generate_2d_spirob.py` — generate MuJoCo XML from spiral parameters
  - `spirob_metrics.py` — batch parameter sweep; edit `VARIABLE_PARAMS`/`GEOM_SCENARIOS` in the file to configure runs; `simulation_configs.json` is a **generated artifact** (saved via `spir_sim.save_configs_to_json()`), not an input
  - `spirob_analyse.py [--enable-position-estimation]` — compute metrics for all runs; ACC+GYRO dead-reckoning writes `pos_estimate`/`vel_estimate` columns back to `data.parquet`
  - `spirob_sysid.py [--mode uniform|per-joint] [--method differential_evolution|Nelder-Mead|L-BFGS-B]` — system identification via scipy.optimize against a ground-truth MuJoCo simulation; `SysIdConfig` holds GT params and initial guesses; results saved via `--save-params build/sysid_params.json`
  - `spirob_plot_gui.py` — interactive Qt GUI for filtering/plotting runs
  - `spirob_digital_twin.py` — passive MuJoCo viewer with optional hardware (pyserial/customtkinter)
- **Test**: `uv run pytest`
- **Docs**: `cd docs && uv run make html`
- **Headless video**: `PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video`

## Architecture & Data Flow
```
SysIdConfig / VARIABLE_PARAMS+GEOM_SCENARIOS
    → generate_xml_string()        # SpiralCalculator (bisect) → XMLBuilder + SensorRegistry
    → mj.MjModel.from_xml_string()
    → run_simulation_with_data_collection()   # ControllerFunc callback → Polars DataFrame
    → exporter.save_experiment(df, record)    # atomic write: data.parquet + meta.json
    → analyzer.load_experiment("Run_XXX")     # → (ExperimentRecord, LazyFrame)
    → meta_analyzer / plots.py               # lazy aggregation, Matplotlib/Seaborn output
```

## Schema-Driven Patterns
- `data_schema.py` Pydantic models: `ExperimentConfig`, `ExperimentRecord`, `SensorMeta`, `DataGroup` enum
- `exporter.generate_sensor_meta(df)` — auto-infers `SensorMeta` list from DataFrame column names; used when saving experiments; call this rather than constructing `SensorMeta` manually
- `meta_analyzer.crawl_experiments()` — discovers all `Run_XXX` dirs; used by `spirob_analyse.py`

## Sensor Naming Convention (critical — column names must match exactly)
| Pattern | Example columns | Notes |
|---|---|---|
| 1D sensor | `tendon_frc_0`, `tendon_pos_0` | single column, no suffix |
| 3D sensor (uppercase) | `acc_0_X`, `gyro_0_Z`, `body_spiral_0_contact_force_Y` | acc, gyro, contact forces |
| 3D geom/estimate (lowercase) | `geom_pos_0_x`, `pos_estimate_0_z` | geom_pos and pos_estimate groups |
| 4D vel estimate | `vel_estimate_0_x/y/z/_norm` | 4 columns with `_norm` as 4th |
| 4D quat estimate | `quat_estimate_0_w/x/y/z` | quaternion orientation |

Sensors registered in `generate_xml_string()` via `SensorRegistry.register("acc", "accelerometer")` **before** `XMLBuilder` is called. To add a sensor type: register it there and add a `DataGroup` entry.

## Controller Interface
```python
ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]
# Tendon actuators: ctrlrange=[-50, 0] — pull only; positive ctrl values are no-ops
# Built-ins: static_controller, ramped_controller, sine_controller, PIDController (class)
data.ctrl[0] = -ramp * 5.0  # negative = tension
```

## Key Patterns
```python
# Spiral XML generation
xml = sg.generate_xml_string(L_target=0.3, base_d=0.05, tip_d=0.01, Delta_theta_deg=30)

# Simulation + data collection
df, record = spir_sim.run_simulation_with_data_collection(
    model, controller=ramped_controller, sim_time=2.0, include_geom_pos=True)

# Load experiment (lazy)
record, lf = analyzer.load_experiment("Run_001")   # LazyFrame — don't .collect() early

# Contact forces: extracted automatically, world-frame via contact.frame rotation matrix
# Forces: body1 += F_world, body2 -= F_world (Newton 3rd law)

# Metadata filtering
meta_df = plots.load_run_metadata()
run_ids = plots.filter_runs(meta_df, {"controller_info": "Ramped", "L_target_min": 0.3})
```

## Critical Files
- `src/math_spirob/__init__.py` — public API (import from here, not submodules)
- `src/math_spirob/data_schema.py` — all Pydantic models
- `src/math_spirob/spirob_simulate.py` — simulation loop, controllers, contact force extraction
- `src/math_spirob/spirob_generator.py` — `SpiralCalculator`, `XMLBuilder`, `SensorRegistry`
- `src/math_spirob/exporter.py` — `save_experiment()`, `generate_sensor_meta()`
- `apps/spirob_metrics.py` — edit `VARIABLE_PARAMS`/`GEOM_SCENARIOS` for batch runs
- `apps/spirob_sysid.py` — `SysIdConfig`, `SystemIdentifier`, normalized parameter optimization