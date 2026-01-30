# Mujoco Spirob - AI Agent Instructions

## Project Overview
This codebase implements mathematical and simulation tools for logarithmic spiral structures (SpiRob) using MuJoCo physics engine. Core components:
- `src/math_spirob/`: Pure math functions, MuJoCo XML generation, simulation runners, and experiment analysis
- `apps/`: Executable scripts for generation, control, analysis workflows, and hardware integration (digital twin)
- `build/experiments/`: Simulation run outputs with sensor data (Parquet) and metadata (JSON)

## Key Architecture Patterns
- **Modular Design**: Library (`src/`) separated from applications (`apps/`), enabling reusable components
- **Data Flow**: Config → XML generation → MuJoCo simulation → Sensor data (Polars DF) → Metrics/plots
  - Simulation configs in `simulation_configs.json` define parameter sweeps for batch experiments
  - Each run stored in `build/experiments/Run_XXX/` with `meta.json` (Pydantic-validated) and `data.parquet`
- **Schema-Driven**: Use Pydantic models (`data_schema.py`) for experiment configs and sensor metadata validation
  - `ExperimentConfig`: Simulation parameters (L_target, base_d, tip_d, controller, geom)
  - `ExperimentRecord`: Metadata with sensor list, config, timestamp for each run
  - `SensorMeta`: Describes sensor group (DataGroup enum), dimension, unit, column names
- **Lazy Evaluation**: Polars LazyFrames (`pl.scan_parquet`) for efficient data processing in `meta_analyzer.py`

## Essential Workflows
- **Setup**: `uv venv && uv pip install -e .` (uv manages deps and venv - NO pip or virtualenv!)
- **Run Apps**: Always prefix with `uv run apps/<script>.py` (examples below)
  - `uv run apps/generate_2d_spirob.py` - Generate MuJoCo XML from spiral parameters
  - `uv run apps/spirob_metrics.py` - Run parameter sweep from `simulation_configs.json`
  - `uv run apps/spirob_analyse.py` - Compute metrics for all runs (use `--enable-position-estimation` for ACC+GYRO fusion)
  - `uv run apps/spirob_plot_gui.py` - Interactive GUI for plotting experiments
  - `uv run apps/spirob_digital_twin.py` - Hardware control interface (ESP32 serial communication)
- **Test**: `uv run pytest` (tests in `tests/`, focus on math functions, data schemas, contact forces)
- **Docs**: `cd docs && uv run make html` (Sphinx docs → `docs/_build/html/index.html`)
- **Video Recording**: `PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video` (headless rendering)

## Coding Conventions
- **Imports**: Relative imports within package (e.g., `from .math_spirob import rho`), absolute in apps (e.g., `import math_spirob.spirob_generator as sg`)
- **Data Handling**: 
  - Polars for columnar data (prefer LazyFrames for large datasets), NumPy for math arrays, Pydantic for validation
  - Always use Parquet format for sensor data storage (columnar, compressed, schema-enforced)
- **Sensor Naming Convention**: Follow `DataGroup` enum with dimension-aware suffixes:
  - 1D sensors: `tendon_frc_0`, `tendon_pos_0` (single column)
  - 3D sensors: `acc_0_X`, `acc_0_Y`, `acc_0_Z` (three columns with axis suffix)
  - Body contact forces: `body_spiral_0_contact_force_X/Y/Z` (auto-generated from MuJoCo contacts)
  - Position estimates: `pos_estimate_0_x/y/z`, `vel_estimate_0_x/y/z` (lowercase for estimated data)
- **Controller Interface**: Use `ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]`
  - Examples: `static_controller`, `ramped_controller`, `sine_controller` in `spirob_simulate.py`
  - Controllers modify `data.ctrl` array based on time/step index
- **Error Handling**: Raise descriptive exceptions (e.g., `FileNotFoundError` for missing experiment dirs)
- **Plotting**: Matplotlib/Seaborn in analysis scripts, save to `build/` directory

## Common Patterns
- **Spiral Generation**: 
  ```python
  xml_string = spirob_generator.generate_xml_string(L_target=0.3, base_d=0.05, tip_d=0.01, Delta_theta_deg=30)
  # Use SpiralCalculator internally to solve spiral parameters via bisect method
  ```
- **Simulation with Data Collection**:
  ```python
  df, record = spirob_simulate.run_simulation_with_data_collection(
      model, controller=ramped_controller, sim_time=2.0, 
      include_geom_pos=True, enable_position_estimation=False
  )
  # Returns Polars DataFrame with time series and ExperimentRecord with metadata
  ```
- **Experiment Loading**:
  ```python
  record, lf = analyzer.load_experiment("Run_001")  # Returns ExperimentRecord + LazyFrame
  df = lf.collect()  # Collect LazyFrame to DataFrame when needed
  ```
- **Config Parsing**: 
  - Read `simulation_configs.json` as list[dict], iterate with `itertools.product()` for parameter combinations
  - Validate each config with `ExperimentConfig(**config_dict)` before simulation
- **Data Storage**: Use `exporter.save_experiment(df, record)` to save Parquet + meta.json atomically
- **Body Contact Forces**: Automatically extracted in `extract_body_contact_forces()` using `mj.mj_contactForce()`
  - Transform from contact frame to world frame via rotation matrix
  - Forces applied/accumulated per body (+F on body1, -F on body2)
- **Plotting API**:
  ```python
  plots.plot_time_series(run_ids=["Run_001"], sensors=["acc_0"], axes=["X", "Y", "Z"])
  plots.plot_comparison(sensor="tendon_frc_0", axis=None, metric="mean", group_by="L_target")
  plots.quick_plot("acc_0", "X", "mean")  # Shortcut for common plots
  ```
- **Metadata Filtering**:
  ```python
  meta_df = plots.load_run_metadata()
  run_ids = plots.filter_runs(meta_df, {"controller_info": "Ramped", "L_target_min": 0.3})
  ```

## Integration Points
- **MuJoCo Physics Engine**:
  - XML models define worldbody, sensors, actuators (equality-constrained tendons)
  - Sensors registered via `SensorRegistry` in `spirob_generator.py` (acc, gyro, tendon_frc, etc.)
  - Contact forces extracted per-timestep via `mj.mj_contactForce()` and `contact.frame` rotation
- **External Hardware** (Digital Twin):
  - `spirob_digital_twin.py` provides GUI for ESP32 serial control (115200 baud, dual-motor actuation)
  - Commands sent as `SET M1:<force> M2:<force>\n` format
  - Uses CustomTkinter for UI, threading for serial I/O (50ms rate limit)
- **Data Persistence**:
  - Experiments stored as `build/experiments/Run_XXX/{meta.json, data.parquet}`
  - Summary CSV generated by `meta_analyzer.run_meta_analysis()` → `build/meta_analysis_summary.csv`
- **Visualization**:
  - Live plotting: PyQtGraph in `live_plotter.py` (real-time data streaming)
  - Static analysis: Matplotlib/Seaborn via `plots.py` module
  - GUI: Qt-based interactive filtering and plotting (`spirob_plot_gui.py`)
- **Position Estimation**: 
  - `SimpleSegmentEstimator` integrates ACC+GYRO to estimate 3D position/orientation per segment
  - Enable via `--enable-position-estimation` flag in `spirob_analyse.py`
  - Adds `pos_estimate_X_x/y/z`, `vel_estimate_X_x/y/z` columns to data.parquet

## Package Management & Dependencies
- **uv only**: All commands MUST use `uv run` prefix (no pip, virtualenv, or python -m)
- Key dependencies: mujoco>=3.3.7, polars>=1.35.2, pydantic>=2.0, matplotlib, seaborn, pyqtgraph, scipy
- Qt libraries: pyqt5/pyqt6/pyside6 for Linux GUI support (platform-conditional in pyproject.toml)
- Video export: imageio[ffmpeg] for simulation recording
- Hardware: pyserial, customtkinter for digital twin GUI

## Critical Files Reference
- `src/math_spirob/__init__.py`: Public API exports (import from here, not submodules)
- `src/math_spirob/data_schema.py`: Pydantic models (ExperimentConfig, ExperimentRecord, SensorMeta, DataGroup)
- `src/math_spirob/spirob_simulate.py`: Simulation loop, controller interface, contact force extraction
- `src/math_spirob/spirob_generator.py`: XML generation (SpiralCalculator, XMLBuilder, SensorRegistry)
- `apps/spirob_metrics.py`: Parameter sweep orchestration (VARIABLE_PARAMS, GEOM_SCENARIOS)
- `simulation_configs.json`: Batch experiment definitions (parsed by spirob_metrics.py)
- `README.md`: Setup guide, `pyproject.toml`: dependency manifest

## Development Tips
- Use `tree --gitignore` to inspect project structure
- Always validate configs with Pydantic before simulation to catch parameter errors early
- For large datasets, keep LazyFrames lazy until final computation (avoid premature `.collect()`)
- Check `build/meta_analysis_summary.csv` for quick overview of all experiment results
- Add new sensors via `SensorRegistry.register()` in spiral generation, update `DataGroup` enum
- Controllers are pure functions - use closures or classes (PIDController pattern) for stateful control