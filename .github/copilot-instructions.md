# Mujoco Spirob - AI Agent Instructions

## Project Overview
This codebase implements mathematical and simulation tools for logarithmic spiral structures (SpiRob) using MuJoCo physics engine. Core components:
- `src/math_spirob/`: Pure math functions, MuJoCo XML generation, simulation runners, and experiment analysis
- `apps/`: Executable scripts for generation, control, and analysis workflows
- `build/experiments/`: Simulation run outputs with sensor data and metadata

## Key Architecture Patterns
- **Modular Design**: Library (`src/`) separated from applications (`apps/`), enabling reusable components
- **Data Flow**: Config (JSON) → XML generation → MuJoCo simulation → Sensor data (Polars DF) → Metrics analysis
- **Schema-Driven**: Use Pydantic models (`data_schema.py`) for experiment configs and sensor metadata
- **Lazy Evaluation**: Polars LazyFrames for efficient data processing in analysis (`meta_analyzer.py`)

## Essential Workflows
- **Setup**: `uv venv && uv pip install -e .` (uv manages dependencies and virtual env)
- **Run Apps**: `uv run apps/<script>.py` (e.g., `uv run apps/generate_2d_spirob.py`)
- **Test**: `uv run pytest` (tests in `tests/`, focus on math functions and data schemas)
- **Docs**: `cd docs && uv run make html` (Sphinx docs in `docs/_build/html/`)
- **GUI Analysis**: `uv run apps/spirob_plot_gui.py` for interactive plotting
- **Video Recording**: `PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video` for simulation videos

## Coding Conventions
- **Imports**: Absolute imports within package (e.g., `from .math_spirob import rho`)
- **Data Handling**: Polars for columnar data, NumPy for math arrays, Pydantic for validation
- **Sensor Naming**: Follow `DataGroup` enum (ACC, GYRO, TENDON_FRC, etc.) with dimension-aware columns like `acc_0_X`, `body_spiral_0_contact_force_Y`
- **Error Handling**: Raise descriptive exceptions (e.g., `FileNotFoundError` for missing experiment dirs)
- **Plotting**: Matplotlib/Seaborn in analysis scripts, save to `build/` directory

## Common Patterns
- **Spiral Generation**: Use `spirob_generator.generate_xml_string(L_target, base_d, tip_d)` for MuJoCo models
- **Simulation**: `spirob_simulate.run_simulation_with_data_collection()` with controller callbacks
- **Analysis**: Load experiments via `analyzer.load_experiment()`, compute metrics with LazyFrames
- **Config Parsing**: Read `simulation_configs.json` as list of dicts, validate with `ExperimentConfig`
- **Data Storage**: Use Parquet for efficient storage of sensor data and summaries (columnar, compressed)
- **Plotting API**: Use `plots.plot_time_series()` and `plots.plot_comparison()` for high-level visualizations
- **Body Contact Forces**: Automatically extracted and stored as `body_<name>_contact_force_X/Y/Z` sensors

## Integration Points
- **MuJoCo**: Physics simulation with XML models; sensors defined in XML, data collected via `mj.MjData`
- **External Data**: Experiments stored in `build/experiments/Run_XXX/` with `meta.json` and Parquet files
- **Visualization**: Live plotting with PyQtGraph (`live_plotter.py`), static plots with Matplotlib

Reference: `README.md` for setup, `pyproject.toml` for dependencies, `src/math_spirob/__init__.py` for public API.