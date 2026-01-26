# Mujoco Spirob

This project provides mathematical and control utilities for working with logarithmic spiral structures (SpiRob). It includes a modular architecture with:

- A **library** (`src/math_spirob`) containing reusable mathematical functions and data classes.
- Multiple **applications** in the `apps/` directory demonstrating how to use the library.
- **Unit tests** in the `tests/` folder.
- **Documentation** powered by Sphinx in the `docs/` folder.

---

## Project Setup

All project dependencies are managed with **[uv](https://github.com/astral-sh/uv)** — a fast Python package and environment manager.

### Install the project

From the project root directory, run:

```bash
uv venv
uv pip install -e .
```

## Running the Applications

After installation, you can execute the programs inside the apps/ directory. For example:

```bash
uv run apps/generate_2d_spirob.py
```
This will start the 2D generation script using the installed dependencies and environment managed by uv.

## Running the GUI

The project includes a graphical user interface for plotting and analyzing experiments:

```bash
uv run apps/spirob_plot_gui.py
```

This GUI allows you to:
- Select and filter experiment runs
- Choose sensors, axes, and metrics
- Generate various plots (time series, comparisons, force distributions) interactively

## Running Tests

Tests are located in the `tests/` folder. You can run them using:

```bash
uv run pytest
```

## Building the Documentation

Documentation is written using Sphinx. To build it:

```bash
cd docs
uv run make html
```
The generated HTML files will be available in docs/build/html. There you can open the `index.html` with the browser of you choice. 

## Project Structure

You can generate the project structure yourself using `tree`.  
Run the following command:

```bash
tree --gitignore
```

```bash
mujoco_spirob/
├── apps
│   ├── controll_spirob_test.py
│   ├── generate_2d_spirob.py
│   ├── __init__.py
│   ├── live_plotter.py
│   ├── spirob_analyse.py
│   ├── spirob_metrics.py
│   ├── spirob_mujoco_plot.py
│   ├── spirob_plot_cli.py
│   ├── spirob_plot_gui.py
│   └── test_plots.py
├── docs
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   └── spirob_docs
│       ├── math_spirob.rst
│       └── spirob_generator.rst
├── pyproject.toml
├── README.md
├── src
│   └── math_spirob
│       ├── analyzer.py
│       ├── data_schema.py
│       ├── exporter.py
│       ├── __init__.py
│       ├── math_spirob.py
│       ├── meta_analyzer.py
│       ├── plots.py
│       ├── spirob_generator.py
│       └── spirob_simulate.py
```

## Notes

All dependencies, environments, and executions should be handled via uv for full reproducibility.

Code inside `src/` follows a clean, modular structure suitable for documentation with Sphinx and testing with Pytest.

## Recording Simulation Videos

The `spirob_metrics.py` application supports recording videos of MuJoCo simulations for each run:

```bash
PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video
```

This will:
- Record a video for each simulation run
- Save videos as `video.mp4` in each run's directory (e.g., `build/experiments/Run_001_.../video.mp4`)
- Use default settings: 1280x720 resolution at 30 FPS
- Videos show the complete simulation from a fixed camera perspective
- Video length matches simulation time (real-time playback)

**Video Configuration Options:**

```bash
# Custom resolution and FPS
PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video --video-resolution 1920x1080 --video-fps 60

# High-quality recording
PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video --video-resolution 1920x1080 --video-fps 30

# Low-quality for faster processing
PYOPENGL_PLATFORM=egl uv run apps/spirob_metrics.py --record-video --video-resolution 640x480 --video-fps 24
```

**Command-line Arguments:**
- `--record-video`: Enable video recording for all simulation runs
- `--video-resolution WIDTHxHEIGHT`: Set video resolution (default: 1280x720)
- `--video-fps FPS`: Set video frame rate (default: 30)

**System Requirements for Video Recording:**
- EGL-compatible graphics drivers (most modern GPUs support this)
- Set environment variable `PYOPENGL_PLATFORM=egl` before running
- In headless environments (no display), EGL must be available

**Video Technical Details:**
- Frames are rendered at the specified FPS by subsampling simulation steps
- Images are automatically corrected for orientation (MuJoCo renders bottom-up, videos play top-down)
- Camera FOV is adjusted to maintain consistent horizontal field of view across different aspect ratios
- Uses imageio with ffmpeg backend for MP4 encoding (macro_block_size=1 to avoid codec warnings)
- Supports resolutions from 640x480 to 1920x1080 and higher

If EGL is not available, video recording will be skipped with a warning, but simulations will continue normally.

Each application in `apps/` demonstrates a concrete use case of the math_spirob library.


## Analyse

To start the analyse which calulates the different metrics run

```bash
uv run ./apps/spirob_analyse.py
```

```bash
uv run ./apps/spirob_analyse.py --enable-position-estimation
```





