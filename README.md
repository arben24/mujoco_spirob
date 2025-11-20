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
uv pip install -e .
```

## Running the Applications

After installation, you can execute the programs inside the apps/ directory. For example:

```bash
uv run apps/generate_2d_spirob.py
```
This will start the 2D generation script using the installed dependencies and environment managed by uv.

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


```bash
mujoco_spirob/
├── pyproject.toml
├── README.md
├── src/
│   └── math_spirob/
│       ├── __init__.py
│       ├── math_spirob.py
        └── spirob_generator.py
├── apps/
│   ├── generate_2d_spirob.py
│   └── controll_spirob_test.py
├── tests/
│   └── test_math_spirob.py
└── docs/
    ├── source/
    └── build/
```

## Notes

All dependencies, environments, and executions should be handled via uv for full reproducibility.

Code inside `src/` follows a clean, modular structure suitable for documentation with Sphinx and testing with Pytest.

Each application in `apps/` demonstrates a concrete use case of the math_spirob library.






