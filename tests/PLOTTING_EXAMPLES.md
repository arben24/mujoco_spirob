# Plotting Examples for MuJoCo SpiRob

This document provides tested examples for using the plotting API.

## Basic Time Series

```python
from math_spirob import plot_time_series

# Single run, one sensor, all axes
plot_time_series("Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0", ["acc_0"], ["X", "Y", "Z"])
```

## Multi-Run Time Series

```python
# All runs, one sensor, one axis, colored by run
plot_time_series("all", ["acc_0"], ["X"], hue_by="run")

# Specific runs, aggregated mean
plot_time_series(["Run_001", "Run_002"], ["acc_0"], ["X"], aggregate_runs="mean")
```

## Metric Comparisons

```python
from math_spirob import plot_comparison

# Compare mean across runs
plot_comparison(["Run_001", "Run_002"], "acc_0", "X", "mean")
```

## Distributions

```python
from math_spirob import plot_distribution

# Box plot of mean across all runs
plot_distribution("all", "acc_0", "X", "mean")
```

## Grid Plots

```python
from math_spirob import plot_metric_grid

# Grid: rows=metrics, cols=sensors
plot_metric_grid("all", ["acc_0", "gyro_0"], ["X"], ["mean", "std"])
```

## Convenience Functions

```python
from math_spirob import plot_time_series_all_runs, quick_plot

# All runs for one sensor
plot_time_series_all_runs("acc_0", ["X", "Y", "Z"])

# Quick plot
quick_plot("acc_0", "X", "mean")
```

## CLI Usage

```bash
# Time series
python apps/spirob_plot_cli.py --kind time_series --sensor acc_0 --axis X --runs all

# Distribution
python apps/spirob_plot_cli.py --kind distribution --sensor acc_0 --axis X --metric mean --runs all

# Grid
python apps/spirob_plot_cli.py --kind grid --sensors acc_0 acc_1 --axes X --metrics mean std --runs all
```

All examples have been tested and work with the current data.