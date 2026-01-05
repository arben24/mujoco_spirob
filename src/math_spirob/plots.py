import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path
from .analyzer import load_experiment
from .meta_analyzer import load_summary_parquet, load_multiple_summaries_parquet

def load_experiment_parquet(run_id: str, base_dir: str = "build") -> pl.DataFrame:
    """
    Loads the raw sensor data for an experiment from Parquet.
    """
    _, lf = load_experiment(run_id, base_dir)
    return lf.collect()

def load_multiple_experiments_parquet(run_ids: List[str], base_dir: str = "build") -> pl.DataFrame:
    """
    Loads and concatenates raw sensor data for multiple experiments.
    Adds an 'experiment_id' column.
    """
    dfs = []
    for run_id in run_ids:
        df = load_experiment_parquet(run_id, base_dir)
        df = df.with_columns(pl.lit(run_id).alias("experiment_id"))
        dfs.append(df)
    return pl.concat(dfs)

def plot_time_series(run_id: str, sensors: List[str], axes: List[str], metric: str = "raw", base_dir: str = "build", figsize: tuple = (12, 8)):
    """
    Plots time series for specified sensors, axes, and metric.

    For metric="raw", plots raw sensor data over time.
    For aggregated metrics like "mean", "std", etc., plots constant values (not time series).

    Example:
        plot_time_series("Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0", sensors=["acc_0"], axes=["X", "Y", "Z"], metric="raw")
    """
    if metric == "raw":
        df = load_experiment_parquet(run_id, base_dir)
        plt.figure(figsize=figsize)
        for sensor in sensors:
            for axis in axes:
                col = f"{sensor}_{axis}"
                if col in df.columns:
                    plt.plot(df["time_s"], df[col], label=f"{sensor} {axis}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(f"Time Series: {run_id} - {', '.join(sensors)} {', '.join(axes)}")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # For aggregated metrics, load summary and plot as points
        df = load_summary_parquet(base_dir)
        df = df.filter(pl.col("run_id") == run_id)
        plt.figure(figsize=figsize)
        for sensor in sensors:
            for axis in axes:
                col = f"{sensor}_{axis}_{metric}"
                if col in df.columns:
                    value = df[col].item()
                    plt.scatter([0], [value], label=f"{sensor} {axis} {metric}")
        plt.title(f"Aggregated Metric: {run_id} - {metric}")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_comparison(run_ids: List[str], sensor: str, axis: str, metric: str, base_dir: str = "build", figsize: tuple = (10, 6)):
    """
    Compares a specific metric across multiple experiments.

    Example:
        plot_comparison(["Run_001", "Run_002"], sensor="acc_0", axis="X", metric="mean")
    """
    df = load_multiple_summaries_parquet(run_ids, base_dir)
    col = f"{sensor}_{axis}_{metric}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in summary data.")
    
    df_pd = df.to_pandas()
    plt.figure(figsize=figsize)
    sns.barplot(data=df_pd, x="experiment_id", y=col)
    plt.title(f"Comparison: {sensor} {axis} {metric}")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Experiment ID")
    plt.grid(True)
    plt.show()