"""
Plotting utilities for MuJoCo SpiRob experiments.

Examples:
    # Time series for one run
    plot_time_series("Run_001", ["acc_0"], ["X", "Y", "Z"])

    # Time series for all runs, colored by run
    plot_time_series("all", ["acc_0"], ["X"], hue_by="run")

    # Aggregated time series across runs
    plot_time_series(["Run_001", "Run_002"], ["acc_0"], ["X"], aggregate_runs="mean")

    # Metric grid
    plot_metric_grid("all", ["acc_0", "acc_1"], ["X"], ["mean", "std"])

    # Distribution
    plot_distribution("all", "acc_0", "X", "mean")

    # Quick plot
    quick_plot("acc_0", "X", "mean")
"""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union, Literal, Dict, Any
from pathlib import Path
from .analyzer import load_experiment
from .meta_analyzer import load_summary_parquet, load_multiple_summaries_parquet
import json

def list_available_runs(base_dir: str = "build") -> List[str]:
    """
    Lists all available run IDs from the summary parquet file.
    """
    try:
        df = load_summary_parquet(base_dir)
        return sorted(df["run_id"].unique().to_list())
    except Exception:
        # Fallback: list directories in build/experiments
        experiments_dir = Path(base_dir) / "experiments"
        if experiments_dir.exists():
            return sorted([d.name for d in experiments_dir.iterdir() if d.is_dir()])
        return []

def load_run_metadata(base_dir: str = "build") -> pl.DataFrame:
    """
    Loads metadata for all runs from meta.json files into a Polars DataFrame.
    
    Columns include: run_id, timestamp, L_target, base_d, tip_d, Delta_theta_deg, 
    sim_time, controller_info, geom_type, geom_params, include_geom_pos, version.
    """
    experiments_dir = Path(base_dir) / "experiments"
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    metadata = []
    for run_dir in experiments_dir.iterdir():
        if run_dir.is_dir():
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                config = data.get("config", {})
                record = {
                    "run_id": data.get("run_id"),
                    "timestamp": data.get("timestamp"),
                    "L_target": config.get("L_target"),
                    "base_d": config.get("base_d"),
                    "tip_d": config.get("tip_d"),
                    "Delta_theta_deg": config.get("Delta_theta_deg"),
                    "sim_time": config.get("sim_time"),
                    "controller_info": config.get("controller_info"),
                    "geom_type": config.get("geom_type"),
                    "geom_params": str(config.get("geom_params")),  # JSON string for simplicity
                    "include_geom_pos": config.get("include_geom_pos"),
                    "version": data.get("version"),
                }
                metadata.append(record)
    
    return pl.DataFrame(metadata)

def list_available_meta_fields(base_dir: str = "build") -> List[str]:
    """
    Lists all available metadata fields from the run metadata.
    """
    df = load_run_metadata(base_dir)
    return df.columns

def filter_runs(meta_df: pl.DataFrame, filters: Dict[str, Any]) -> List[str]:
    """
    Filters runs based on metadata conditions.
    
    Supports:
    - Equality: {"controller_info": "PID"}
    - Ranges: {"L_target_min": 0.2, "L_target_max": 0.4}
    
    Returns list of run_ids that match all conditions.
    """
    filtered_df = meta_df
    
    for key, value in filters.items():
        if key.endswith("_min"):
            col = key[:-4]
            filtered_df = filtered_df.filter(pl.col(col) >= value)
        elif key.endswith("_max"):
            col = key[:-4]
            filtered_df = filtered_df.filter(pl.col(col) <= value)
        else:
            filtered_df = filtered_df.filter(pl.col(key) == value)
    
    run_ids = filtered_df["run_id"].to_list()
    if not run_ids:
        raise ValueError(f"No runs found matching filters: {filters}")
    return run_ids

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
    Only includes columns present in all experiments.
    """
    dfs = []
    
    # First pass: collect all column sets
    column_sets = []
    for run_id in run_ids:
        df = load_experiment_parquet(run_id, base_dir)
        column_sets.append(set(df.columns))
    
    # Find common columns
    common_columns = set.intersection(*column_sets) if column_sets else set()
    
    # Second pass: load and filter to common columns
    for run_id in run_ids:
        df = load_experiment_parquet(run_id, base_dir)
        df = df.select(list(common_columns))
        df = df.with_columns(pl.lit(run_id).alias("experiment_id"))
        dfs.append(df)
    
    return pl.concat(dfs)

def plot_time_series(run_ids: Union[str, List[str], None] = None, sensors: List[str] = None, axes: List[str] = None, metric: str = "raw", 
                     base_dir: str = "build", figsize: tuple = (12, 8), hue_by: Optional[str] = None, 
                     facet_by: Optional[str] = None, aggregate_runs: Optional[str] = None, 
                     save_path: Optional[str] = None, yscale: str = "linear", filters: Optional[Dict[str, Any]] = None):
    """
    Plots time series for specified sensors, axes, and metric across multiple runs.

    Args:
        run_ids: Single run ID, list of run IDs, or "all" for all available runs. If None and filters provided, runs are filtered.
        sensors: List of sensor names.
        axes: List of axes (e.g., ["X", "Y", "Z"]).
        metric: "raw" for time series, or aggregated metric like "mean".
        hue_by: If "run", color by run in single plot.
        facet_by: If "run", create subplots per run.
        aggregate_runs: Aggregate across runs, e.g., "mean", "median".
        save_path: Path to save plot, if None, shows plot.
        yscale: "linear" or "log".
        filters: Dict of metadata filters to select runs.

    Examples:
        plot_time_series("Run_001", sensors=["acc_0"], axes=["X", "Y", "Z"])
        plot_time_series("all", sensors=["acc_0"], axes=["X"], hue_by="run")
        plot_time_series(filters={"controller_info": "PID"}, sensors=["acc_0"], axes=["X"])
    """
    if filters and run_ids is None:
        meta_df = load_run_metadata(base_dir)
        run_ids = filter_runs(meta_df, filters)
    elif run_ids == "all":
        run_ids = list_available_runs(base_dir)
    elif isinstance(run_ids, str):
        run_ids = [run_ids]
    
    if not run_ids:
        raise ValueError("No run_ids specified or found via filters.")
    
    if metric == "raw":
        df = load_multiple_experiments_parquet(run_ids, base_dir)
        df_pd = df.to_pandas()
        
        if facet_by == "run":
            g = sns.FacetGrid(df_pd, col="experiment_id", col_wrap=1, height=figsize[1], aspect=figsize[0]/figsize[1])
            g.map_dataframe(sns.lineplot, x="time_s", y=None, hue=None)
            for sensor in sensors:
                for axis in axes:
                    col = f"{sensor}_{axis}"
                    if col in df.columns:
                        plt.plot(df_pd["time_s"], df_pd[col], label=f"{sensor} {axis}")
            # Need to adjust for multiple lines
            # This is simplified; for proper faceting, need to melt or something
        elif hue_by == "run":
            plt.figure(figsize=figsize)
            for sensor in sensors:
                for axis in axes:
                    col = f"{sensor}_{axis}"
                    if col not in df.columns:
                        col = sensor
                    if col in df.columns:
                        sns.lineplot(data=df_pd, x="time_s", y=col, hue="experiment_id", label=f"{sensor} {axis}")
        else:
            plt.figure(figsize=figsize)
            for run_id in run_ids:
                df_run = df.filter(pl.col("experiment_id") == run_id)
                df_pd_run = df_run.to_pandas()
                for sensor in sensors:
                    for axis in axes:
                        col = f"{sensor}_{axis}"
                        if col not in df.columns:
                            col = sensor
                        if col in df.columns:
                            plt.plot(df_pd_run["time_s"], df_pd_run[col], label=f"{run_id} {sensor} {axis}")
        
        if aggregate_runs:
            # Aggregate across runs
            agg_df = df.group_by("time_s").agg([pl.col(f"{s}_{a}").mean().alias(f"{s}_{a}_mean") for s in sensors for a in axes])
            agg_pd = agg_df.to_pandas()
            for sensor in sensors:
                for axis in axes:
                    col = f"{sensor}_{axis}_mean"
                    plt.plot(agg_pd["time_s"], agg_pd[col], label=f"Aggregated {sensor} {axis}", linewidth=3)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(f"Time Series: {', '.join(run_ids)} - {', '.join(sensors)} {', '.join(axes)}")
        plt.legend()
        plt.grid(True)
        plt.yscale(yscale)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    else:
        # For aggregated metrics, similar to before but for multiple runs
        df = load_multiple_summaries_parquet(run_ids, base_dir)
        df_pd = df.to_pandas()
        plt.figure(figsize=figsize)
        for sensor in sensors:
            for axis in axes:
                col = f"{sensor}_{axis}_{metric}"
                if col not in df.columns:
                    col = f"{sensor}_{metric}"
                if col in df.columns:
                    if hue_by == "run":
                        sns.barplot(data=df_pd, x="experiment_id", y=col, hue="experiment_id")
                    else:
                        sns.barplot(data=df_pd, x="experiment_id", y=col)
        plt.title(f"Aggregated Metric: {', '.join(run_ids)} - {metric}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Experiment ID")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def plot_time_series_all_runs(sensor: str, axes: List[str], metric: str = "raw", base_dir: str = "build", 
                              figsize: tuple = (12, 8), hue_by: str = "run", **kwargs):
    """
    Plots time series for a sensor across all available runs.

    Args:
        sensor: Sensor name.
        axes: List of axes.
        metric: "raw" or aggregated.
        hue_by: How to distinguish runs, default "run".
        **kwargs: Passed to plot_time_series.

    Example:
        plot_time_series_all_runs("acc_0", ["X", "Y", "Z"])
    """
    plot_time_series("all", [sensor], axes, metric, base_dir, figsize, hue_by=hue_by, **kwargs)

def plot_metric_grid(run_ids: Union[str, List[str], Literal["all"]], sensors: List[str], axes: List[str], 
                     metrics: List[str], base_dir: str = "build", figsize: tuple = (12, 8), 
                     save_path: Optional[str] = None):
    """
    Creates a grid of subplots for metrics across sensors and runs.

    Rows: metrics, Columns: sensors, Colors: runs.

    Example:
        plot_metric_grid("all", ["acc_0"], ["X"], ["mean", "std"])
    """
    if run_ids == "all":
        run_ids = list_available_runs(base_dir)
    elif isinstance(run_ids, str):
        run_ids = [run_ids]
    
    df = load_multiple_summaries_parquet(run_ids, base_dir)
    df_pd = df.to_pandas()
    
    fig, axes_arr = plt.subplots(len(metrics), len(sensors), figsize=figsize, sharex=True)
    if len(metrics) == 1 and len(sensors) == 1:
        axes_arr = [[axes_arr]]
    elif len(metrics) == 1:
        axes_arr = [axes_arr]
    elif len(sensors) == 1:
        axes_arr = [[ax] for ax in axes_arr]
    
    for i, metric in enumerate(metrics):
        for j, sensor in enumerate(sensors):
            ax = axes_arr[i][j]
            for axis in axes:
                col = f"{sensor}_{axis}_{metric}"
                if col in df.columns:
                    sns.barplot(data=df_pd, x="experiment_id", y=col, ax=ax, label=f"{axis}")
            ax.set_title(f"{sensor} {metric}")
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_distribution(run_ids: Union[str, List[str], Literal["all"]], sensor: str, axis: str, metric: str, 
                      base_dir: str = "build", figsize: tuple = (10, 6), kind: str = "box", 
                      save_path: Optional[str] = None):
    """
    Plots distribution of a metric across runs.

    Args:
        run_ids: Runs to include.
        sensor: Sensor name.
        axis: Axis.
        metric: Metric like "mean".
        kind: "box" or "violin".

    Example:
        plot_distribution("all", "acc_0", "X", "mean")
    """
    if run_ids == "all":
        run_ids = list_available_runs(base_dir)
    elif isinstance(run_ids, str):
        run_ids = [run_ids]
    
    df = load_multiple_summaries_parquet(run_ids, base_dir)
    col = f"{sensor}_{axis}_{metric}"
    if col not in df.columns:
        col = f"{sensor}_{metric}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found.")
    
    df_pd = df.to_pandas()
    plt.figure(figsize=figsize)
    if kind == "box":
        sns.boxplot(data=df_pd, x="experiment_id", y=col)
    elif kind == "violin":
        sns.violinplot(data=df_pd, x="experiment_id", y=col)
    plt.title(f"Distribution: {sensor} {axis} {metric}")
    plt.ylabel(f"{metric.capitalize()}")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def quick_plot(sensor: str, axis: str, metric: str, run_ids: Union[str, List[str], Literal["all"]] = "all", 
               base_dir: str = "build", **kwargs):
    """
    Quick plot for a sensor, axis, metric across runs.

    Example:
        quick_plot("acc_0", "X", "mean")
    """
    if metric == "raw":
        plot_time_series(run_ids, [sensor], [axis], metric, base_dir, **kwargs)
    else:
        plot_distribution(run_ids, sensor, axis, metric, base_dir, **kwargs)

def plot_comparison(run_ids: Union[str, List[str], Literal["all"], None] = None, sensor: str = None, axis: str = None, metric: str = None, base_dir: str = "build", figsize: tuple = (10, 6), save_path: Optional[str] = None, filters: Optional[Dict[str, Any]] = None):
    """
    Compares a specific metric across multiple experiments.

    Args:
        run_ids: List of run IDs, "all", or None if filters provided.
        sensor: Sensor name.
        axis: Axis.
        metric: Metric like "mean".
        base_dir: Base directory.
        figsize: Figure size.
        save_path: Path to save plot.
        filters: Dict of metadata filters.

    Example:
        plot_comparison(["Run_001", "Run_002"], sensor="acc_0", axis="X", metric="mean")
        plot_comparison(filters={"controller_info": "PID"}, sensor="acc_0", axis="X", metric="mean")
    """
    if filters and run_ids is None:
        meta_df = load_run_metadata(base_dir)
        run_ids = filter_runs(meta_df, filters)
    elif run_ids == "all":
        run_ids = list_available_runs(base_dir)
    elif isinstance(run_ids, str):
        run_ids = [run_ids]
    
    if not run_ids:
        raise ValueError("No run_ids specified or found via filters.")
    
    # Rest remains the same
    if run_ids == "all":
        run_ids = list_available_runs(base_dir)
    elif isinstance(run_ids, str):
        run_ids = [run_ids]
    
    df = load_multiple_summaries_parquet(run_ids, base_dir)
    col = f"{sensor}_{axis}_{metric}"
    if col not in df.columns:
        col = f"{sensor}_{metric}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in summary data.")
    
    df_pd = df.to_pandas()
    plt.figure(figsize=figsize)
    sns.barplot(data=df_pd, x="experiment_id", y=col)
    plt.title(f"Comparison: {sensor} {axis} {metric}")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Experiment ID")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_time_series_filtered(filters: Dict[str, Any], sensors: List[str], axes: List[str], metric: str = "raw", 
                              base_dir: str = "build", **kwargs):
    """
    Plots time series for runs matching the filters.

    Args:
        filters: Dict of metadata filters.
        sensors: List of sensor names.
        axes: List of axes.
        metric: "raw" or aggregated.
        **kwargs: Passed to plot_time_series.

    Example:
        plot_time_series_filtered({"controller_info": "PID"}, ["acc_0"], ["X"])
    """
    plot_time_series(run_ids=None, sensors=sensors, axes=axes, metric=metric, base_dir=base_dir, filters=filters, **kwargs)

def plot_comparison_filtered(filters: Dict[str, Any], sensor: str, axis: str, metric: str, 
                             base_dir: str = "build", **kwargs):
    """
    Compares a metric for runs matching the filters.

    Args:
        filters: Dict of metadata filters.
        sensor: Sensor name.
        axis: Axis.
        metric: Metric.
        **kwargs: Passed to plot_comparison.

    Example:
        plot_comparison_filtered({"geom_type": "cylinder"}, "acc_0", "X", "mean")
    """
    plot_comparison(run_ids=None, sensor=sensor, axis=axis, metric=metric, base_dir=base_dir, filters=filters, **kwargs)