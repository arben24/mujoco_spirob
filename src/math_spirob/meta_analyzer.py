import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from .data_schema import ExperimentRecord, DataGroup
from .analyzer import load_experiment

def crawl_experiments(base_dir: str = "build") -> List[str]:
    """
    Crawls the experiments directory and returns a list of run_ids.
    """
    experiments_path = Path(base_dir) / "experiments"
    if not experiments_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_path}")
    return [d.name for d in experiments_path.iterdir() if d.is_dir()]

def compute_metrics(record: ExperimentRecord, lf: pl.LazyFrame) -> Dict[str, Any]:
    """
    Computes metrics for an experiment using lazy Polars operations.
    Example: Max tendon force across all tendon_frc sensors.
    """
    metrics = {}

    # Find tendon_frc sensors
    tendon_sensors = [s for s in record.sensors if s.group == DataGroup.TENDON_FRC]
    if tendon_sensors:
        # Collect all columns for tendon_frc
        tendon_columns = []
        for sensor in tendon_sensors:
            tendon_columns.extend(sensor.columns)
        # Compute max across all tendon force columns
        max_tendon_frc = lf.select(pl.max_horizontal(tendon_columns)).collect().item()
        metrics["max_tendon_frc"] = max_tendon_frc

    # Add more metrics as needed, e.g., mean acc magnitude
    acc_sensors = [s for s in record.sensors if s.group == DataGroup.ACC]
    if acc_sensors:
        acc_columns = []
        for sensor in acc_sensors:
            acc_columns.extend(sensor.columns)
        # Compute mean of magnitudes (sqrt(x^2 + y^2 + z^2)) but simplified: mean of x for now
        mean_acc_x = lf.select(pl.col(acc_columns[0]).mean()).collect().item()  # Assuming first acc sensor x
        metrics["mean_acc_x"] = mean_acc_x

    return metrics

def aggregate_experiments(base_dir: str = "build") -> pl.DataFrame:
    """
    Aggregates data from all experiments into a summary DataFrame.
    """
    run_ids = crawl_experiments(base_dir)
    summary_data = []

    for run_id in run_ids:
        try:
            record, lf = load_experiment(run_id, base_dir)
            metrics = compute_metrics(record, lf)

            # Extract parameters
            params = {
                "run_id": run_id,
                "L_target": record.config.L_target,
                "base_d": record.config.base_d,
                "sim_time": record.config.sim_time,
                "controller_info": record.config.controller_info,
                "geom_type": record.config.geom_type,
            }
            params.update(metrics)
            summary_data.append(params)

        except Exception as e:
            print(f"Warning: Skipping experiment {run_id} due to error: {e}")
            continue

    return pl.DataFrame(summary_data)

def save_summary(df: pl.DataFrame, base_dir: str = "build"):
    """
    Saves the summary DataFrame to CSV.
    """
    output_path = Path(base_dir) / "meta_analysis_summary.csv"
    df.write_csv(str(output_path))
    print(f"Summary saved to {output_path}")

def plot_trends(df: pl.DataFrame, x_param: str = "L_target", y_metric: str = "mean_acc_x"):
    """
    Plots a trend using matplotlib/seaborn.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.to_pandas(), x=x_param, y=y_metric)
    plt.title(f"Trend: {y_metric} vs {x_param}")
    plt.xlabel(x_param)
    plt.ylabel(y_metric)
    plt.grid(True)
    plt.show()

def run_meta_analysis(base_dir: str = "build", plot: bool = True):
    """
    Runs the full meta-analysis: aggregate, save, and optionally plot.
    """
    df = aggregate_experiments(base_dir)
    save_summary(df, base_dir)
    if plot:
        plot_trends(df)
    return df