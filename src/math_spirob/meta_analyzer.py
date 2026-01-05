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
    """
    metrics = {}

    # Find tendon_frc sensors and compute min, max, mean for each
    tendon_sensors = [s for s in record.sensors if s.group == DataGroup.TENDON_FRC]
    for sensor in tendon_sensors:
        col = sensor.columns[0]  # Assuming dimension 1, so one column
        sensor_name = sensor.name
        min_val = lf.select(pl.col(col).min()).collect().item()
        max_val = lf.select(pl.col(col).max()).collect().item()
        mean_val = lf.select(pl.col(col).mean()).collect().item()
        metrics[f"{sensor_name}_min"] = min_val
        metrics[f"{sensor_name}_max"] = max_val
        metrics[f"{sensor_name}_mean"] = mean_val

    # Compute statistics for ACC sensors (x, y, z axes)
    acc_sensors = [s for s in record.sensors if s.group == DataGroup.ACC]
    for sensor in acc_sensors:
        for col in sensor.columns:
            axis = col.split('_')[-1]  # 'X', 'Y', 'Z'
            mean_val = lf.select(pl.col(col).mean()).collect().item()
            std_val = lf.select(pl.col(col).std()).collect().item()
            min_val = lf.select(pl.col(col).min()).collect().item()
            max_val = lf.select(pl.col(col).max()).collect().item()
            skew_val = lf.select(pl.col(col).skew()).collect().item()
            kurt_val = lf.select(pl.col(col).kurtosis()).collect().item()
            metrics[f"{sensor.name}_{axis}_mean"] = mean_val
            metrics[f"{sensor.name}_{axis}_std"] = std_val
            metrics[f"{sensor.name}_{axis}_min"] = min_val
            metrics[f"{sensor.name}_{axis}_max"] = max_val
            metrics[f"{sensor.name}_{axis}_skew"] = skew_val
            metrics[f"{sensor.name}_{axis}_kurtosis"] = kurt_val

    # Compute statistics for GYRO sensors (x, y, z axes)
    gyro_sensors = [s for s in record.sensors if s.group == DataGroup.GYRO]
    for sensor in gyro_sensors:
        for col in sensor.columns:
            axis = col.split('_')[-1]  # 'X', 'Y', 'Z'
            mean_val = lf.select(pl.col(col).mean()).collect().item()
            std_val = lf.select(pl.col(col).std()).collect().item()
            min_val = lf.select(pl.col(col).min()).collect().item()
            max_val = lf.select(pl.col(col).max()).collect().item()
            skew_val = lf.select(pl.col(col).skew()).collect().item()
            kurt_val = lf.select(pl.col(col).kurtosis()).collect().item()
            metrics[f"{sensor.name}_{axis}_mean"] = mean_val
            metrics[f"{sensor.name}_{axis}_std"] = std_val
            metrics[f"{sensor.name}_{axis}_min"] = min_val
            metrics[f"{sensor.name}_{axis}_max"] = max_val
            metrics[f"{sensor.name}_{axis}_skew"] = skew_val
            metrics[f"{sensor.name}_{axis}_kurtosis"] = kurt_val

    # Compute statistics for BODY_CONTACT_FRC sensors (x, y, z axes)
    body_contact_sensors = [s for s in record.sensors if s.group == DataGroup.BODY_CONTACT_FRC]
    for sensor in body_contact_sensors:
        for col in sensor.columns:
            axis = col.split('_')[-1]  # 'X', 'Y', 'Z'
            mean_val = lf.select(pl.col(col).mean()).collect().item()
            std_val = lf.select(pl.col(col).std()).collect().item()
            min_val = lf.select(pl.col(col).min()).collect().item()
            max_val = lf.select(pl.col(col).max()).collect().item()
            skew_val = lf.select(pl.col(col).skew()).collect().item()
            kurt_val = lf.select(pl.col(col).kurtosis()).collect().item()
            metrics[f"{sensor.name}_{axis}_mean"] = mean_val
            metrics[f"{sensor.name}_{axis}_std"] = std_val
            metrics[f"{sensor.name}_{axis}_min"] = min_val
            metrics[f"{sensor.name}_{axis}_max"] = max_val
            metrics[f"{sensor.name}_{axis}_skew"] = skew_val
            metrics[f"{sensor.name}_{axis}_kurtosis"] = kurt_val

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
    Saves the summary DataFrame to CSV and Parquet formats.
    Parquet is preferred for efficient storage and loading of columnar data with many columns.
    """
    output_path_csv = Path(base_dir) / "meta_analysis_summary.csv"
    output_path_parquet = Path(base_dir) / "meta_analysis_summary.parquet"
    df.write_csv(str(output_path_csv))
    df.write_parquet(str(output_path_parquet))
    print(f"Summary saved to {output_path_csv} and {output_path_parquet}")

def load_summary_parquet(base_dir: str = "build") -> pl.DataFrame:
    """
    Loads the summary DataFrame from Parquet format.
    """
    output_path_parquet = Path(base_dir) / "meta_analysis_summary.parquet"
    if not output_path_parquet.exists():
        raise FileNotFoundError(f"Summary Parquet file not found: {output_path_parquet}")
    return pl.read_parquet(str(output_path_parquet))

def load_multiple_summaries_parquet(experiment_ids: List[str], base_dir: str = "build") -> pl.DataFrame:
    """
    Loads and concatenates summary DataFrames for multiple experiments from Parquet.
    Adds an 'experiment_id' column for identification.
    """
    dfs = []
    for exp_id in experiment_ids:
        try:
            df = load_summary_parquet(base_dir)
            df = df.filter(pl.col("run_id") == exp_id).with_columns(pl.lit(exp_id).alias("experiment_id"))
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Summary for {exp_id} not found.")
    if not dfs:
        raise ValueError("No summaries found for the given experiment IDs.")
    return pl.concat(dfs)

def plot_trends(df: pl.DataFrame, x_param: str = "L_target", y_metrics: List[str] = ["mean_acc_x"], title: str = None, figsize: tuple = (10, 6), hue: str = None, annotate_runs: bool = False):
    """
    Plots trends using matplotlib/seaborn. Supports multiple y_metrics.
    If multiple y_metrics, uses subplots or a melted plot.
    Set annotate_runs=True to label points with run_id.
    """
    df_pd = df.to_pandas()
    
    if len(y_metrics) == 1:
        plt.figure(figsize=figsize)
        ax = sns.scatterplot(data=df_pd, x=x_param, y=y_metrics[0], hue=hue)
        plt.title(title or f"Trend: {y_metrics[0]} vs {x_param}")
        plt.xlabel(x_param)
        plt.ylabel(y_metrics[0])
        plt.grid(True)
        if annotate_runs:
            for _, row in df_pd.iterrows():
                ax.text(row[x_param], row[y_metrics[0]], row['run_id'], fontsize=8, ha='right')
        plt.show()
    else:
        # For multiple y_metrics, melt the dataframe and plot
        melted = df_pd.melt(id_vars=[x_param, hue, 'run_id'] if hue else [x_param, 'run_id'], value_vars=y_metrics, var_name='Metric', value_name='Value')
        plt.figure(figsize=figsize)
        ax = sns.scatterplot(data=melted, x=x_param, y='Value', hue='Metric', style=hue)
        plt.title(title or f"Trends vs {x_param}")
        plt.xlabel(x_param)
        plt.ylabel("Value")
        plt.grid(True)
        if annotate_runs:
            # For multiple metrics, annotate might be cluttered, but possible
            for _, row in melted.iterrows():
                ax.text(row[x_param], row['Value'], row['run_id'], fontsize=6, ha='right')
        plt.show()

def run_meta_analysis(base_dir: str = "build", plot: bool = False, y_metrics: List[str] = None):
    """
    Runs the full meta-analysis: aggregate, save, and optionally plot.
    """
    df = aggregate_experiments(base_dir)
    save_summary(df, base_dir)
    if plot:
        if y_metrics is None:
            # Default to some metrics, e.g., tendonfrc means
            y_metrics = [col for col in df.columns if col.endswith('_mean') and 'tendonfrc' in col]
            if not y_metrics:
                y_metrics = ["mean_acc_x"]
        plot_trends(df, y_metrics=y_metrics)
    return df