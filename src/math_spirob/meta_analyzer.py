import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
from .data_schema import ExperimentRecord, DataGroup
from .analyzer import load_experiment
from .simple_segment_estimator import SimpleSegmentEstimator

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

    # Compute contact force distribution metrics
    if body_contact_sensors:
        # Calculate force norms for each segment
        norm_cols = []
        for sensor in body_contact_sensors:
            x_col, y_col, z_col = sensor.columns
            norm_col = f"{sensor.name}_norm"
            lf = lf.with_columns(
                (pl.col(x_col).pow(2) + pl.col(y_col).pow(2) + pl.col(z_col).pow(2)).sqrt().alias(norm_col)
            )
            norm_cols.append(norm_col)
        
        # Total contact force per timestep
        total_force_col = "total_contact_force"
        lf = lf.with_columns(pl.sum_horizontal(norm_cols).alias(total_force_col))
        
        # Relative shares per segment, avoiding division by zero
        share_cols = []
        for sensor, norm_col in zip(body_contact_sensors, norm_cols):
            share_col = f"{sensor.name}_share"
            lf = lf.with_columns(
                pl.when(pl.col(total_force_col) > 0)
                .then(pl.col(norm_col) / pl.col(total_force_col))
                .otherwise(0.0)
                .alias(share_col)
            )
            share_cols.append(share_col)
        
        # Aggregate metrics: mean and std of shares, max of norms and shares
        for sensor, norm_col, share_col in zip(body_contact_sensors, norm_cols, share_cols):
            mean_share = lf.select(pl.col(share_col).mean()).collect().item()
            std_share = lf.select(pl.col(share_col).std()).collect().item()
            max_norm = lf.select(pl.col(norm_col).max()).collect().item()
            max_share = lf.select(pl.col(share_col).max()).collect().item()
            
            metrics[f"{sensor.name}_contact_share_mean"] = mean_share
            metrics[f"{sensor.name}_contact_share_std"] = std_share
            metrics[f"{sensor.name}_contact_force_max"] = max_norm
            metrics[f"{sensor.name}_contact_share_max"] = max_share
        
        # Find segment with highest peak force and highest peak share
        if body_contact_sensors:
            max_force_sensor = max(body_contact_sensors, key=lambda s: metrics[f"{s.name}_contact_force_max"])
            max_share_sensor = max(body_contact_sensors, key=lambda s: metrics[f"{s.name}_contact_share_max"])
            
            metrics["max_contact_force_segment"] = max_force_sensor.name
            metrics["max_contact_force_value"] = metrics[f"{max_force_sensor.name}_contact_force_max"]
            metrics["max_contact_share_segment"] = max_share_sensor.name
            metrics["max_contact_share_value"] = metrics[f"{max_share_sensor.name}_contact_share_max"]

    # Position estimation using ACC + GYRO
    enable_estimation = getattr(record.config, 'enable_position_estimation', False)
    if enable_estimation:
        metrics.update(_compute_position_metrics(record, lf))

    return metrics

def _compute_position_metrics(record: ExperimentRecord, lf: pl.LazyFrame) -> Dict[str, Any]:
    """
    Compute position estimation metrics using SimpleSegmentEstimator.
    """
    metrics = {}
    
    # Collect segment IDs from ACC sensors (assuming acc_0, acc_1, ... correspond to segments)
    acc_sensors = [s for s in record.sensors if s.group == DataGroup.ACC]
    gyro_sensors = [s for s in record.sensors if s.group == DataGroup.GYRO]
    
    if not acc_sensors or not gyro_sensors:
        return metrics
    
    # Extract segment IDs (assuming naming like acc_0, gyro_0, etc.)
    segment_ids = []
    for sensor in acc_sensors:
        try:
            seg_id = int(sensor.name.split('_')[1])
            segment_ids.append(seg_id)
        except (IndexError, ValueError):
            continue
    
    segment_ids = sorted(list(set(segment_ids)))
    
    if not segment_ids:
        return metrics
    
    # Initialize estimator
    initial_positions = (getattr(record.config, 'initial_positions', {}) if record.config else {}) or {}
    initial_orientations = (getattr(record.config, 'initial_orientations', {}) if record.config else {}) or {}
    
    estimator = SimpleSegmentEstimator(
        segment_ids=segment_ids,
        initial_positions=initial_positions,
        initial_orientations=initial_orientations
    )
    
    # Collect data
    df = lf.collect()
    time_col = 'time_s'
    if time_col not in df.columns:
        # Estimate dt from config
        dt = getattr(record.config, 'dt', 0.01)  # Default 100Hz
        time_steps = len(df)
        times = np.arange(0, time_steps * dt, dt)
        df = df.with_columns(pl.Series(time_col, times))
    
    # Process each timestep
    for row in df.iter_rows(named=True):
        sensor_data = {}
        for seg_id in segment_ids:
            acc_cols = [f'acc_{seg_id}_X', f'acc_{seg_id}_Y', f'acc_{seg_id}_Z']
            gyro_cols = [f'gyro_{seg_id}_X', f'gyro_{seg_id}_Y', f'gyro_{seg_id}_Z']
            
            if all(col in row for col in acc_cols + gyro_cols):
                sensor_data[seg_id] = {
                    'acc': [row[col] for col in acc_cols],
                    'gyro': [row[col] for col in gyro_cols]
                }
        
        if sensor_data:
            dt = row.get('dt', 0.01) if row is not None else 0.01  # Use dt column if available, else default
            estimator.update_batch(sensor_data, dt)
    
    # Extract final metrics
    all_states = estimator.get_all_states()
    
    max_drift = 0.0
    max_drift_seg = None
    
    for seg_id, state in all_states.items():
        pos = state.position
        vel = state.velocity
        quat = state.orientation
        
        # Position metrics
        metrics[f'seg_{seg_id}_pos_x'] = pos[0]
        metrics[f'seg_{seg_id}_pos_y'] = pos[1]
        metrics[f'seg_{seg_id}_pos_z'] = pos[2]
        
        # Orientation metrics
        metrics[f'seg_{seg_id}_quat_w'] = quat[0]
        metrics[f'seg_{seg_id}_quat_x'] = quat[1]
        metrics[f'seg_{seg_id}_quat_y'] = quat[2]
        metrics[f'seg_{seg_id}_quat_z'] = quat[3]
        
        # Velocity metrics
        vel_norm = np.linalg.norm(vel)
        metrics[f'seg_{seg_id}_vel_norm'] = vel_norm
        
        # Drift metric (vertical displacement from initial)
        drift = abs(pos[2])  # Assuming z is vertical
        metrics[f'seg_{seg_id}_drift_z'] = drift
        
        if drift > max_drift:
            max_drift = drift
            max_drift_seg = seg_id
    
    # Global metrics
    if segment_ids:
        tip_seg = max(segment_ids)  # Assume highest ID is tip
        tip_state = all_states[tip_seg]
        tip_pos = tip_state.position
        tip_quat = tip_state.orientation
        
        metrics['tip_position_x'] = tip_pos[0]
        metrics['tip_position_y'] = tip_pos[1]
        metrics['tip_position_z'] = tip_pos[2]
        
        metrics['tip_orientation_w'] = tip_quat[0]
        metrics['tip_orientation_x'] = tip_quat[1]
        metrics['tip_orientation_y'] = tip_quat[2]
        metrics['tip_orientation_z'] = tip_quat[3]
        
        if max_drift_seg is not None:
            metrics['max_drift_segment'] = max_drift_seg
            metrics['max_drift_value'] = max_drift
    
    return metrics

def aggregate_experiments(base_dir: str = "build") -> pl.DataFrame:
    """
    Aggregates data from all experiments into a summary DataFrame.
    Ensures unique run_ids by skipping duplicates.
    """
    run_ids = crawl_experiments(base_dir)
    print(f"Found {len(run_ids)} experiment directories")
    summary_data = []
    processed_run_ids = set()

    for run_id in run_ids:
        if run_id in processed_run_ids:
            print(f"Warning: Skipping duplicate run_id {run_id}")
            continue
        print(f"Processing run_id: {run_id}")
        try:
            record, lf = load_experiment(run_id, base_dir)
            metrics = compute_metrics(record, lf)

            # Extract parameters
            params = {
                "run_id": run_id,
                "L_target": getattr(record.config, 'L_target', None) if record.config else None,
                "base_d": getattr(record.config, 'base_d', None) if record.config else None,
                "sim_time": getattr(record.config, 'sim_time', None) if record.config else None,
                "controller_info": getattr(record.config, 'controller_info', None) if record.config else None,
                "geom_type": getattr(record.config, 'geom_type', None) if record.config else None,
            }
            params.update(metrics)
            summary_data.append(params)
            processed_run_ids.add(run_id)

        except Exception as e:
            print(f"Warning: Skipping experiment {run_id} due to error: {e}")
            continue

    print(f"Successfully processed {len(summary_data)} unique experiments")
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

def plot_force_distribution(run_id: str, base_dir: str = "build", figsize: tuple = (10, 6), save_path: Optional[str] = None):
    """
    Plots the mean contact force distribution across segments for a given run.
    Shows the average share of total contact force per segment.
    """
    df = load_summary_parquet(base_dir)
    run_data = df.filter(pl.col("run_id") == run_id)
    if run_data.is_empty():
        raise ValueError(f"Run {run_id} not found in summary.")
    
    # Extract share_mean columns
    share_cols = [col for col in df.columns if col.endswith("_contact_share_mean")]
    if not share_cols:
        print("No contact share metrics found.")
        return
    
    segments = [col.replace("_contact_share_mean", "") for col in share_cols]
    shares = [run_data.select(pl.col(col)).item() for col in share_cols]
    
    # Filter out None and zero shares for readability
    filtered = [(seg, sh) for seg, sh in zip(segments, shares) if sh is not None and sh > 0.001]
    if not filtered:
        print("No significant contact shares found.")
        return
    
    segments, shares = zip(*filtered)
    
    plt.figure(figsize=figsize)
    plt.bar(segments, shares)
    plt.xlabel("Segment")
    plt.ylabel("Mean Contact Force Share")
    plt.title(f"Mean Contact Force Distribution - {run_id}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_force_peak_usage(run_id: str, metric: str = "force", base_dir: str = "build", figsize: tuple = (10, 6), save_path: Optional[str] = None):
    """
    Plots the peak contact force usage across segments for a given run.
    metric: "force" for max norms, "share" for max shares.
    """
    df = load_summary_parquet(base_dir)
    run_data = df.filter(pl.col("run_id") == run_id)
    if run_data.is_empty():
        raise ValueError(f"Run {run_id} not found in summary.")
    
    suffix = "_contact_force_max" if metric == "force" else "_contact_share_max"
    peak_cols = [col for col in df.columns if col.endswith(suffix)]
    if not peak_cols:
        print(f"No {metric} peak metrics found.")
        return
    
    segments = [col.replace(suffix, "") for col in peak_cols]
    peaks = [run_data.select(pl.col(col)).item() for col in peak_cols]
    
    # Filter out None and zero peaks
    filtered = [(seg, pk) for seg, pk in zip(segments, peaks) if pk is not None and pk > 0.001]
    if not filtered:
        print("No significant peaks found.")
        return
    
    segments, peaks = zip(*filtered)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(segments, peaks)
    plt.xlabel("Segment")
    plt.ylabel(f"Max Contact {metric.title()}")
    plt.title(f"Peak Contact {metric.title()} Usage - {run_id}")
    plt.xticks(rotation=45, ha='right')
    
    # Highlight the max segment
    max_idx = peaks.index(max(peaks))
    bars[max_idx].set_color('red')
    plt.text(max_idx, peaks[max_idx], f"Max: {segments[max_idx]}", ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def run_meta_analysis(base_dir: str = "build", plot: bool = False, y_metrics: List[str] = None):
    """
    Runs the full meta-analysis: aggregate, save, and optionally plot.
    Checks for duplicate run_ids after aggregation.
    """
    df = aggregate_experiments(base_dir)
    
    if df.is_empty():
        print("No summary data to process")
        return df
    
    # Check for duplicate run_ids
    run_id_counts = df.group_by("run_id").len()
    duplicates = run_id_counts.filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        print("Error: Found duplicate run_ids in summary:")
        print(duplicates)
        raise ValueError("Duplicate run_ids detected. Please ensure unique run_ids.")
    
    save_summary(df, base_dir)
    if plot:
        if y_metrics is None:
            # Default to some metrics, e.g., tendonfrc means
            y_metrics = [col for col in df.columns if col.endswith('_mean') and 'tendonfrc' in col]
            if not y_metrics:
                y_metrics = ["mean_acc_x"]
        plot_trends(df, y_metrics=y_metrics)
    return df