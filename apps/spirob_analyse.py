import argparse
from pathlib import Path
import polars as pl

import math_spirob.meta_analyzer as ma
from math_spirob.analyzer import load_experiment
from math_spirob.simple_segment_estimator import SimpleSegmentEstimator
import math_spirob.exporter as exp


def run_analysis(enable_position_estimation: bool = False, position_estimator_segments: list = None):
    run_ids = ma.crawl_experiments()
    for run_id in run_ids:
        print(f"Processing {run_id}")
        record, lf = load_experiment(run_id)

        if record.config is None:
            print(f"Warning: record.config is None for {run_id}, skipping")
            continue

        # Apply CLI-configured estimation flags into record.config
        try:
            record.config.enable_position_estimation = True
            if position_estimator_segments:
                record.config.position_estimator_segments = position_estimator_segments
        except AttributeError:
            print(f"Warning: record.config is None for {run_id}, skipping position estimation")
            continue

        # If enabled, perform per-timestep estimation and update data.parquet
        if getattr(record.config, 'enable_position_estimation', False):
            print(f"  Running position estimation for {run_id}")
            # Load full dataframe
            base_path = Path("build") / "experiments" / run_id
            data_path = base_path / "data.parquet"
            df = pl.read_parquet(str(data_path))

            # Determine segment ids
            segs = getattr(record.config, 'position_estimator_segments', None)
            if not segs:
                # derive from ACC sensors
                segs = []
                for s in record.sensors:
                    if s.group.value == 'accelerometer' and s.name.startswith('acc_'):
                        try:
                            segs.append(int(s.name.split('_')[1]))
                        except Exception:
                            continue
                segs = sorted(list(set(segs)))

            if not segs:
                print(f"  No segments found for {run_id}, skipping position estimation")
            else:
                # initial positions/orientations from config if present
                initial_positions = getattr(record.config, 'initial_positions', {}) or {}
                initial_orientations = getattr(record.config, 'initial_orientations', {}) or {}

                # Initialize estimator
                est = SimpleSegmentEstimator(segment_ids=segs,
                                             initial_positions=initial_positions,
                                             initial_orientations=initial_orientations)

                n = df.height
                # prepare storage
                pos_storage = {s: {'x': [None]*n, 'y': [None]*n, 'z': [None]*n} for s in segs}
                quat_storage = {s: {'w': [None]*n, 'x': [None]*n, 'y': [None]*n, 'z': [None]*n} for s in segs}
                vel_storage = {s: {'vx': [None]*n, 'vy': [None]*n, 'vz': [None]*n, 'norm': [None]*n} for s in segs}

                # determine dt series
                if 'time_s' in df.columns:
                    times = df['time_s'].to_list()
                    dts = [times[i+1]-times[i] for i in range(len(times)-1)] + ([times[-1]-times[-2]] if len(times)>1 else [getattr(record.config, 'dt', 0.01)])
                else:
                    default_dt = getattr(record.config, 'dt', 0.01)
                    dts = [default_dt]*n

                # iterate rows
                for i, row in enumerate(df.iter_rows(named=True)):
                    sensor_data = {}
                    for seg in segs:
                        acc_cols = [f'acc_{seg}_X', f'acc_{seg}_Y', f'acc_{seg}_Z']
                        gyro_cols = [f'gyro_{seg}_X', f'gyro_{seg}_Y', f'gyro_{seg}_Z']
                        if all(c in df.columns for c in acc_cols+gyro_cols):
                            sensor_data[seg] = {
                                'acc': [row[c] for c in acc_cols],
                                'gyro': [row[c] for c in gyro_cols]
                            }
                    dt = dts[i] if i < len(dts) else dts[-1]
                    if sensor_data:
                        est.update_batch(sensor_data, dt)

                    # record states for each seg
                    states = est.get_all_states()
                    for seg in segs:
                        st = states.get(seg)
                        if st is not None:
                            pos_storage[seg]['x'][i] = float(st.position[0])
                            pos_storage[seg]['y'][i] = float(st.position[1])
                            pos_storage[seg]['z'][i] = float(st.position[2])
                            quat_storage[seg]['w'][i] = float(st.orientation[0])
                            quat_storage[seg]['x'][i] = float(st.orientation[1])
                            quat_storage[seg]['y'][i] = float(st.orientation[2])
                            quat_storage[seg]['z'][i] = float(st.orientation[3])
                            vel_storage[seg]['vx'][i] = float(st.velocity[0])
                            vel_storage[seg]['vy'][i] = float(st.velocity[1])
                            vel_storage[seg]['vz'][i] = float(st.velocity[2])
                            vel_storage[seg]['norm'][i] = float(__import__('numpy').linalg.norm(st.velocity))

                # Add columns to dataframe
                for seg in segs:
                    df = df.with_columns([
                        pl.Series(f"pos_estimate_{seg}_x", pos_storage[seg]['x']),
                        pl.Series(f"pos_estimate_{seg}_y", pos_storage[seg]['y']),
                        pl.Series(f"pos_estimate_{seg}_z", pos_storage[seg]['z']),
                        pl.Series(f"pos_estimate_{seg}_quat_w", quat_storage[seg]['w']),
                        pl.Series(f"pos_estimate_{seg}_quat_x", quat_storage[seg]['x']),
                        pl.Series(f"pos_estimate_{seg}_quat_y", quat_storage[seg]['y']),
                        pl.Series(f"pos_estimate_{seg}_quat_z", quat_storage[seg]['z']),
                        pl.Series(f"vel_estimate_{seg}_x", vel_storage[seg]['vx']),
                        pl.Series(f"vel_estimate_{seg}_y", vel_storage[seg]['vy']),
                        pl.Series(f"vel_estimate_{seg}_z", vel_storage[seg]['vz']),
                        pl.Series(f"vel_estimate_{seg}_norm", vel_storage[seg]['norm']),
                    ])

                # Save updated dataframe and metadata
                exp.save_experiment(df, record)
                print(f"  Position time series added and saved for {run_id}")

    # Skip meta-analysis if errors occur
    try:
        # After per-run processing, run meta-analysis to regenerate summary
        summary_df = ma.run_meta_analysis(plot=False)
        print("Meta-analysis completed.")
        print(summary_df)
    except Exception as e:
        print(f"Warning: Meta-analysis failed (this can happen if runs have inconsistent columns): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpiRob Analysis with optional position estimation")
    parser.add_argument('--enable-position-estimation', action='store_true', help='Enable ACC+GYRO position estimation and save time-series to data.parquet')
    parser.add_argument('--position-estimator-segments', nargs='+', type=int, help='List of segment IDs to estimate, e.g. 0 5 10')
    args = parser.parse_args()

    run_analysis(enable_position_estimation=args.enable_position_estimation,
                 position_estimator_segments=args.position_estimator_segments)