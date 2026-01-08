#!/usr/bin/env python3
"""
CLI for plotting MuJoCo SpiRob data.

Usage:
    python apps/spirob_plot_cli.py --kind time_series --sensor acc_0 --axis X --runs all
    python apps/spirob_plot_cli.py --kind comparison --sensor acc_0 --axis X --metric mean --runs Run_001 Run_002
    python apps/spirob_plot_cli.py --kind distribution --sensor acc_0 --axis X --metric mean --runs all
    python apps/spirob_plot_cli.py --kind grid --sensors acc_0 acc_1 --axes X --metrics mean std --runs all
"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_spirob.plots import plot_time_series, plot_comparison, plot_distribution, plot_metric_grid

def main():
    parser = argparse.ArgumentParser(description="Plot MuJoCo SpiRob data")
    parser.add_argument("--kind", choices=["time_series", "comparison", "distribution", "grid"], required=True)
    parser.add_argument("--sensor", help="Sensor name")
    parser.add_argument("--sensors", nargs="+", help="List of sensors")
    parser.add_argument("--axis", help="Axis")
    parser.add_argument("--axes", nargs="+", help="List of axes")
    parser.add_argument("--metric", help="Metric")
    parser.add_argument("--metrics", nargs="+", help="List of metrics")
    parser.add_argument("--runs", nargs="+", help="Run IDs or 'all'")
    parser.add_argument("--base_dir", default="build")
    parser.add_argument("--save_path", help="Path to save plot")

    args = parser.parse_args()

    runs = args.runs if args.runs != ["all"] else "all"
    if isinstance(runs, list) and len(runs) == 1 and runs[0] == "all":
        runs = "all"

    if args.kind == "time_series":
        sensors = args.sensors or [args.sensor]
        axes = args.axes or [args.axis]
        plot_time_series(runs, sensors, axes, args.metric or "raw", args.base_dir, save_path=args.save_path)
    elif args.kind == "comparison":
        plot_comparison(runs, args.sensor, args.axis, args.metric, args.base_dir, save_path=args.save_path)
    elif args.kind == "distribution":
        plot_distribution(runs, args.sensor, args.axis, args.metric, args.base_dir, save_path=args.save_path)
    elif args.kind == "grid":
        plot_metric_grid(runs, args.sensors, args.axes, args.metrics, args.base_dir, save_path=args.save_path)

if __name__ == "__main__":
    main()