import math_spirob as ms

if __name__ == "__main__":
    # Example: Load summary and plot comparison
    try:
        # Plot comparison of mean acceleration X across experiments
        ms.plot_comparison(
            run_ids=["Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0", "Run_002_Ramped_Cyl_Sz0.06-0.10-0.00_L0.30_T2.0", "Run_003_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0"],
            sensor="acc_10",
            axis="X",
            metric="mean"
        )
        print("Comparison plot generated.")
    except Exception as e:
        print(f"Error in plotting: {e}")

    # Example: Time series plot (if data available)
    try:
        ms.plot_time_series(
            run_id="Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0",
            sensors=["acc_0"],
            axes=["X", "Y", "Z"],
            metric="raw"
        )
        print("Time series plot generated.")
    except Exception as e:
        print(f"Error in time series plotting: {e}")