import math_spirob.meta_analyzer as ma

if __name__ == "__main__":
    # Run the meta-analysis without plotting (since we're in terminal)
    summary_df = ma.run_meta_analysis(plot=False)   #True/False
    ma.plot_trends(summary_df, x_param="L_target", y_metrics=["tendonfrc_0_mean", "tendonfrc_1_mean"], title="Custom Plot", annotate_runs=True)
    print("Meta-analysis completed.")
    print(summary_df)