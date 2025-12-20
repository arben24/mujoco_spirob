import math_spirob.meta_analyzer as ma

if __name__ == "__main__":
    # Run the meta-analysis without plotting (since we're in terminal)
    summary_df = ma.run_meta_analysis(plot=True)   #True/False
    print("Meta-analysis completed.")
    print(summary_df)