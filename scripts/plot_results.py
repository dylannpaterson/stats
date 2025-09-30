import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_model_results(summary_filepath, output_image_path):
    """
    Loads a model summary file and creates a scatter plot comparing
    observed values to estimated values with credible intervals.
    """
    print(f"Loading results from: {summary_filepath}")
    df = pd.read_csv(summary_filepath)

    # Separate the data into suppressed (NaN) and rounded (not NaN)
    suppressed_data = df[df['OBS_VALUE'].isna()].copy()
    rounded_data = df[df['OBS_VALUE'].notna()].copy()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))

    # --- Plot Rounded Data ---
    if not rounded_data.empty:
        # Calculate asymmetric error bars for the 95% CI
        y_err_lower = rounded_data['estimated_count_map'] - rounded_data['ci_95_lower']
        y_err_upper = rounded_data['ci_95_upper'] - rounded_data['estimated_count_map']
        y_err = [y_err_lower, y_err_upper]

        ax.errorbar(
            x=rounded_data['OBS_VALUE'],
            y=rounded_data['estimated_count_map'],
            yerr=y_err,
            fmt='o',
            color='royalblue',
            ecolor='lightsteelblue',
            elinewidth=3,
            capsize=0,
            label='Rounded Observations (RR3)',
            alpha=0.7,
            markersize=5
        )

    # --- Plot Suppressed Data ---
    if not suppressed_data.empty:
        # For suppressed data, we plot them with x=-1 for clear visualization
        y_err_lower_supp = suppressed_data['estimated_count_map'] - suppressed_data['ci_95_lower']
        y_err_upper_supp = suppressed_data['ci_95_upper'] - suppressed_data['estimated_count_map']
        y_err_supp = [y_err_lower_supp, y_err_upper_supp]
        
        ax.errorbar(
            x=np.full(len(suppressed_data), -1), # Plot at x=-1
            y=suppressed_data['estimated_count_map'],
            yerr=y_err_supp,
            fmt='^', # Use triangles for suppressed
            color='seagreen',
            ecolor='mediumaquamarine',
            elinewidth=3,
            capsize=0,
            label='Suppressed Observations (Imputed)',
            alpha=0.8,
            markersize=6
        )

    # --- Plot y=x line for reference ---
    max_val = max(df['OBS_VALUE'].max(), df['estimated_count_map'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='y = x (No Change)')

    ax.set_xlabel("Original Observed Value (Suppressed shown at x=-1)")
    ax.set_ylabel("Model's Estimated MAP Count (with 95% Credible Interval)")
    ax.set_title("Model Results: Observed vs. Estimated MAP Counts")
    ax.legend()
    
    # Set axis limits for better viewing
    ax.axis('equal')
    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=-2)


    print(f"Saving plot to: {output_image_path}")
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Use one of the generated summary files as an example
    summary_file = 'data/outputs/simultaneous_results_summary.csv'
    output_image = 'results_comparison.png'
    
    if os.path.exists(summary_file):
        plot_model_results(summary_file, output_image)
    else:
        print(f"Error: Summary file not found at {summary_file}")
        print("Please ensure the simultaneous_bayesian_model.py has been run and has produced output.")