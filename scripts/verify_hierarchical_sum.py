import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def combine_ta_summaries(summary_dir, output_path):
    """
    Combines individual TA summary files into a single CSV.
    """
    if os.path.exists(output_path):
        return pd.read_csv(output_path, dtype=str)

    print("Combined TA summary not found. Generating it now...")
    ta_files = [f for f in os.listdir(summary_dir) if f.startswith('ta_summary_') and f.endswith('.csv')]
    if not ta_files:
        print("No TA summary files found to combine.")
        return None

    pattern = re.compile(r"ta_summary_hh_(\w+)_bed_(\w+)\.csv")
    df_list = []
    for filename in ta_files:
        match = pattern.match(filename)
        if match:
            hh, bed = match.groups()
            temp_df = pd.read_csv(os.path.join(summary_dir, filename), dtype=str)
            temp_df['household_composition_code'] = hh
            temp_df['num_bedrooms'] = bed
            df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined TA summary saved to '{output_path}'")
    return combined_df

def verify_estimates_overlaid_corrected():
    """
    Verifies hierarchical summing, plotting both observed data and model estimates
    on the same graph for direct comparison. This version corrects the .UPPER() typo.
    """
    # --- Configuration ---
    estimates_to_plot = ['mean', 'map']
    
    # --- File Paths ---
    summaries_dir = 'data/outputs/bayesian_summaries_v2'
    output_plot_template = 'hierarchical_sum_verification_overlaid_{estimate_type}.png'
    sa2_combined_path = 'data/outputs/combined_sa2_summary_complete.csv'
    ta_combined_path = 'data/outputs/combined_ta_summary.csv'

    # --- 1. Load Data ---
    print("Loading data...")
    if not os.path.exists(sa2_combined_path):
        print(f"FATAL: Combined SA2 summary not found at '{sa2_combined_path}'.")
        return
    sa2_data = pd.read_csv(sa2_combined_path, dtype=str)
    ta_data = combine_ta_summaries(summaries_dir, ta_combined_path)
    if ta_data is None: return

    # --- 2. Data Cleaning & Type Conversion ---
    print("Standardizing column names and types...")
    value_cols_to_convert = ['OBS_VALUE', 'suppressed_count', 'estimated_count_mean', 'estimated_count_map']
    for col in value_cols_to_convert:
        if col in sa2_data.columns:
            sa2_data[col] = pd.to_numeric(sa2_data[col], errors='coerce').fillna(0)
        if col in ta_data.columns:
            ta_data[col] = pd.to_numeric(ta_data[col], errors='coerce').fillna(0)
    
    ta_data = ta_data.rename(columns={'area_code': 'ta_code'})

    # --- Loop and Plot ---
    for estimate_type in estimates_to_plot:
        estimate_col = f'estimated_count_{estimate_type}'
        # --- CORRECTED LINE ---
        print(f"--- Verifying: {estimate_type.upper()} ---")

        # --- 3. Aggregate SA2 Data ---
        grouping_cols = ['ta_code', 'household_composition_code', 'num_bedrooms']
        agg_dict = {
            f'sa2_summed_{estimate_type}': (estimate_col, 'sum'),
            'sa2_summed_observed': ('OBS_VALUE', 'sum')
        }
        sa2_aggregated = sa2_data.groupby(grouping_cols).agg(**agg_dict).reset_index()

        # --- 4. Merge for Comparison ---
        comparison_df = pd.merge(sa2_aggregated, ta_data, on=grouping_cols, how='inner')

        if comparison_df.empty:
            # --- CORRECTED LINE ---
            print(f"Merge was empty for {estimate_type.upper()}. Skipping plot.")
            continue

        # --- 5. Plot the Results on a single graph ---
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot Observed Data
        ax.scatter(comparison_df['suppressed_count'], comparison_df['sa2_summed_observed'], alpha=0.7, edgecolors='w', s=60, c='seagreen', label='Observed Data')
        
        # Plot Model Estimate Data
        model_color = 'royalblue' if estimate_type == 'mean' else 'darkviolet'
        # --- CORRECTED LINE ---
        ax.scatter(comparison_df[estimate_col], comparison_df[f'sa2_summed_{estimate_type}'], alpha=0.7, edgecolors='w', s=60, c=model_color, label=f'Model {estimate_type.upper()} Estimate')

        # Plot Perfect Agreement Line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Agreement')
        
        ax.set_xlabel('TA-Level Value (Direct)', fontsize=14)
        ax.set_ylabel('Sum of SA2-Level Values', fontsize=14)
        # --- CORRECTED LINE ---
        ax.set_title(f'Hierarchical Sum Verification ({estimate_type.upper()} vs. Observed)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_aspect('equal', 'box')
        ax.grid(True, which="both", ls="--")

        plt.tight_layout()
        output_path = output_plot_template.format(estimate_type=estimate_type)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Verification plot saved to: {output_path}")

if __name__ == '__main__':
    verify_estimates_overlaid_corrected()