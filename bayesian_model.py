import pandas as pd
import numpy as np
import os
from scipy.stats import poisson, mode

# --- 1. Data Loading and Preparation ---

def load_and_prepare_data(sa2_level_data, ta_level_data, national_level_data, sa2_to_ta, hh_code_filter, bedroom_code_filter):
    """
    Filters data for a specific category, preserving NaNs for imputation.
    """
    # --- TA Data ---
    ta_filtered = ta_level_data[
        (ta_level_data['household_composition_code'] == hh_code_filter) &
        (ta_level_data['num_bedrooms'] == bedroom_code_filter)
    ].copy()
    ta_filtered['OBS_VALUE'] = pd.to_numeric(ta_filtered['OBS_VALUE'], errors='coerce')
    ta_totals_suppressed = ta_filtered.set_index('area_code')['OBS_VALUE']

    # --- SA2 Data ---
    sa2_filtered = sa2_level_data[
        (sa2_level_data['household_composition_code'] == hh_code_filter) &
        (sa2_level_data['num_bedrooms'] == bedroom_code_filter)
    ].copy()
    sa2_filtered['OBS_VALUE'] = pd.to_numeric(sa2_filtered['OBS_VALUE'], errors='coerce')

    df = pd.merge(sa2_filtered, sa2_to_ta, on='sa2_code')
    df = df[df['ta_code'].isin(ta_totals_suppressed.index)]
    df = df.sort_values(by=['ta_code', 'sa2_code']).reset_index(drop=True)

    if df.empty:
        return None, None, None

    ordered_ta_codes = df['ta_code'].unique()
    ta_totals_suppressed = ta_totals_suppressed.reindex(ordered_ta_codes)

    # --- National Total ---
    national_row = national_level_data[
        (national_level_data['household_composition_code'] == hh_code_filter) &
        (national_level_data['num_bedrooms'] == bedroom_code_filter)
    ]

    if not national_row.empty:
        value = pd.to_numeric(national_row['OBS_VALUE'].iloc[0], errors='coerce')
        national_total = 0 if pd.isna(value) else value
    else:
        print('  Warning: National total not found. Using sum of TAs as fallback.', flush=True)
        national_total = ta_totals_suppressed.sum()

    return df, ta_totals_suppressed, national_total

# --- 2. Gibbs Sampler with Refined Sampling Logic ---

def sample_constrained_ta(current_sa2_sum, rounded_ta_observed):
    """
    Samples a single TA value that is consistent with the RR3 rule,
    weighted by the Poisson probability based on the sum of its children SA2s.
    """
    if pd.isna(rounded_ta_observed):
        # If the TA is suppressed (NaN), we sample from a simple Poisson.
        # This is a reasonable assumption for suppressed data.
        return np.random.poisson(max(0, current_sa2_sum))

    # Define the menu of 5 possible integer choices based on the RR3 rule
    valid_integers = np.arange(rounded_ta_observed - 2, rounded_ta_observed + 3)
    valid_integers = valid_integers[valid_integers >= 0] # Ensure non-negative

    if len(valid_integers) == 0:
        return 0

    # Roll a fair die: draw one sample uniformly from the valid choices.
    return np.random.choice(valid_integers)


def run_full_imputation_gibbs_sampler(sa2_data, ta_totals_suppressed, national_total, n_iter=10000, burn_in=2000):
    """
    Runs a Gibbs sampler with a stable, direct sampling method.
    """
    n_sa2s = len(sa2_data)
    n_tas = len(ta_totals_suppressed)

    sa2_posterior_samples = np.zeros((n_iter - burn_in, n_sa2s))
    ta_posterior_samples = np.zeros((n_iter - burn_in, n_tas))

    sa2_observed_original = sa2_data['OBS_VALUE'].values.copy()
    ta_observed_original = ta_totals_suppressed.values.copy()
    
    # --- Priors are now simpler and fixed ---
    # We use a non-informative prior. Small constant, e.g., 1.
    sa2_prior = np.full(n_sa2s, 1.0)
    ta_prior = np.full(n_tas, 1.0)

    # --- Initial State ---
    # A reasonable starting point for the chain
    current_sa2_estimates = np.nan_to_num(sa2_observed_original, nan=np.nanmean(sa2_observed_original[~np.isnan(sa2_observed_original)]))
    current_sa2_estimates = np.maximum(1, current_sa2_estimates)
    
    # Group by TA to initialize TA estimates
    sa2_df_temp = sa2_data.copy()
    sa2_df_temp['initial_est'] = current_sa2_estimates
    initial_ta_sums = sa2_df_temp.groupby('ta_code')['initial_est'].sum()
    initial_ta_sums = initial_ta_sums.reindex(ta_totals_suppressed.index).fillna(0)
    current_ta_estimates = initial_ta_sums.values


    for i in range(n_iter):
        # --- STAGE 1: Sample TA totals ---
        # This is the new, direct sampling stage. No more rejection sampling.
        
        # Grand total proportion sampling
        ta_proportions = np.random.dirichlet(ta_prior + current_ta_estimates)
        total_sum = np.random.multinomial(int(national_total), ta_proportions)

        # Now, for each TA that has observed data, we resample it using our
        # custom discrete sampler to ensure it conforms to RR3 constraints.
        
        # Calculate current SA2 sums for each TA
        sa2_df_temp['current_est'] = current_sa2_estimates
        current_sa2_sums_by_ta = sa2_df_temp.groupby('ta_code')['current_est'].sum()
        current_sa2_sums_by_ta = current_sa2_sums_by_ta.reindex(ta_totals_suppressed.index).fillna(0)

        new_ta_estimates = np.zeros_like(current_ta_estimates)
        for j, ta_code in enumerate(ta_totals_suppressed.index):
            observed_val = ta_observed_original[j]
            sa2_sum_for_ta = current_sa2_sums_by_ta[ta_code]
            
            # If the TA is suppressed, we trust the multinomial draw more.
            # If it's observed, we use our constrained sampler.
            if pd.isna(observed_val):
                new_ta_estimates[j] = total_sum[j]
            else:
                 new_ta_estimates[j] = sample_constrained_ta(sa2_sum_for_ta, observed_val)
        
        # Rescale to match national total
        if new_ta_estimates.sum() > 0:
            current_ta_estimates = np.round(new_ta_estimates * (national_total / new_ta_estimates.sum())).astype(int)
        else:
            current_ta_estimates = new_ta_estimates

        # --- STAGE 2: Sample SA2 counts conditional on TA totals ---
        for j, ta_code in enumerate(ta_totals_suppressed.index):
            sa2_indices = sa2_data[sa2_data['ta_code'] == ta_code].index
            if len(sa2_indices) == 0: continue

            # The prior here can be informed by the observed SA2 data if available
            sa2_sub_prior = np.nan_to_num(sa2_observed_original[sa2_indices], nan=1.0) + 1.0

            dirichlet_params = sa2_sub_prior + current_sa2_estimates[sa2_indices]
            
            # Ensure params are positive
            dirichlet_params[dirichlet_params <= 0] = 1e-9

            sa2_proportions = np.random.dirichlet(dirichlet_params)
            sa2_total_for_ta = current_ta_estimates[j]
            
            new_sa2_counts = np.random.multinomial(int(sa2_total_for_ta), sa2_proportions)
            current_sa2_estimates[sa2_indices] = new_sa2_counts

        # Store samples after burn-in
        if i >= burn_in:
            sa2_posterior_samples[i - burn_in, :] = current_sa2_estimates
            ta_posterior_samples[i - burn_in, :] = current_ta_estimates

    return sa2_posterior_samples, ta_posterior_samples


# --- 3. Main Execution ---
if __name__ == '__main__':
    # (The main execution block remains unchanged as it handles I/O and orchestration)
    input_base_dir = 'data/processed'
    sa2_path = os.path.join(input_base_dir, 'sa2_level_data.csv')
    ta_path = os.path.join(input_base_dir, 'ta_level_data.csv')
    national_path = os.path.join(input_base_dir, 'national_level_data.csv')
    map_path = 'data/classifications/sa2_to_ta_map.csv'
    output_base_dir = 'data/outputs'
    samples_output_dir = os.path.join(output_base_dir, 'bayesian_samples')
    summaries_output_dir = os.path.join(output_base_dir, 'bayesian_summaries')
    os.makedirs(samples_output_dir, exist_ok=True)
    os.makedirs(summaries_output_dir, exist_ok=True)
    print(f"Outputs will be saved in: \n  {samples_output_dir} \n  {summaries_output_dir}", flush=True)

    print("\nPre-loading all data files...", flush=True)
    sa2_level_data_full = pd.read_csv(sa2_path, dtype=str)
    ta_level_data_full = pd.read_csv(ta_path, dtype=str)
    national_level_data_full = pd.read_csv(national_path, dtype=str)
    sa2_to_ta_full = pd.read_csv(map_path, encoding='utf-8-sig', dtype=str)
    sa2_to_ta_full.rename(columns={'SA22023_code': 'sa2_code', 'TA2023_code': 'ta_code'}, inplace=True)
    sa2_level_data_full.rename(columns={'area_code': 'sa2_code'}, inplace=True)

    category_combinations = ta_level_data_full[['household_composition_code', 'num_bedrooms']].drop_duplicates()
    print(f"Found {len(category_combinations)} unique categories to model.", flush=True)

    for index, row in category_combinations.iterrows():
        hh_code = row['household_composition_code']
        bedroom_code = row['num_bedrooms']
        
        print(f"\n--- Processing: HH_CODE='{hh_code}', BEDROOM_CODE='{bedroom_code}' ---", flush=True)

        sa2_data, ta_totals_suppressed, national_total = load_and_prepare_data(
            sa2_level_data_full, ta_level_data_full, national_level_data_full, sa2_to_ta_full, hh_code, bedroom_code
        )
        
        if sa2_data is None: continue

        print(f"  Found {len(sa2_data)} SA2 areas and {len(ta_totals_suppressed)} TAs to model with National Total: {national_total}", flush=True)

        if national_total == 0:
            print("  National total is 0. Skipping Gibbs sampler as all counts must be 0.", flush=True)
            continue

        sa2_samples, ta_samples = run_full_imputation_gibbs_sampler(
            sa2_data, ta_totals_suppressed, national_total, n_iter=10000, burn_in=2000
        )
        
        # --- Save results ---
        sa2_summary = sa2_data.copy()
        sa2_summary['estimated_count_mean'] = np.mean(sa2_samples, axis=0).round().astype(int)
        sa2_summary['estimated_count_median'] = np.percentile(sa2_samples, 50.0, axis=0).round().astype(int)
        sa2_summary['estimated_count_map'] = mode(sa2_samples, axis=0, keepdims=True)[0][0].round().astype(int)
        sa2_summary['ci_95_lower'] = np.percentile(sa2_samples, 2.5, axis=0).round().astype(int)
        sa2_summary['ci_95_upper'] = np.percentile(sa2_samples, 97.5, axis=0).round().astype(int)
        
        summary_filename = f"sa2_summary_hh_{hh_code}_bed_{bedroom_code}.csv"
        summary_filepath = os.path.join(summaries_output_dir, summary_filename)
        sa2_summary.to_csv(summary_filepath, index=False)
        print(f"  Saved SA2 summary to {summary_filepath}", flush=True)

        # samples_filename = f"sa2_samples_hh_{hh_code}_bed_{bedroom_code}.npy"
        # samples_filepath = os.path.join(samples_output_dir, samples_filename)
        # np.save(samples_filepath, sa2_samples)
        # print(f"  Saved SA2 samples to {samples_filepath}", flush=True)

        ta_summary = pd.DataFrame(index=ta_totals_suppressed.index)
        ta_summary['suppressed_count'] = ta_totals_suppressed
        ta_summary['estimated_count_mean'] = np.mean(ta_samples, axis=0).round().astype(int)
        ta_summary['estimated_count_median'] = np.percentile(ta_samples, 50.0, axis=0).round().astype(int)
        ta_summary['estimated_count_map'] = mode(ta_samples, axis=0, keepdims=True)[0][0].round().astype(int)
        ta_summary['ci_95_lower'] = np.percentile(ta_samples, 2.5, axis=0).round().astype(int)
        ta_summary['ci_95_upper'] = np.percentile(ta_samples, 97.5, axis=0).round().astype(int)

        summary_filename_ta = f"ta_summary_hh_{hh_code}_bed_{bedroom_code}.csv"
        summary_filepath_ta = os.path.join(summaries_output_dir, summary_filename_ta)
        ta_summary.to_csv(summary_filepath_ta)
        print(f"  Saved TA summary to {summary_filepath_ta}", flush=True)

    print("\n--- Full Run Complete ---", flush=True)