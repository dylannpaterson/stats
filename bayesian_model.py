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

# --- 2. Gibbs Sampler with Full Imputation ---

def sample_truncated_poisson(lam, k_max, size=1):
    """
    Samples from a Poisson distribution truncated at k_max.
    Vectorized for performance.
    """
    k_values = np.arange(k_max + 1)
    
    # If lam is a scalar, expand it to an array
    if np.isscalar(lam):
        lam = np.full(size, lam)
    
    # Calculate PMF for each lambda
    pmf_matrix = poisson.pmf(k_values[:, np.newaxis], lam)
    
    # Normalize PMFs
    pmf_sum = pmf_matrix.sum(axis=0)
    # Avoid division by zero for cases where sum is 0
    pmf_sum[pmf_sum == 0] = 1
    normalized_pmf = pmf_matrix / pmf_sum

    # Sample for each lambda
    samples = np.array([np.random.choice(k_values, p=p_col) for p_col in normalized_pmf.T])
    return samples


def run_full_imputation_gibbs_sampler(sa2_data, ta_totals_suppressed, national_total, n_iter=5000, burn_in=1000):
    """
    Runs a Gibbs sampler that imputes both missing (suppressed) 
    and noisy (RR3 rounded) data.
    """
    n_sa2s = len(sa2_data)
    n_tas = len(ta_totals_suppressed)
    
    sa2_posterior_samples = np.zeros((n_iter - burn_in, n_sa2s))
    ta_posterior_samples = np.zeros((n_iter - burn_in, n_tas))

    sa2_observed_original = sa2_data['OBS_VALUE'].values.copy()
    ta_observed_original = ta_totals_suppressed.values.copy()

    sa2_alpha_prior = np.nan_to_num(sa2_observed_original, nan=3.0) + 1
    ta_alpha_prior = np.nan_to_num(ta_observed_original, nan=3.0) + 1
    
    current_sa2_estimates = sa2_alpha_prior.copy()
    current_ta_estimates = ta_alpha_prior.copy()

    for i in range(n_iter):
        # STAGE 1 & 2: Sample true TA and SA2 counts
        ta_dirichlet_params = ta_alpha_prior + current_ta_estimates
        ta_proportions = np.random.dirichlet(ta_dirichlet_params)

        # --- Rejection sampling to constrain TA totals ---
        max_attempts = 100
        for attempt in range(max_attempts):
            sampled_ta_totals = np.random.multinomial(int(national_total), ta_proportions)
            
            is_compatible = True
            rounded_ta_idx = np.where(~np.isnan(ta_observed_original))[0]
            
            # Check only the rounded (non-suppressed) TAs
            if not np.all(
                (ta_observed_original[rounded_ta_idx] - 2 <= sampled_ta_totals[rounded_ta_idx]) &
                (sampled_ta_totals[rounded_ta_idx] <= ta_observed_original[rounded_ta_idx] + 2)
            ):
                is_compatible = False
            
            if is_compatible:
                break
        
        if not is_compatible:
            # If no compatible sample is found, we might just use the last one and warn the user.
            # This is a simplification. A more robust solution might be needed if this happens often.
            if i > burn_in: # Only warn after burn-in
                print(f"Warning: Could not find a compatible TA sample in iteration {i}. Using unconstrained sample.", flush=True)

        current_ta_estimates = sampled_ta_totals

        sampled_ta_totals_s = pd.Series(sampled_ta_totals, index=ta_totals_suppressed.index)
        for ta_code in sampled_ta_totals_s.index:
            sa2_indices = sa2_data[sa2_data['ta_code'] == ta_code].index
            if len(sa2_indices) == 0: continue

            sa2_dirichlet_params = sa2_alpha_prior[sa2_indices] + current_sa2_estimates[sa2_indices]
            sa2_proportions = np.random.dirichlet(sa2_dirichlet_params)
            sa2_total_for_ta = sampled_ta_totals_s[ta_code]
            new_sa2_counts = np.random.multinomial(int(sa2_total_for_ta), sa2_proportions)
            current_sa2_estimates[sa2_indices] = new_sa2_counts

        # STAGE 3: Impute observed data to inform the next iteration's prior
        # a) Impute TA data
        missing_ta_idx = np.where(np.isnan(ta_observed_original))[0]
        rounded_ta_idx = np.where(~np.isnan(ta_observed_original))[0]
        
        imputed_ta_values = np.zeros_like(ta_alpha_prior)
        if len(missing_ta_idx) > 0:
            imputed_ta_values[missing_ta_idx] = sample_truncated_poisson(lam=current_ta_estimates[missing_ta_idx], k_max=5, size=len(missing_ta_idx))
        if len(rounded_ta_idx) > 0:
            perturbation = np.random.randint(-2, 3, size=len(rounded_ta_idx))
            imputed_ta_values[rounded_ta_idx] = ta_observed_original[rounded_ta_idx] + perturbation
        ta_alpha_prior = np.maximum(0, imputed_ta_values) + 10

        # b) Impute SA2 data
        missing_sa2_idx = np.where(np.isnan(sa2_observed_original))[0]
        rounded_sa2_idx = np.where(~np.isnan(sa2_observed_original))[0]

        imputed_sa2_values = np.zeros_like(sa2_alpha_prior)
        if len(missing_sa2_idx) > 0:
            imputed_sa2_values[missing_sa2_idx] = sample_truncated_poisson(lam=current_sa2_estimates[missing_sa2_idx], k_max=5, size=len(missing_sa2_idx))
        if len(rounded_sa2_idx) > 0:
            perturbation = np.random.randint(-2, 3, size=len(rounded_sa2_idx))
            imputed_sa2_values[rounded_sa2_idx] = sa2_observed_original[rounded_sa2_idx] + perturbation
        sa2_alpha_prior = np.maximum(0, imputed_sa2_values) + 10

        # Store samples after burn-in
        if i >= burn_in:
            sa2_posterior_samples[i - burn_in, :] = current_sa2_estimates
            ta_posterior_samples[i - burn_in, :] = current_ta_estimates
            
    return sa2_posterior_samples, ta_posterior_samples

# --- 3. Main Execution ---

if __name__ == '__main__':
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
            sa2_data, ta_totals_suppressed, national_total, n_iter=10000, burn_in=1000
        )
        
        # --- Save results ---
        sa2_summary = sa2_data.copy()
        sa2_summary['estimated_count_mean'] = np.mean(sa2_samples, axis=0).round().astype(int)
        sa2_summary['estimated_count_map'] = mode(sa2_samples, axis=0)[0].round().astype(int)
        sa2_summary['ci_95_lower'] = np.percentile(sa2_samples, 2.5, axis=0).round().astype(int)
        sa2_summary['ci_95_upper'] = np.percentile(sa2_samples, 97.5, axis=0).round().astype(int)
        
        summary_filename = f"sa2_summary_hh_{hh_code}_bed_{bedroom_code}.csv"
        summary_filepath = os.path.join(summaries_output_dir, summary_filename)
        sa2_summary.to_csv(summary_filepath, index=False)
        print(f"  Saved SA2 summary to {summary_filepath}", flush=True)

        samples_filename = f"sa2_samples_hh_{hh_code}_bed_{bedroom_code}.npy"
        samples_filepath = os.path.join(samples_output_dir, samples_filename)
        np.save(samples_filepath, sa2_samples)
        print(f"  Saved SA2 samples to {samples_filepath}", flush=True)

        ta_summary = pd.DataFrame(index=ta_totals_suppressed.index)
        ta_summary['suppressed_count'] = ta_totals_suppressed
        ta_summary['estimated_count_mean'] = np.mean(ta_samples, axis=0).round().astype(int)
        ta_summary['estimated_count_map'] = mode(ta_samples, axis=0)[0].round().astype(int)
        ta_summary['ci_95_lower'] = np.percentile(ta_samples, 2.5, axis=0).round().astype(int)
        ta_summary['ci_95_upper'] = np.percentile(ta_samples, 97.5, axis=0).round().astype(int)

        summary_filename_ta = f"ta_summary_hh_{hh_code}_bed_{bedroom_code}.csv"
        summary_filepath_ta = os.path.join(summaries_output_dir, summary_filename_ta)
        ta_summary.to_csv(summary_filepath_ta)
        print(f"  Saved TA summary to {summary_filepath_ta}", flush=True)

    print("\n--- Full Run Complete ---", flush=True)
