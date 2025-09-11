import pandas as pd
import numpy as np
import os
from scipy.stats import poisson

# --- 1. Data Loading and Preparation ---


def load_and_pivot_data(sa2_path, ta_path, national_path, map_path):
    """
    Loads all data and pivots it into a wide format (areas x categories).
    """
    print("Loading and pivoting data...")
    sa2_df = pd.read_csv(sa2_path, dtype=str)
    ta_df = pd.read_csv(ta_path, dtype=str)
    national_df = pd.read_csv(national_path, dtype=str)
    sa2_to_ta = pd.read_csv(map_path, encoding="utf-8-sig", dtype=str)

    for df in [sa2_df, ta_df, national_df]:
        df["category"] = df["household_composition_code"] + "_" + df["num_bedrooms"]
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    sa2_wide = pd.pivot_table(
        sa2_df, values="OBS_VALUE", index="area_code", columns="category"
    )
    ta_wide = pd.pivot_table(
        ta_df, values="OBS_VALUE", index="area_code", columns="category"
    )
    national_s = pd.pivot_table(
        national_df, values="OBS_VALUE", columns="category"
    ).iloc[0]

    common_categories = sorted(
        list(set(sa2_wide.columns) & set(ta_wide.columns) & set(national_s.index))
    )
    sa2_wide = sa2_wide[common_categories].sort_index()
    ta_wide = ta_wide[common_categories].sort_index()
    national_s = national_s[common_categories]

    sa2_to_ta.rename(
        columns={"SA22023_code": "sa2_code", "TA2023_code": "ta_code"}, inplace=True
    )
    sa2_to_ta = sa2_to_ta[sa2_to_ta["sa2_code"].isin(sa2_wide.index)]
    sa2_to_ta = sa2_to_ta[sa2_to_ta["ta_code"].isin(ta_wide.index)].sort_values(
        "sa2_code"
    )

    sa2_wide = sa2_wide.loc[sa2_to_ta["sa2_code"]]
    ta_wide = ta_wide.sort_index()

    print("Data loading and pivoting complete.")
    return sa2_wide, ta_wide, national_s, sa2_to_ta


# --- 2. Gibbs Sampler Implementation ---


def sample_truncated_poisson(lam, k_max, size=1):
    k_values = np.arange(k_max + 1)
    if np.isscalar(lam):
        lam = np.full(size, lam)
    pmf_matrix = poisson.pmf(k_values[:, np.newaxis], lam)
    pmf_sum = pmf_matrix.sum(axis=0)
    pmf_sum[pmf_sum == 0] = 1
    normalized_pmf = pmf_matrix / pmf_sum
    return np.array([np.random.choice(k_values, p=p_col) for p_col in normalized_pmf.T])


def run_simultaneous_gibbs_sampler(
    sa2_wide, ta_wide, national_s, sa2_to_ta, n_iter, burn_in, samples_path
):
    """
    Runs a Gibbs sampler, saving samples to a memory-mapped file on disk.
    """
    n_sa2s, n_cats = sa2_wide.shape
    n_tas = ta_wide.shape[0]
    n_samples_to_store = n_iter - burn_in

    # Create a memory-mapped file on disk to store samples, avoiding high RAM usage
    shape = (n_samples_to_store, n_sa2s, n_cats)
    sa2_posterior_samples = np.memmap(
        samples_path, dtype="float64", mode="w+", shape=shape
    )
    print(f"Created memory-mapped file for samples at: {samples_path}")

    sa2_obs = sa2_wide.values
    ta_obs = ta_wide.values

    sa2_alpha = np.nan_to_num(sa2_obs, nan=3.0) + 1
    ta_alpha = np.nan_to_num(ta_obs, nan=3.0) + 1

    current_sa2_est = sa2_alpha.copy()
    current_ta_est = ta_alpha.copy()

    print(f"Running SIMULTANEOUS Gibbs sampler for {n_iter} iterations...")
    for i in range(n_iter):
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{n_iter}")

        for c_idx, cat in enumerate(ta_wide.columns):
            if national_s[cat] == 0:
                current_ta_est[:, c_idx] = 0
                continue
            ta_dirichlet_params = ta_alpha[:, c_idx] + current_ta_est[:, c_idx]
            ta_proportions = np.random.dirichlet(ta_dirichlet_params)
            sampled_tas = np.random.multinomial(int(national_s[cat]), ta_proportions)
            current_ta_est[:, c_idx] = sampled_tas

        for j_idx, ta_code in enumerate(ta_wide.index):
            sa2s_in_ta_mask = (sa2_to_ta["ta_code"] == ta_code).values
            for c_idx, cat in enumerate(sa2_wide.columns):
                sa2_total_for_ta_cat = current_ta_est[j_idx, c_idx]
                if sa2_total_for_ta_cat == 0:
                    current_sa2_est[sa2s_in_ta_mask, c_idx] = 0
                    continue
                sa2_dirichlet_params = (
                    sa2_alpha[sa2s_in_ta_mask, c_idx]
                    + current_sa2_est[sa2s_in_ta_mask, c_idx]
                )
                sa2_proportions = np.random.dirichlet(sa2_dirichlet_params)
                new_sa2_counts = np.random.multinomial(
                    int(sa2_total_for_ta_cat), sa2_proportions
                )
                current_sa2_est[sa2s_in_ta_mask, c_idx] = new_sa2_counts

        ta_nan_mask = np.isnan(ta_obs)
        if np.any(ta_nan_mask):
            ta_alpha[ta_nan_mask] = (
                sample_truncated_poisson(current_ta_est[ta_nan_mask], k_max=5) + 1
            )
        sa2_nan_mask = np.isnan(sa2_obs)
        if np.any(sa2_nan_mask):
            sa2_alpha[sa2_nan_mask] = (
                sample_truncated_poisson(current_sa2_est[sa2_nan_mask], k_max=5) + 1
            )

        if i >= burn_in:
            sa2_posterior_samples[i - burn_in, :, :] = current_sa2_est

    print("Gibbs sampler finished.")
    # The function no longer returns the samples array


# --- 3. Main Execution ---

if __name__ == "__main__":
    input_base_dir = "/home/dylan/Documents/stats/data/processed"
    output_base_dir = "/home/dylan/Documents/stats/data/outputs"
    sa2_path = os.path.join(input_base_dir, "sa2_level_data.csv")
    ta_path = os.path.join(input_base_dir, "ta_level_data.csv")
    national_path = os.path.join(input_base_dir, "national_level_data.csv")
    map_path = "/home/dylan/Documents/stats/data/classifications/sa2_to_ta_map.csv"
    summary_output_path = os.path.join(
        output_base_dir, "simultaneous_results_summary.csv"
    )
    samples_output_path = os.path.join(output_base_dir, "simultaneous_raw_samples.mmap")

    sa2_wide, ta_wide, national_s, sa2_to_ta = load_and_pivot_data(
        sa2_path, ta_path, national_path, map_path
    )

    N_ITER = 5000
    BURN_IN = 1000

    run_simultaneous_gibbs_sampler(
        sa2_wide,
        ta_wide,
        national_s,
        sa2_to_ta,
        n_iter=N_ITER,
        burn_in=BURN_IN,
        samples_path=samples_output_path,
    )

    # --- Process Results from Disk ---
    print("\nProcessing posterior samples from disk to calculate summaries...")

    # Open the memory-mapped file for reading
    shape = (N_ITER - BURN_IN, sa2_wide.shape[0], sa2_wide.shape[1])
    all_samples = np.memmap(samples_output_path, dtype="float64", mode="r", shape=shape)

    # Calculate summary stats (this will be memory efficient)
    mean_estimates = np.mean(all_samples, axis=0)
    ci_lower = np.percentile(all_samples, 2.5, axis=0)
    ci_upper = np.percentile(all_samples, 97.5, axis=0)

    # Create DataFrames from the matrices
    mean_df = pd.DataFrame(
        mean_estimates, index=sa2_wide.index, columns=sa2_wide.columns
    )
    lower_df = pd.DataFrame(ci_lower, index=sa2_wide.index, columns=sa2_wide.columns)
    upper_df = pd.DataFrame(ci_upper, index=sa2_wide.index, columns=sa2_wide.columns)

    # Melt all dataframes to long format
    obs_long = sa2_wide.reset_index().melt(
        id_vars="area_code", var_name="category", value_name="OBS_VALUE"
    )
    mean_long = mean_df.reset_index().melt(
        id_vars="area_code", var_name="category", value_name="estimated_count_mean"
    )
    lower_long = lower_df.reset_index().melt(
        id_vars="area_code", var_name="category", value_name="ci_95_lower"
    )
    upper_long = upper_df.reset_index().melt(
        id_vars="area_code", var_name="category", value_name="ci_95_upper"
    )

    # Merge into a single summary dataframe
    summary_df = pd.merge(obs_long, mean_long, on=["area_code", "category"])
    summary_df = pd.merge(summary_df, lower_long, on=["area_code", "category"])
    summary_df = pd.merge(summary_df, upper_long, on=["area_code", "category"])

    int_cols = ["estimated_count_mean", "ci_95_lower", "ci_95_upper"]
    summary_df[int_cols] = summary_df[int_cols].round().astype(int)

    summary_df[["household_composition_code", "num_bedrooms"]] = summary_df[
        "category"
    ].str.split("_", expand=True)
    summary_df.drop(columns=["category"], inplace=True)

    print(f"Saving final summary results to {summary_output_path}")
    summary_df.to_csv(summary_output_path, index=False)
    print("Successfully saved simultaneous results.")
