import pandas as pd
import os
import re

def synthesize_sa2_summaries():
    """
    Finds all individual SA2 summary CSVs, combines them, identifies any
    missing household/bedroom combinations from the master list, fills them
    with zero-count data, and saves the complete result.
    """
    # --- File Paths ---
    input_base_dir = 'data/processed'
    summaries_dir = 'data/outputs/bayesian_summaries_v2'
    output_path = 'data/outputs/combined_sa2_summary.csv'
    map_path = 'data/classifications/sa2_to_ta_map.csv'
    ta_level_path = os.path.join(input_base_dir, 'ta_level_data.csv')

    # --- Pre-flight Checks ---
    if not os.path.exists(summaries_dir):
        print(f"Error: Summaries directory not found -> '{summaries_dir}'")
        return
    if not os.path.exists(ta_level_path):
        print(f"Error: Master TA level data not found -> '{ta_level_path}'")
        return
    if not os.path.exists(map_path):
        print(f"Error: SA2 to TA map not found -> '{map_path}'")
        return

    # --- 1. Get the MASTER list of all possible combinations ---
    # This is the source of truth for what *should* exist.
    print("Loading master category list from TA data...")
    ta_level_data_full = pd.read_csv(ta_level_path, dtype=str)
    master_combinations = ta_level_data_full[['household_composition_code', 'num_bedrooms']].drop_duplicates()
    master_combinations_set = {tuple(x) for x in master_combinations.to_numpy()}
    print(f"Found {len(master_combinations_set)} total possible combinations.")

    # --- 2. Load and combine EXISTING summaries ---
    all_summary_files = [f for f in os.listdir(summaries_dir) if f.startswith('sa2_summary_hh_') and f.endswith('.csv')]
    pattern = re.compile(r"sa2_summary_hh_(\w+)_bed_(\w+)\.csv")
    
    df_list = []
    processed_combinations_set = set()

    for filename in all_summary_files:
        match = pattern.match(filename)
        if match:
            hh_code, bedroom_code = match.group(1), match.group(2)
            processed_combinations_set.add((hh_code, bedroom_code))
            
            filepath = os.path.join(summaries_dir, filename)
            temp_df = pd.read_csv(filepath, dtype=str)
            df_list.append(temp_df)
    
    if not df_list:
        print("No existing summary files found to process.")
        combined_df = pd.DataFrame()
    else:
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Combined {len(processed_combinations_set)} existing summaries into a single dataframe.")

    # --- 3. Identify and generate MISSING summaries ---
    missing_combinations = master_combinations_set - processed_combinations_set
    print(f"Identified {len(missing_combinations)} missing combinations to generate as zero-count.")

    if missing_combinations:
        # We need a list of all SA2s to build the zero-count rows
        sa2_map = pd.read_csv(map_path, dtype=str)
        all_sa2_codes = sa2_map['SA22023_code'].unique()

        # Get the complete column structure from an existing summary file
        # This is safer than hardcoding column names
        if not df_list:
             print("Error: Cannot generate missing summaries without at least one existing summary file to use as a template.")
             return
        
        template_cols = pd.read_csv(os.path.join(summaries_dir, all_summary_files[0]), dtype=str).columns
        
        missing_dfs = []
        for hh_code, bedroom_code in missing_combinations:
            # Create a dataframe for this missing combination
            missing_data = []
            for sa2 in all_sa2_codes:
                # This is a simplified row; the original summaries have more detail per SA2.
                # Let's replicate the structure of the *input* SA2 data for a zero-count entry.
                # Assuming the summary file just adds estimate columns to the input file structure.
                row = {
                    'sa2_code': sa2,
                    'household_composition_code': hh_code,
                    'num_bedrooms': bedroom_code,
                    'OBS_VALUE': '0',
                    'estimated_count_mean': '0',
                    'estimated_count_map': '0',
                    'ci_95_lower': '0',
                    'ci_95_upper': '0'
                }
                # Add any other columns present in the template with a default value
                for col in template_cols:
                    if col not in row:
                        row[col] = '0' # Or some other sensible default
                missing_data.append(row)
            
            missing_df_for_combo = pd.DataFrame(missing_data)
            # Ensure column order matches the original
            missing_df_for_combo = missing_df_for_combo[template_cols.tolist()]
            missing_dfs.append(missing_df_for_combo)

        # Append the new zero-count dataframes to the main list
        df_list.extend(missing_dfs)
        combined_df = pd.concat(df_list, ignore_index=True, sort=False)

    # --- 4. Finalize and Save ---
    # Reorder columns for clarity
    key_cols = ['household_composition_code', 'num_bedrooms', 'sa2_code']
    other_cols = [c for c in combined_df.columns if c not in key_cols]
    final_df = combined_df[key_cols + other_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"\nSynthesis complete!")
    print(f"Final dataset has {len(final_df)} rows from {len(master_combinations_set)} total combinations.")
    print(f"Fully synthesized data saved to: {output_path}")


if __name__ == '__main__':
    synthesize_sa2_summaries()