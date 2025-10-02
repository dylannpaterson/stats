import pandas as pd

def analyze_bayesian_summary(sa2_summary_path, ta_data_path):
    """
    Analyzes the Bayesian summary data by comparing observed values with estimated mean and MAP values at the TA level.

    Args:
        sa2_summary_path (str): The path to the SA2 summary CSV file.
        ta_data_path (str): The path to the TA level data CSV file.
    """
    try:
        # Read the CSV files
        sa2_df = pd.read_csv(sa2_summary_path)
        ta_df = pd.read_csv(ta_data_path)

        # Extract hh_code and bed_code from the sa2_summary_path filename
        parts = sa2_summary_path.split('_')
        hh_code = int(parts[-3])
        bed_code = int(parts[-1].split('.')[0])

        # Filter ta_df for the specific household composition and number of bedrooms
        ta_df_filtered = ta_df[(ta_df['household_composition_code'] == hh_code) & (ta_df['num_bedrooms'] == bed_code)]

        # Group sa2_df by 'ta_code' and sum the relevant columns
        sa2_agg = sa2_df.groupby('ta_code').agg({
            'estimated_count_mean': 'sum',
            'estimated_count_map': 'sum'
        }).reset_index()

        # Merge the aggregated SA2 data with the filtered TA data
        # Rename 'area_code' to 'ta_code' in ta_df_filtered for merging
        ta_df_filtered = ta_df_filtered.rename(columns={'area_code': 'ta_code'})
        merged_df = pd.merge(sa2_agg, ta_df_filtered, on='ta_code')

        # Select and rename columns for the final comparison table
        comparison_df = merged_df[['ta_code', 'OBS_VALUE', 'estimated_count_mean', 'estimated_count_map']]
        comparison_df = comparison_df.rename(columns={'OBS_VALUE': 'TA_OBS_VALUE'})

        # Print the results in a formatted table
        print("Comparison of TA Observed vs. Estimated values:")
        print(comparison_df.to_string())

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def analyze_national_summary(sa2_summary_path, national_data_path):
    """
    Analyzes the Bayesian summary data by comparing observed values with estimated mean and MAP values at the National level.

    Args:
        sa2_summary_path (str): The path to the SA2 summary CSV file.
        national_data_path (str): The path to the National level data CSV file.
    """
    try:
        # Read the CSV files
        sa2_df = pd.read_csv(sa2_summary_path)
        national_df = pd.read_csv(national_data_path)

        # Extract hh_code and bed_code from the sa2_summary_path filename
        parts = sa2_summary_path.split('_')
        hh_code = int(parts[-3])
        bed_code = int(parts[-1].split('.')[0])

        # Filter national_df for the specific household composition and number of bedrooms
        national_df_filtered = national_df[(national_df['household_composition_code'] == hh_code) & (national_df['num_bedrooms'] == bed_code)]

        # Sum the estimated counts from the SA2 data to get a national total
        national_est_mean = sa2_df['estimated_count_mean'].sum()
        national_est_map = sa2_df['estimated_count_map'].sum()

        # Get the national observed value
        national_obs_value = national_df_filtered['OBS_VALUE'].iloc[0]

        # Create a comparison dictionary
        comparison_data = {
            'National_OBS_VALUE': [national_obs_value],
            'National_est_mean': [national_est_mean],
            'National_est_map': [national_est_map]
        }
        comparison_df = pd.DataFrame(comparison_data)

        # Print the results in a formatted table
        print("Comparison of National Observed vs. Estimated values:")
        print(comparison_df.to_string())


    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sa2_summary_file = "/home/dylan/stats/data/outputs/bayesian_summaries/sa2_summary_hh_1102_bed_02.csv"
    ta_data_file = "/home/dylan/stats/data/processed/ta_level_data.csv"
    national_data_file = "/home/dylan/stats/data/processed/national_level_data.csv"
    
    analyze_bayesian_summary(sa2_summary_file, ta_data_file)
    print("\n" + "="*50 + "\n")
    analyze_national_summary(sa2_summary_file, national_data_file)
