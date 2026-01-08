import pandas as pd

def get_descriptive_stats(input_file):
    """
    Reads an Excel file and prints descriptive statistics for all features.

    Args:
        input_file (str): The name of the Excel file (.xlsx).
    """
    try:
        # Read data from the Excel file
        df = pd.read_excel(input_file)
        
        # Preprocessing: Replace 'no data' and empty cells with 0
        df = df.replace(['no data', 'No data'], 0).fillna(0)
        
        # Select all feature columns (excluding the first two and the last)
        features_df = df.iloc[:, 2:]
        
        # Generate and print the descriptive statistics
        print("Descriptive Statistics for Features:")
        print(features_df.describe())
        
        stats_df = features_df.describe()
        # Xuất thống kê mô tả ra file Excel
        stats_df.to_excel("descriptive_stats.xlsx")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the file name.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the function ---
# Make sure to place your 'data_raw.xlsx' file in the same directory as this script.
get_descriptive_stats("raw_data.xlsx")