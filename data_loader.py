import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks
from utils.data_to_text import extract_features_from_time_series, save_features_as_text

def plot_time_series(df, datetime_col, value_col, output_dir, file_name_no_ext):
    """
    Plot time series data and save the figure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data
    datetime_col : str
        Name of the column containing datetime information
    value_col : str
        Name of the column containing the values to plot
    output_dir : str
        Directory to save the plot
    file_name_no_ext : str
        Base filename without extension for the output file
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[datetime_col], df[value_col], linestyle='-', marker='', linewidth=1)
    
    # Add title and labels
    plt.title(f'Time Series: {value_col}')
    plt.xlabel('Time')
    plt.ylabel(value_col)
    
    # Format the date axis nicely
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save the figure
    plot_filename = os.path.join(output_dir, f"{value_col}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"  Saved time series plot to {plot_filename}")
    
    return plot_filename

def process_excel_file(file_path, output_dir='./docs', value_col=None):
    """
    Load a specific Excel file, extract time series features,
    and save them as a text file in the output directory.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    output_dir : str
        Directory to save text files
    value_col : str, optional
        Name of the column containing values to analyze. If None, will use the second column.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get the filename without extension
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}.txt")
        
        print(f"Processing {file_name}...")
        
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Identify datetime and value columns
        datetime_col = df.columns[0]  # Assume first column is datetime
        
        # Use specified value column or default to second column
        if value_col is None:
            value_col = df.columns[1]  # Default to second column
        elif value_col not in df.columns:
            print(f"  Error: Specified column '{value_col}' not found in the data")
            return False
        
        print(f"  Identified columns: datetime={datetime_col}, value={value_col}")
        print(f"  Data shape: {df.shape}")
        
        # Convert datetime column if it's not already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                print(f"  Converted {datetime_col} to datetime format")
            except Exception as e:
                print(f"  Error converting {datetime_col} to datetime: {e}")
                return False
        
        # Plot the time series data
        plot_time_series(df, datetime_col, value_col, output_dir, file_name_no_ext)
        
        # Extract features
        features_df = extract_features_from_time_series(
            data=df,
            datetime_col=datetime_col,
            value_col=value_col,
            window_size=5,  # 5-minute window
            step_size=1     # 1-minute step
        )
        
        print(f"  Extracted {len(features_df)} feature windows")
        
        # Save features as text
        save_features_as_text(features_df, output_file)
        
        print(f"  Saved features to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_multiple_files(file_paths, output_dir='./docs', value_cols=None):
    """
    Process multiple Excel files and convert them to text files.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to Excel files
    output_dir : str
        Directory to save text files
    value_cols : dict, optional
        Dictionary mapping file paths to value column names
        Example: {'./labeled_data/battery_status.xlsx': 'battery_percent'}
    """
    success_count = 0
    fail_count = 0
    
    # Initialize value_cols if None
    if value_cols is None:
        value_cols = {}
    
    for file_path in file_paths:
        # Get the specific value column for this file if specified
        value_col = value_cols.get(file_path)
        
        if process_excel_file(file_path, output_dir, value_col):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"Processing complete! Successfully processed {success_count} files.")
    if fail_count > 0:
        print(f"Failed to process {fail_count} files.")

def main():
    """Main function to run the data loader"""
    # Define output directory
    output_dir = './docs'
    
    # List of specific files to process
    file_paths = [
        './labeled_data/battery_status.xlsx',
        './labeled_data/cpu_usage.xlsx',
        './labeled_data/network_traffic.xlsx'
        # Add more file paths as needed
    ]
    
    # Optional: Specify value columns for specific files
    value_cols = {
        './labeled_data/battery_status.xlsx': 'battery_percent',
        './labeled_data/cpu_usage.xlsx': 'cpu_usage',
        # Only specify when you need to override the default behavior
    }
    
    # Process the files with specific value columns
    process_multiple_files(file_paths, output_dir, value_cols)
    
    # Alternatively, process a single file directly with a specific value column
    # process_excel_file('./labeled_data/battery_status.xlsx', output_dir, 'battery_percent')

if __name__ == "__main__":
    main()
