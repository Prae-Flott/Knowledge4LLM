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

def load_data(file_path, output_dir='./docs', value_col=None):
    """
    Load a specific Excel file, extract time series features,
    save them as a text file, and return the last description.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    output_dir : str
        Directory to save text files
    value_col : str, optional
        Name of the column containing values to analyze. If None, will use the second column.
        
    Returns:
    --------
    str or None:
        The last description if successful, None otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get the filename without extension
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}_{value_col}.txt")
        
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
            return None
        
        print(f"  Identified columns: datetime={datetime_col}, value={value_col}")
        print(f"  Data shape: {df.shape}")
        
        # Convert datetime column if it's not already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                print(f"  Converted {datetime_col} to datetime format")
            except Exception as e:
                print(f"  Error converting {datetime_col} to datetime: {e}")
                return None
        
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
        
        # Get the last description
        if not features_df.empty:
            last_description = features_df.iloc[-1]['description']
        else:
            last_description = f"No features were extracted from {file_name}"
        
        # Save features as text
        save_features_as_text(features_df, output_file)
        print(f"  Saved features to {output_file}")
        
        return last_description
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def read_latest_description(file_name, docs_dir='./labeled_data'):
    """
    Read the latest description from a saved text file for a given value column.
    
    Parameters:
    -----------
    file_name_no_ext : str
        Filename without extension (e.g., 'battery_status')
    value_col : str
        Name of the value column (e.g., 'battery_percent')
    docs_dir : str
        Directory where the text files are saved
        
    Returns:
    --------
    str or None:
        The latest description (last paragraph) from the file if found, None otherwise
    """
    try:
        # Construct the expected file path
        file_path = os.path.join(docs_dir, f"{file_name}.txt")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs (descriptions)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Return the last paragraph if available
        if paragraphs:
            return paragraphs[-1]
        else:
            print(f"No descriptions found in {file_path}")
            return None
    
    except Exception as e:
        print(f"Error reading latest description: {e}")
        return None

def main():
    """
    Demo function to test the data loader with a single file.
    """
    # Define output directory and file path
    output_dir = './labeled_data'
    file_path = './labeled_data/wheel_wrapped.xlsx'
    value_col = 'pwm_right'  # Optional, set to None to use default
    
    # Process the file and get the last description
    last_description = load_data(file_path, output_dir, value_col)
    
    if last_description:
        print("\nLast description:")
        print("-" * 80)
        print(last_description)
        print("-" * 80)
        
        # # Demonstrate how to read the latest description from the saved file
        # file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        # read_description = read_latest_description(file_name_no_ext, value_col, output_dir)
        
        # print("\nReading latest description from saved file:")
        # print("-" * 80)
        # print(read_description)
        # print("-" * 80)
        
        # print("Processing complete!")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    main()
