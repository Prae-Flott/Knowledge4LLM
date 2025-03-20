import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import fft
from scipy.signal import find_peaks

def extract_features_from_time_series(data, datetime_col='timestamp', value_col='value', 
                                     window_size=5, step_size=1):
    """
    Extract features from time series data using sliding windows.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the time series data.
    datetime_col : str
        Name of the column containing datetime information.
    value_col : str
        Name of the column containing the values.
    window_size : int
        Size of the sliding window in minutes.
    step_size : int
        Step size for sliding the window in minutes.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the extracted features and their text descriptions.
    """
    # Convert datetime column to datetime type if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Sort data by timestamp
    data = data.sort_values(by=datetime_col)
    
    # Initialize results list
    results = []
    
    # Get start and end times
    start_time = data[datetime_col].min()
    end_time = data[datetime_col].max()
    
    # Generate sliding windows
    current_time = start_time
    while current_time + timedelta(minutes=window_size) <= end_time:
        window_end = current_time + timedelta(minutes=window_size)
        
        # Filter data for current window
        window_data = data[(data[datetime_col] >= current_time) & 
                          (data[datetime_col] < window_end)]
        
        if not window_data.empty:
            # Extract features
            values = window_data[value_col].values
            
            # Basic statistics
            min_val = np.min(values)
            max_val = np.max(values)
            avg_val = np.mean(values)
            std_val = np.std(values)
            
            # Frequency analysis (FFT)
            fft_values = fft.fft(values)
            fft_magnitudes = np.abs(fft_values)
            dominant_freq_idx = np.argmax(fft_magnitudes[1:]) + 1  # Skip DC component

            # Peak detection
            peak_threshold = avg_val + 1.5 * std_val
            peaks, _ = find_peaks(values, height=peak_threshold, distance=3)
            peak_count = len(peaks)
            peak_heights = values[peaks] if peak_count > 0 else np.array([])
            
            # If we have enough data points, calculate the sampling rate
            if len(values) > 1:
                time_diffs = np.diff(window_data[datetime_col].astype(np.int64) // 10**9)
                avg_sampling_period = np.mean(time_diffs)
                sampling_rate = 1 / (avg_sampling_period) if avg_sampling_period > 0 else 0
                dominant_freq = (dominant_freq_idx * sampling_rate / len(values)) if sampling_rate > 0 else 0
            else:
                dominant_freq = 0
            
            # Create text description
            description = (
                f"From {current_time.strftime('%Y-%m-%d %H:%M:%S')} to "
                f"{window_end.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"the minimum value was {min_val:.2f}, "
                f"the maximum value was {max_val:.2f}, "
                f"and the average value was {avg_val:.2f}. "
                f"The dominant frequency component was approximately {dominant_freq:.4f} Hz."
                f" The standard deviation was {std_val:.2f}."
                f"{'There were ' + str(peak_count) + ' peaks detected' if peak_count > 0 else 'No peaks were detected'}."
                f"{'The highest peak had a value of ' + f'{np.max(peak_heights):.2f}' + ' and occurred at ' + window_data.iloc[peaks[np.argmax(peak_heights)]][datetime_col].strftime('%H:%M:%S') if peak_count > 0 else ''}"
            )
            
            # Store results
            results.append({
                'window_start': current_time,
                'window_end': window_end,
                'min_value': min_val,
                'max_value': max_val,
                'avg_value': avg_val,
                'std_value': std_val,
                'peak_count': peak_count,
                'dominant_frequency': dominant_freq,
                'description': description
            })
        
        # Slide window
        current_time += timedelta(minutes=step_size)
    
    return pd.DataFrame(results)

def save_features_as_text(features_df, output_file='time_series_features.txt'):
    """
    Save extracted features as text.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing the features and descriptions.
    output_file : str
        Path to the output text file.
    """
    with open(output_file, 'w') as f:
        for _, row in features_df.iterrows():
            f.write(row['description'] + '\n\n')
    print(f"Features saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Create sample time series data
    # In a real scenario, you would load your data from a file or database
    np.random.seed(42)
    
    # Generate 1 hour of data with readings every 10 seconds
    base_time = datetime.now().replace(microsecond=0, second=0, minute=0)
    timestamps = [base_time + timedelta(seconds=10*i) for i in range(360)]  # 1 hour = 360 * 10 seconds
    
    # Generate sine wave with some noise
    t = np.linspace(0, 2*np.pi, 360)
    values = 10 * np.sin(t) + np.random.normal(0, 1, 360)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    print(f"Sample data created with {len(data)} points spanning {timestamps[0]} to {timestamps[-1]}")
    
    # Extract features
    features = extract_features_from_time_series(data)
    
    # Display features
    print(f"Extracted {len(features)} feature windows")
    print(features[['window_start', 'window_end', 'min_value', 'max_value', 'avg_value']].head())
    
    # Save as text
    save_features_as_text(features)
    
    # Print examples
    print("\nExample descriptions:")
    for desc in features['description'].head(3):
        print(desc)
        print("-" * 80)