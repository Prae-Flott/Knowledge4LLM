import json
from datetime import datetime, timedelta

def split_data_by_count(data, chunk_size):
    """
    Splits the list of time-series data into chunks of a given fixed size.
    
    Parameters:
        data (list): List of dictionary records.
        chunk_size (int): Number of records per chunk.
        
    Returns:
        List of chunks, where each chunk is a list of dictionary records.
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def split_data_by_time(data, time_interval_minutes):
    """
    Splits the time-series data into chunks based on a time interval.
    All records whose timestamps fall within the same time interval (in minutes)
    are grouped together in one chunk.
    
    Parameters:
        data (list): List of dictionary records with a "Timestamp" field in the format "%Y-%m-%d %H:%M:%S".
        time_interval_minutes (int): The time interval (in minutes) used to group records.
    
    Returns:
        List of chunks, where each chunk is a list of dictionary records.
    """
    if not data:
        return []

    # Define the time format
    time_format = "%Y-%m-%d %H:%M:%S"
    interval = timedelta(minutes=time_interval_minutes)
    
    chunks = []
    # Initialize the first chunk with the first record.
    current_chunk = [data[0]]
    current_start = datetime.strptime(data[0]["Timestamp"], time_format)

    # Loop through remaining records
    for record in data[1:]:
        ts = datetime.strptime(record["Timestamp"], time_format)
        # If the record falls within the current interval, add to current_chunk
        if ts - current_start <= interval:
            current_chunk.append(record)
        else:
            # Otherwise, save the current chunk and start a new one.
            chunks.append(current_chunk)
            current_chunk = [record]
            current_start = ts

    # Append the last chunk if it exists.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
