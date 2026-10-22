"""Single-Pass Streaming Extraction Pipeline for Large Occupancy Event Logs.

This module streams the massive 9.09-million row raw camera occupancy event log
(`5occupancySensor_MayToDec2024_9MRows.csv`) in memory-efficient chunks of 1,000,000 rows.
It extracts ground-truth occupancy events matching the active deployment windows of
Room 869 and Room 326 without exceeding system RAM limits.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "4. Reducing data Code/Occupancy data set creation.py"
    ```
"""

import os
from typing import Optional, Tuple

import pandas as pd

# Path configuration robust across working directories
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')

DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
RAW_DIR: str = os.path.join(PROJECT_ROOT, '1. Given Data')
RAW_OCCUPANCY_FILE: str = os.path.join(RAW_DIR, '5occupancySensor_MayToDec2024_9MRows.csv')


def get_sensor_time_bounds(
    sensor_id: str, 
    default_start: Optional[str] = None, 
    default_end: Optional[str] = None
) -> Tuple[str, str]:
    """Retrieves minimum and maximum bounding timestamps from training and validation datasets.

    Args:
        sensor_id (str): Sensor identifier string (e.g., '869' or '326').
        default_start (Optional[str], optional): Fallback start timestamp. Defaults to None.
        default_end (Optional[str], optional): Fallback end timestamp. Defaults to None.

    Returns:
        Tuple[str, str]: Chronological (min_timestamp, max_timestamp) strings.

    Raises:
        FileNotFoundError: If train or validation CSV files are absent and fallbacks are omitted.
        KeyError: If no recognized timestamp column is present in the CSV files.

    Examples:
        >>> start_ts, end_ts = get_sensor_time_bounds('869')
    """
    train_path = os.path.join(DATA_DIR, f"{sensor_id}_train.csv")
    val_path = os.path.join(DATA_DIR, f"{sensor_id}_validation.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        if default_start and default_end:
            print(f"Train/val files for {sensor_id} not found. Using fallback bounds: {default_start} to {default_end}")
            return default_start, default_end
        raise FileNotFoundError(f"Missing train or validation file for Sensor {sensor_id} in {DATA_DIR}.")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    time_cols = ['dt', 'timestamp', 'collecteddate', 'occupancystatuschangedate', 'time', 'date']
    t_train = next((c for c in df_train.columns if c.strip().lower() in time_cols), None)
    t_val = next((c for c in df_val.columns if c.strip().lower() in time_cols), None)

    if not t_train or not t_val:
        raise KeyError(f"Could not identify datetime column in {sensor_id} train/val files.")

    combined_ts = pd.concat([df_train[t_train], df_val[t_val]]).astype(str).str.slice(0, 19)
    min_ts = str(combined_ts.min())
    max_ts = str(combined_ts.max())
    
    return min_ts, max_ts


def create_occupancy_datasets() -> None:
    """Streams raw 9M-row occupancy logs and extracts aligned subsets for Rooms 869 and 326.

    Streams chunks of 1M rows using low memory allocation, filters event timestamps
    against active sensor windows, and writes clean tabular outputs to '5. Reduced Data/'.

    Returns:
        None

    Raises:
        FileNotFoundError: If the raw 9M-row occupancy CSV is missing from '1. Given Data/'.
    """
    print("=" * 80)
    print("OCCUPANCY DATASET EXTRACTION PIPELINE")
    print("=" * 80)
    
    if not os.path.exists(RAW_OCCUPANCY_FILE):
        print(f"Notice: Raw occupancy CSV not found at: {RAW_OCCUPANCY_FILE}. Skipping stream extraction.")
        return

    # Determine exact operational date ranges for extraction
    start_869, end_869 = get_sensor_time_bounds('869', '2024-03-15 00:00:14', '2024-06-14 03:43:04')
    start_326, end_326 = get_sensor_time_bounds('326', '2025-10-01 00:01:03', '2026-04-22 23:20:33')

    print(f"Target Time Window for Sensor 869: {start_869}  -->  {end_869}")
    print(f"Target Time Window for Sensor 326: {start_326}  -->  {end_326}")
    print(f"Source file: {RAW_OCCUPANCY_FILE}")
    print("-" * 80)

    cols_to_load = ['occupancystatuschangedate', 'id', 'headcount', 'occupancystatus', 'floorspaceid']
    out_869_path = os.path.join(DATA_DIR, '869_occupancy.csv')
    out_326_path = os.path.join(DATA_DIR, '326_occupancy.csv')

    chunks_869 = []
    total_rows_869 = 0
    total_rows_326 = 0
    first_chunk_326 = True
    processed_rows = 0
    chunksize = 1_000_000

    print("Beginning single-pass streaming of raw occupancy data in chunks of 1M rows...")

    for chunk in pd.read_csv(RAW_OCCUPANCY_FILE, usecols=cols_to_load, chunksize=chunksize, low_memory=False):
        processed_rows += len(chunk)
        # Fast string slice ('YYYY-MM-DD HH:MM:SS') avoids parsing billions of subsecond datetime strings
        ts_slice = chunk['occupancystatuschangedate'].astype(str).str.slice(0, 19)

        # 1. Filter for Room 869
        mask_869 = (ts_slice >= start_869) & (ts_slice <= end_869)
        if mask_869.any():
            sub_869 = chunk.loc[mask_869, cols_to_load]
            chunks_869.append(sub_869)
            total_rows_869 += len(sub_869)

        # 2. Filter for Room 326 (Append directly to disk to minimize memory consumption)
        mask_326 = (ts_slice >= start_326) & (ts_slice <= end_326)
        if mask_326.any():
            sub_326 = chunk.loc[mask_326, cols_to_load]
            sub_326.to_csv(out_326_path, mode='w' if first_chunk_326 else 'a', 
                           header=first_chunk_326, index=False)
            first_chunk_326 = False
            total_rows_326 += len(sub_326)

        print(f"Processed {processed_rows:,} raw rows... [869 matches: {total_rows_869:,} | 326 matches: {total_rows_326:,}]")

    print("\nFinalizing Room 869 occupancy file...")
    if chunks_869:
        df_869 = pd.concat(chunks_869, ignore_index=True)
        df_869 = df_869.sort_values(by=['occupancystatuschangedate', 'id']).reset_index(drop=True)
        df_869.to_csv(out_869_path, index=False)
        print(f" -> Saved: {out_869_path} ({len(df_869):,} rows)")
    else:
        print("Notice: No matching records found for Room 869.")

    if os.path.exists(out_326_path):
        print(f"\nFinalized Room 326 occupancy file: {out_326_path} ({total_rows_326:,} rows)")

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    create_occupancy_datasets()
