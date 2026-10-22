"""Environmental Telemetry Train and Validation Dataset Partitioning Pipeline.

This module processes the master tabular environmental sensor dataset, filters for the
empirically validated active deployment lifecycles of Room 869 (mid-March to mid-June 2024)
and Room 326 (October 2025 to end of April 2026), and partitions them into chronological
two-thirds training and one-third validation splits.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "4. Reducing data Code/Splitting data sets validation and training.py"
    ```
"""

import os
from typing import List, Optional

import pandas as pd

# Define paths robustly relative to repository root
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')
DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')


def parse_timestamp_series(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and parses timestamp strings into ISO-standard datetime format.

    Scans column headers for known timestamp aliases, extracts the 19-character
    ISO string representation ('YYYY-MM-DD HH:MM:SS'), and sorts records chronologically.

    Args:
        df (pd.DataFrame): Raw DataFrame containing telemetry records.

    Returns:
        pd.DataFrame: DataFrame with standardized 'dt' column, sorted by time.

    Raises:
        KeyError: If no recognized datetime column exists in the DataFrame headers.

    Examples:
        >>> parsed_df = parse_timestamp_series(raw_df)
    """
    possible_names: List[str] = ['timestamp', 'dt', 'collecteddate', 'time', 'date', 'datetime']
    
    target_col: Optional[str] = next((col for col in df.columns if col.strip().lower() in possible_names), None)
            
    if target_col is None:
        raise KeyError(f"Could not find timestamp column in headers: {list(df.columns)}")
        
    # Extract 19-char string slice ('YYYY-MM-DD HH:MM:SS') before parsing to ignore timezone offsets
    df['dt'] = pd.to_datetime(df[target_col].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').reset_index(drop=True)


def main() -> None:
    """Executes filtering and chronological train-validation splitting for Rooms 869 and 326.

    Slices active physical deployment windows, strips sensor flatline artifacts,
    applies strict chronological 2/3 training and 1/3 validation splits to preserve
    temporal causality, and exports cleaned subsets to '5. Reduced Data/'.

    Returns:
        None
    """
    print("=" * 80)
    print("SPLITTING ENVIRONMENTAL DATASETS (TRAIN & VALIDATION)")
    print("=" * 80)

    raw_env_file = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')
    if not os.path.exists(raw_env_file):
        print(f"Environmental dataset not found at: {raw_env_file}")
        return

    print(f"Loading environmental dataset: {raw_env_file}...")
    df_raw = pd.read_csv(raw_env_file)
    df_raw.columns = df_raw.columns.str.strip().str.replace('\ufeff', '')
    df_raw = parse_timestamp_series(df_raw)

    df_raw['sensor_id'] = df_raw['sensor_id'].astype(str)

    # 1. Process Room 869 (Mid-March to Mid-June 2024 active window)
    print("\nProcessing Room 869 datasets...")
    # Exclude post-June flatline at 520.0 ppm caused by sensor ADC saturation
    mask_869 = (
        (df_raw['sensor_id'] == '869') & 
        (df_raw['dt'] >= '2024-03-15') & 
        (df_raw['dt'] <= '2024-06-14 23:59:59') & 
        (df_raw['co2_ppm'] != 520.0)
    )
    df_869 = df_raw[mask_869].sort_values('dt').reset_index(drop=True)

    env_cols: List[str] = ['dt', 'co2_ppm', 'temperature_c', 'lvoc_ppb']
    df_869_env = df_869[[c for c in env_cols if c in df_869.columns]].copy()

    # Chronological split: first 2/3 for training, remaining 1/3 for validation
    split_idx_869 = int(len(df_869_env) * (2 / 3))
    env_869_train = df_869_env.iloc[:split_idx_869].copy()
    env_869_val = df_869_env.iloc[split_idx_869:].copy()

    env_869_train.to_csv(os.path.join(DATA_DIR, '869_train.csv'), index=False)
    env_869_val.to_csv(os.path.join(DATA_DIR, '869_validation.csv'), index=False)

    print(f" -> Saved '869_train.csv' ({len(env_869_train):,} rows | {env_869_train['dt'].min()} to {env_869_train['dt'].max()})")
    print(f" -> Saved '869_validation.csv' ({len(env_869_val):,} rows | {env_869_val['dt'].min()} to {env_869_val['dt'].max()})")

    # 2. Process Room 326 (October 2025 to end of April 2026 active window)
    print("\nProcessing Room 326 datasets...")
    # Exclude 2024 dormant period and flatlined 525.0 ppm records
    mask_326 = (
        (df_raw['sensor_id'] == '326') & 
        (df_raw['dt'] >= '2025-10-01') & 
        (df_raw['dt'] <= '2026-04-22 23:59:59') & 
        (df_raw['co2_ppm'] != 525.0)
    )
    df_326 = df_raw[mask_326].sort_values('dt').reset_index(drop=True)

    df_326_env = df_326[[c for c in env_cols if c in df_326.columns]].copy()

    split_idx_326 = int(len(df_326_env) * (2 / 3))
    env_326_train = df_326_env.iloc[:split_idx_326].copy()
    env_326_val = df_326_env.iloc[split_idx_326:].copy()

    env_326_train.to_csv(os.path.join(DATA_DIR, '326_train.csv'), index=False)
    env_326_val.to_csv(os.path.join(DATA_DIR, '326_validation.csv'), index=False)

    print(f" -> Saved '326_train.csv' ({len(env_326_train):,} rows | {env_326_train['dt'].min()} to {env_326_train['dt'].max()})")
    print(f" -> Saved '326_validation.csv' ({len(env_326_val):,} rows | {env_326_val['dt'].min()} to {env_326_val['dt'].max()})")
    print("\nEnvironmental train/validation datasets successfully generated!")


if __name__ == '__main__':
    main()
