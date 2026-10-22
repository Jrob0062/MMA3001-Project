"""Sensor 869 Dataset Extraction and Active Window Filter Utility.

This module extracts subsets for Sensor 869 from the master cleaned tabular dataset,
generating historical full records, valid non-saturated operational windows, and
camera overlap subsets.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "4. Reducing data Code/split 869 data.py"
    ```
"""

import os
from typing import Optional

import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')
DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
DEFAULT_INPUT_FILE: str = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')


def extract_sensor_869_datasets(
    input_filepath: str = DEFAULT_INPUT_FILE
) -> Optional[pd.DataFrame]:
    """Filters and exports operational windows for Sensor 869.

    Args:
        input_filepath (str, optional): Path to cleaned tabular environmental CSV.
            Defaults to DEFAULT_INPUT_FILE.

    Returns:
        Optional[pd.DataFrame]: DataFrame of the valid active window if found, else None.

    Raises:
        FileNotFoundError: If the input CSV does not exist.

    Examples:
        >>> df_active = extract_sensor_869_datasets("data.csv")
    """
    if not os.path.exists(input_filepath):
        print(f"Notice: Input dataset not found at: {input_filepath}. Skipping.")
        return None

    print(f"Loading environmental dataset from: {input_filepath}...")
    df = pd.read_csv(input_filepath)
    df['dt'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    # 1. Full dataset for Sensor 869
    s869_full = df[df['sensor_id'].astype(str) == '869'].copy()
    
    # 2. Valid active window (excluding the flatlined 520.0 ppm CO2 sensor failure)
    s869_valid = df[(df['sensor_id'].astype(str) == '869') & (df['co2_ppm'] != 520.0)].copy()
    
    # 3. Ground-truth occupancy overlap window
    s869_overlap = df[
        (df['sensor_id'].astype(str) == '869') & 
        (df['dt'] >= '2024-03-01') & 
        (df['dt'] <= '2024-05-06')
    ].copy()
    
    out_overlap = os.path.join(DATA_DIR, 'sensor_869_occupancy_overlap.csv')
    out_valid = os.path.join(DATA_DIR, 'sensor_869_valid_window.csv')
    out_full = os.path.join(DATA_DIR, 'sensor_869_full.csv')

    s869_overlap.drop(columns=['dt']).to_csv(out_overlap, index=False)
    s869_valid.drop(columns=['dt']).to_csv(out_valid, index=False)
    s869_full.drop(columns=['dt']).to_csv(out_full, index=False)
    
    print("\nFiles successfully created in '5. Reduced Data/':")
    print(f" - sensor_869_occupancy_overlap.csv ({len(s869_overlap):,} rows)")
    print(f" - sensor_869_valid_window.csv      ({len(s869_valid):,} rows)")
    print(f" - sensor_869_full.csv              ({len(s869_full):,} rows)")
    return s869_valid


def main() -> None:
    """Executes the extraction pipeline for Sensor 869.

    Returns:
        None
    """
    extract_sensor_869_datasets()


if __name__ == '__main__':
    main()
