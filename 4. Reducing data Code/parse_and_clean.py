"""JSON Telemetry Extraction and Tabular Flattening Pipeline.

This module parses raw environmental sensor telemetry containing nested JSON payloads
(`5EnvSensor_MayToDec2024_180kRows.csv`) and unpacks physical environmental channels
(CO2, ambient temperature, volatile organics, particulate matter, humidity, pressure)
into a structured tabular CSV dataset.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "4. Reducing data Code/parse_and_clean.py"
    ```
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')

DEFAULT_RAW_FILE: str = os.path.join(PROJECT_ROOT, '1. Given Data', '5EnvSensor_MayToDec2024_180kRows.csv')
DEFAULT_OUT_FILE: str = os.path.join(PROJECT_ROOT, '5. Reduced Data', '5EnvSensor_Cleaned_Tabular.csv')

VARIABLE_MAPPING: Dict[str, str] = {
    'Battery': 'battery_v',
    'Carbon dioxide': 'co2_ppm',
    'Humidity': 'humidity_pct',
    'Formaldehyde': 'formaldehyde_ugm3',
    'Pressure': 'pressure_mb',
    'Temperature': 'temperature_c',
    'Light Volatile Organic Compounds': 'lvoc_ppb',
    'Particulate matter 1': 'pm1_ugm3',
    'Particulate matter 2.5': 'pm25_ugm3',
    'Particulate matter 4': 'pm4_ugm3',
    'Particulate matter 10': 'pm10_ugm3'
}


def parse_nested_telemetry(
    input_filepath: str, 
    output_filepath: str
) -> Optional[pd.DataFrame]:
    """Flattens nested JSON sensor telemetry strings into clean tabular CSV columns.

    Args:
        input_filepath (str): Path to raw CSV containing 'jsondata' payload strings.
        output_filepath (str): Destination path for the flattened tabular dataset.

    Returns:
        Optional[pd.DataFrame]: Cleaned DataFrame if file exists, else None.

    Raises:
        FileNotFoundError: If the input raw telemetry CSV does not exist.

    Examples:
        >>> df_cleaned = parse_nested_telemetry("raw_telemetry.csv", "flattened.csv")
    """
    if not os.path.exists(input_filepath):
        print(f"Notice: Raw telemetry CSV not found at: {input_filepath}. Skipping parsing.")
        return None

    print(f"Starting extraction and flattening of raw rows from: {input_filepath}")
    t0 = time.time()
    records: List[Dict] = []
    total_parsed = 0
    errors = 0

    with open(input_filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            total_parsed += 1
            if len(row) < 7:
                continue
            row_id = row[0]
            sensor_id = row[1]
            json_str = row[3]
            createdate = row[6]
            
            row_dict = {
                'row_id': int(row_id) if row_id.isdigit() else total_parsed,
                'sensor_id': str(sensor_id),
                'timestamp': createdate[:19],
                'battery_v': np.nan,
                'co2_ppm': np.nan,
                'humidity_pct': np.nan,
                'formaldehyde_ugm3': np.nan,
                'pressure_mb': np.nan,
                'temperature_c': np.nan,
                'lvoc_ppb': np.nan,
                'pm1_ugm3': np.nan,
                'pm25_ugm3': np.nan,
                'pm4_ugm3': np.nan,
                'pm10_ugm3': np.nan
            }

            try:
                data_items = json.loads(json_str)
                if isinstance(data_items, list):
                    for item in data_items:
                        v_name = item.get('VariableName')
                        v_val = item.get('Value')
                        if v_name in VARIABLE_MAPPING and v_val is not None:
                            try:
                                row_dict[VARIABLE_MAPPING[v_name]] = float(v_val)
                            except (ValueError, TypeError):
                                pass
            except (json.JSONDecodeError, TypeError):
                errors += 1

            records.append(row_dict)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed:.2f}s. Parsed: {total_parsed:,} rows (JSON errors: {errors}).")
    print(f"Saved tabular dataset to: {output_filepath}")
    return df


def main() -> None:
    """Executes the raw environmental JSON flattening pipeline.

    Returns:
        None
    """
    parse_nested_telemetry(DEFAULT_RAW_FILE, DEFAULT_OUT_FILE)


if __name__ == '__main__':
    main()
