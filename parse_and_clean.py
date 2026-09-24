import csv
import json
import os
import time
import pandas as pd
import numpy as np

csv_path = r'c:\Users\gregr\Documents\Year_3\MMA3001\5EnvSensor_MayToDec2024_180kRows.csv'
out_csv = r'C:\Users\gregr\.gemini\antigravity\brain\fa95884d-3e0c-4586-8e63-03ba1bc71dea\scratch\cleaned_sensor_data.csv'

print("Starting extraction and flattening of 180k rows...")
t0 = time.time()

# Mapping variable name to clean column name
var_map = {
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

records = []
total_parsed = 0
errors = 0

with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    # id(0), sensorid(1), devicetype(2), jsondata(3), status(4), processing_errors(5), createdate(6), processdate(7)
    for row in reader:
        total_parsed += 1
        row_id = row[0]
        sensor_id = row[1]
        createdate = row[6]
        json_str = row[3]
        
        row_dict = {
            'row_id': int(row_id),
            'sensor_id': str(sensor_id),
            'timestamp': createdate[:19], # YYYY-MM-DD HH:MM:SS
            'tz_offset': createdate[26:] if len(createdate) > 26 else createdate[19:],
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
            items = json.loads(json_str)
            for it in items:
                vname = it.get('variable', {}).get('name')
                col = var_map.get(vname)
                if col:
                    vals = it.get('values', [])
                    if vals and 'value' in vals[0]:
                        row_dict[col] = float(vals[0]['value'])
        except Exception as e:
            errors += 1
            
        records.append(row_dict)
        if total_parsed % 50000 == 0:
            print(f"Processed {total_parsed} rows in {time.time()-t0:.1f}s...")

df = pd.DataFrame(records)
print(f"Extraction completed in {time.time()-t0:.1f}s. Total rows: {len(df)}, JSON errors: {errors}")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['timestamp', 'sensor_id']).reset_index(drop=True)

# Save cleaned CSV
df.to_csv(out_csv, index=False)
print(f"Saved cleaned CSV to {out_csv} ({os.path.getsize(out_csv) / (1024*1024):.2f} MB)")

print("\n--- Summary per sensor ---")
sensor_summary = df.groupby('sensor_id').agg(
    record_count=('row_id', 'count'),
    min_time=('timestamp', 'min'),
    max_time=('timestamp', 'max')
)
print(sensor_summary.to_string())

print("\n--- Non-null counts per variable ---")
print(df.notnull().sum())
