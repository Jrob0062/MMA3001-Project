import os
import pandas as pd

# Define relative path to your data subfolder
DATA_DIR = os.path.join('.', '5. Reduced Data')

def parse_timestamp_series(df):
    """Finds and parses the timestamp column robustly into standard datetime format."""
    possible_names = ['timestamp', 'dt', 'collecteddate', 'time', 'date', 'datetime']
    
    # Identify matching column
    target_col = None
    for col in df.columns:
        if col.strip().lower() in possible_names:
            target_col = col
            break
            
    if target_col is None:
        raise KeyError(f"Could not find timestamp column in headers: {list(df.columns)}")
        
    # Extract 'YYYY-MM-DD HH:MM:SS' (first 19 characters) to drop tz_offset or trailing artifacts
    df['dt'] = pd.to_datetime(df[target_col].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').reset_index(drop=True)


# ==============================================================================
# 1. PROCESS ROOM 869
# ==============================================================================
print("Processing Room 869 datasets...")

df_869 = pd.read_csv(os.path.join(DATA_DIR, 'sensor_869_occupancy_overlap.csv'))
df_869.columns = df_869.columns.str.strip().str.replace('\ufeff', '')

# Parse datetime and sort
df_869 = parse_timestamp_series(df_869)

# Save occupancy dataset
if 'headcount' in df_869.columns:
    occ_869 = df_869[['dt', 'headcount']].copy()
    occ_869.to_csv(os.path.join(DATA_DIR, '869_occupancy.csv'), index=False)
    print(" -> Saved '869_occupancy.csv'")

# Filter environmental columns
env_cols_869 = ['dt', 'co2_ppm', 'temperature_c', 'lvoc_ppb']
available_cols_869 = [c for c in env_cols_869 if c in df_869.columns]
df_869_env = df_869[available_cols_869].copy()

# Chronological 2/3 split
split_idx_869 = int(len(df_869_env) * (2 / 3))

env_869_train = df_869_env.iloc[:split_idx_869].copy()
env_869_val = df_869_env.iloc[split_idx_869:].copy()

env_869_train.to_csv(os.path.join(DATA_DIR, '869_train.csv'), index=False)
env_869_val.to_csv(os.path.join(DATA_DIR, '869_validation.csv'), index=False)
print(" -> Saved '869_train.csv' (First 2/3)")
print(" -> Saved '869_validation.csv' (Last 1/3)")


# ==============================================================================
# 2. PROCESS ROOM 326
# ==============================================================================
print("\nProcessing Room 326 datasets...")

df_326 = pd.read_csv(os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv'))
df_326.columns = df_326.columns.str.strip().str.replace('\ufeff', '')
df_326 = parse_timestamp_series(df_326)

# Occupancy file
df_occ_full = pd.read_csv(os.path.join(DATA_DIR, 'occupancy_data_2026_only.csv'))
df_occ_full.columns = df_occ_full.columns.str.strip().str.replace('\ufeff', '')
df_occ_full = parse_timestamp_series(df_occ_full)

if 'headcount' in df_occ_full.columns:
    occ_326 = df_occ_full[['dt', 'headcount']].copy()
    occ_326.to_csv(os.path.join(DATA_DIR, '326_occupancy.csv'), index=False)
    print(" -> Saved '326_occupancy.csv'")

# Filter environmental columns
env_cols_326 = ['dt', 'co2_ppm', 'temperature_c', 'lvoc_ppb']
available_cols_326 = [c for c in env_cols_326 if c in df_326.columns]
df_326_env = df_326[available_cols_326].copy()

# Chronological 2/3 split
split_idx_326 = int(len(df_326_env) * (2 / 3))

env_326_train = df_326_env.iloc[:split_idx_326].copy()
env_326_val = df_326_env.iloc[split_idx_326:].copy()

env_326_train.to_csv(os.path.join(DATA_DIR, '326_train.csv'), index=False)
env_326_val.to_csv(os.path.join(DATA_DIR, '326_validation.csv'), index=False)
print(" -> Saved '326_train.csv' (First 2/3)")
print(" -> Saved '326_validation.csv' (Last 1/3)")

print("\nProcessing complete!")