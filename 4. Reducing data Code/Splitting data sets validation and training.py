import os
import pandas as pd

# Define relative path to your folder
DATA_DIR = os.path.join('.', '5. Reduced Data')

print("Processing Room 869 datasets...")

# Load Room 869 overlap dataset
df_869 = pd.read_csv(os.path.join(DATA_DIR, 'sensor_869_occupancy_overlap.csv'))

# Clean headers
df_869.columns = df_869.columns.str.strip().str.replace('\ufeff', '')

# Parse and sort timestamps
df_869['dt'] = pd.to_datetime(df_869['timestamp'].astype(str).str.slice(0, 19))
df_869 = df_869.sort_values('dt').reset_index(drop=True)

# --- File 1: Ground-Truth Occupancy (Room 869) ---
occ_869 = df_869[['dt', 'headcount']].copy()
occ_869.to_csv(os.path.join(DATA_DIR, '869_occupancy.csv'), index=False)
print(" -> Saved '869_occupancy.csv'")

# --- Files 2 & 3: Environmental Features (Filtered to CO2, Temp, LVOC) ---
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


print("\nProcessing Room 326 datasets...")

# Load Room 326 environmental dataset
df_326 = pd.read_csv(os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv'))
df_326.columns = df_326.columns.str.strip().str.replace('\ufeff', '')

date_col_326 = next(col for col in ['dt', 'timestamp', 'collecteddate'] if col in df_326.columns)
df_326['dt'] = pd.to_datetime(df_326[date_col_326].astype(str).str.slice(0, 19))
df_326 = df_326.sort_values('dt').reset_index(drop=True)

# --- File 4: Ground-Truth Occupancy (Room 326) ---
# Using occupancy_data_2026_only.csv from your folder
df_occ_full = pd.read_csv(os.path.join(DATA_DIR, 'occupancy_data_2026_only.csv'))
df_occ_full.columns = df_occ_full.columns.str.strip().str.replace('\ufeff', '')

occ_date_col = next(col for col in ['dt', 'timestamp', 'collecteddate'] if col in df_occ_full.columns)
df_occ_full['dt'] = pd.to_datetime(df_occ_full[occ_date_col].astype(str).str.slice(0, 19))

occ_326 = df_occ_full.sort_values('dt')[['dt', 'headcount']].copy()
occ_326.to_csv(os.path.join(DATA_DIR, '326_occupancy.csv'), index=False)
print(" -> Saved '326_occupancy.csv'")

# --- Files 5 & 6: Environmental Features (Filtered to CO2, Temp, LVOC) ---
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

print("\nAll files created successfully inside '5. Reduced Data'!")