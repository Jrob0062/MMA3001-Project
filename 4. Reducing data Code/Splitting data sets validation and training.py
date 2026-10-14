import os
import pandas as pd

# Define paths robustly whether run from repo root or script folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_ROOT, '5. Reduced Data')

def parse_timestamp_series(df):
    """Finds and parses the timestamp column robustly into standard datetime format."""
    possible_names = ['timestamp', 'dt', 'collecteddate', 'time', 'date', 'datetime']
    
    target_col = None
    for col in df.columns:
        if col.strip().lower() in possible_names:
            target_col = col
            break
            
    if target_col is None:
        raise KeyError(f"Could not find timestamp column in headers: {list(df.columns)}")
        
    # Extract 'YYYY-MM-DD HH:MM:SS' (first 19 characters)
    df['dt'] = pd.to_datetime(df[target_col].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').reset_index(drop=True)


def main():
    print("=" * 80)
    print("SPLITTING ENVIRONMENTAL DATASETS (TRAIN & VALIDATION)")
    print("=" * 80)

    # 1. Load Master Cleaned Environmental Tabular Data
    raw_env_file = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')
    print(f"Loading environmental dataset: {raw_env_file}...")
    df_raw = pd.read_csv(raw_env_file)
    df_raw.columns = df_raw.columns.str.strip().str.replace('\ufeff', '')
    df_raw = parse_timestamp_series(df_raw)

    # Ensure sensor_id is string for consistent comparison
    df_raw['sensor_id'] = df_raw['sensor_id'].astype(str)

    # ==============================================================================
    # 1. PROCESS ROOM 869 (Mid-March to Mid-June 2024 active window)
    # ==============================================================================
    print("\nProcessing Room 869 datasets...")
    # Active window before CO2 sensor flatlined at 520 ppm in mid-June
    mask_869 = (
        (df_raw['sensor_id'] == '869') & 
        (df_raw['dt'] >= '2024-03-15') & 
        (df_raw['dt'] <= '2024-06-14 23:59:59') & 
        (df_raw['co2_ppm'] != 520.0)
    )
    df_869 = df_raw[mask_869].sort_values('dt').reset_index(drop=True)

    # Filter environmental feature columns
    env_cols = ['dt', 'co2_ppm', 'temperature_c', 'lvoc_ppb']
    df_869_env = df_869[[c for c in env_cols if c in df_869.columns]].copy()

    # Chronological 2/3 train, 1/3 validation split
    split_idx_869 = int(len(df_869_env) * (2 / 3))
    env_869_train = df_869_env.iloc[:split_idx_869].copy()
    env_869_val = df_869_env.iloc[split_idx_869:].copy()

    env_869_train.to_csv(os.path.join(DATA_DIR, '869_train.csv'), index=False)
    env_869_val.to_csv(os.path.join(DATA_DIR, '869_validation.csv'), index=False)

    print(f" -> Saved '869_train.csv' ({len(env_869_train):,} rows | {env_869_train['dt'].min()} to {env_869_train['dt'].max()})")
    print(f" -> Saved '869_validation.csv' ({len(env_869_val):,} rows | {env_869_val['dt'].min()} to {env_869_val['dt'].max()})")
    print(f" -> Total 869 Coverage: {len(df_869_env):,} rows ({df_869_env['dt'].min()} to {df_869_env['dt'].max()})")

    # ==============================================================================
    # 2. PROCESS ROOM 326 (October 2025 to end of April 2026 active window)
    # ==============================================================================
    print("\nProcessing Room 326 datasets...")
    # Active redeployment period (excluding the inactive/flatlined 525 ppm 2024 data)
    mask_326 = (
        (df_raw['sensor_id'] == '326') & 
        (df_raw['dt'] >= '2025-10-01') & 
        (df_raw['dt'] <= '2026-04-22 23:59:59') & 
        (df_raw['co2_ppm'] != 525.0)
    )
    df_326 = df_raw[mask_326].sort_values('dt').reset_index(drop=True)

    # Filter environmental feature columns
    df_326_env = df_326[[c for c in env_cols if c in df_326.columns]].copy()

    # Chronological 2/3 train, 1/3 validation split
    split_idx_326 = int(len(df_326_env) * (2 / 3))
    env_326_train = df_326_env.iloc[:split_idx_326].copy()
    env_326_val = df_326_env.iloc[split_idx_326:].copy()

    env_326_train.to_csv(os.path.join(DATA_DIR, '326_train.csv'), index=False)
    env_326_val.to_csv(os.path.join(DATA_DIR, '326_validation.csv'), index=False)

    print(f" -> Saved '326_train.csv' ({len(env_326_train):,} rows | {env_326_train['dt'].min()} to {env_326_train['dt'].max()})")
    print(f" -> Saved '326_validation.csv' ({len(env_326_val):,} rows | {env_326_val['dt'].min()} to {env_326_val['dt'].max()})")
    print(f" -> Total 326 Coverage: {len(df_326_env):,} rows ({df_326_env['dt'].min()} to {df_326_env['dt'].max()})")

    print("\nEnvironmental train/validation datasets successfully generated!")


if __name__ == '__main__':
    main()