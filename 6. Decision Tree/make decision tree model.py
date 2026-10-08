import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = os.path.join('.', '5. Reduced Data')

def parse_dt(df):
    """Robust timestamp parser."""
    possible_names = ['timestamp', 'dt', 'collecteddate', 'time', 'date']
    target = next((c for c in df.columns if c.strip().lower() in possible_names), None)
    if not target:
        raise KeyError(f"Timestamp column missing from: {list(df.columns)}")
    df['dt'] = pd.to_datetime(df[target].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').reset_index(drop=True)

# ==============================================================================
# 1. FEATURE ENGINEERING (ENVIRONMENTAL DATA ONLY)
# ==============================================================================
def create_env_features(env_df):
    """Engineers time, lag, and derivative features strictly on environmental data."""
    df = env_df.copy()
    
    # Time Features
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Environmental Lags & Rates of Change
    for col in ['co2_ppm', 'temperature_c', 'lvoc_ppb']:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_diff1'] = df[col].diff(1)
            df[f'{col}_roll3'] = df[col].rolling(window=3).mean()
            
    return df.dropna().reset_index(drop=True)


# ==============================================================================
# 2. LOAD SEPARATE DATASETS (ROOM 869)
# ==============================================================================
print("Loading Room 869 datasets separately...")
raw_869 = pd.read_csv(os.path.join(DATA_DIR, 'sensor_869_occupancy_overlap.csv'))
raw_869.columns = raw_869.columns.str.strip().str.replace('\ufeff', '')
raw_869 = parse_dt(raw_869)

# Strictly Separate Environmental Data and Ground Truth Occupancy
df_env_869 = raw_869[['dt', 'co2_ppm', 'humidity_pct']].copy() # Include available env columns
df_occ_869 = raw_869[['dt', 'headcount']].copy()

# Filter/Add temperature_c or lvoc_ppb if present in raw_869
for extra_col in ['temperature_c', 'lvoc_ppb']:
    if extra_col in raw_869.columns:
        df_env_869[extra_col] = raw_869[extra_col]

# Engineer Features on Environmental Data Alone
df_env_features_869 = create_env_features(df_env_869)

# Chronological 2/3 Train, 1/3 Validation Split on Environmental Features
split_869 = int(len(df_env_features_869) * (2 / 3))
env_train_869 = df_env_features_869.iloc[:split_869].copy()
env_val_869 = df_env_features_869.iloc[split_869:].copy()


# ==============================================================================
# 3. LOAD SEPARATE DATASETS (ROOM 326)
# ==============================================================================
print("Loading Room 326 datasets separately...")
env_326_raw = pd.read_csv(os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv'))
env_326_raw.columns = env_326_raw.columns.str.strip().str.replace('\ufeff', '')
env_326_raw = parse_dt(env_326_raw)

occ_326_raw = pd.read_csv(os.path.join(DATA_DIR, 'occupancy_data_2026_only.csv'))
occ_326_raw.columns = occ_326_raw.columns.str.strip().str.replace('\ufeff', '')
occ_326_raw = parse_dt(occ_326_raw)

# Environmental Features strictly from 326 Environmental CSV
df_env_features_326 = create_env_features(env_326_raw)


# ==============================================================================
# 4. ALIGN FEATURES & TARGETS FOR MODEL TRAINING / EVALUATION
# ==============================================================================
def get_X_y(env_df, occ_df):
    """Aligns separate environmental and occupancy dataframes by timestamp."""
    merged = pd.merge(env_df, occ_df[['dt', 'headcount']], on='dt', how='inner')
    X_cols = [c for c in env_df.columns if c != 'dt']
    return merged[X_cols], merged['headcount']

X_train, y_train = get_X_y(env_train_869, df_occ_869)
X_val_869, y_val_869 = get_X_y(env_val_869, df_occ_869)
X_val_326, y_val_326 = get_X_y(df_env_features_326, occ_326_raw)


# ==============================================================================
# 5. TRAIN & EVALUATE DECISION TREE REGRESSOR
# ==============================================================================
dt_regressor = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
dt_regressor.fit(X_train, y_train)

def print_metrics(model, X, y, name):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"\n--- {name} ---")
    print(f"MAE  : {mae:.2f} headcount")
    print(f"RMSE : {rmse:.2f} headcount")
    print(f"R²   : {r2:.3f}")

print_metrics(dt_regressor, X_train, y_train, "Room 869 (Train)")
print_metrics(dt_regressor, X_val_869, y_val_869, "Room 869 (Validation)")
print_metrics(dt_regressor, X_val_326, y_val_326, "Room 326 (Out-of-Sample Test)")