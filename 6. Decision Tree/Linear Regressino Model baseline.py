import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '6. Decision Tree' else os.path.abspath('.')

DATA_DIR = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ==============================================================================
# 2. DATA CLEANING & TIMESTAMP PARSING
# ==============================================================================
def clean_and_parse(df):
    """
    Strips BOM characters, cleans headers, and standardises datetime.
    Supports 'occupancystatuschangedate', 'dt', 'timestamp', and 'collecteddate'.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    candidates = ['occupancystatuschangedate', 'dt', 'timestamp', 'collecteddate', 'time', 'date']
    dt_col = next((c for c in df.columns if c.lower() in candidates), None)
    if not dt_col:
        raise KeyError(f"Could not identify datetime column in: {list(df.columns)}")
        
    df['dt'] = pd.to_datetime(df[dt_col].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').dropna(subset=['dt']).reset_index(drop=True)


# ==============================================================================
# 3. FEATURE ENGINEERING (ENVIRONMENTAL DYNAMICS)
# ==============================================================================
def create_env_features(env_df):
    """
    Engineers temporal and rate-of-change dynamic features strictly from environmental sensors.
    Keeps feature engineering strictly decoupled from occupancy ground-truth.
    """
    df = env_df.copy()
    
    # Temporal indicators
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Environmental sensor metrics: Lags, First Differences (d[metric]/dt), and Rolling Means
    sensor_cols = [c for c in ['co2_ppm', 'temperature_c', 'lvoc_ppb', 'humidity_pct'] if c in df.columns]
    for col in sensor_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_diff1'] = df[col].diff(1)     # Rate of change: positive when humans exhale CO2
        df[f'{col}_roll3'] = df[col].rolling(3).mean()
        
    return df.dropna().reset_index(drop=True)


# ==============================================================================
# 4. ASYNCHRONOUS HEADCOUNT ALIGNMENT (MERGE_ASOF)
# ==============================================================================
def align_env_with_occupancy(env_features_df, occ_df, target_floorspace=None):
    """
    Pairs periodic environmental sensor readings with asynchronous occupancy changes.
    Because occupancy timestamps ONLY record when headcount changes, merge_asof with
    direction='backward' carries forward the previous state until a new change occurs.
    """
    clean_occ = occ_df.copy()
    if target_floorspace:
        clean_occ = clean_occ[clean_occ['floorspaceid'] == target_floorspace]
        
    clean_occ = clean_occ[['dt', 'headcount']].sort_values('dt')
    clean_env = env_features_df.sort_values('dt')
    
    # Backward merge holds the previous headcount constant until a new reading occurs
    merged = pd.merge_asof(clean_env, clean_occ, on='dt', direction='backward')
    
    # If environmental readings precede the very first camera event, fill with 0
    merged['headcount'] = merged['headcount'].fillna(0).astype(int)
    
    feature_cols = [c for c in env_features_df.columns if c != 'dt']
    return merged[feature_cols], merged['headcount'], merged['dt']


# ==============================================================================
# 5. MODEL EVALUATION HELPER
# ==============================================================================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ==============================================================================
# MAIN COMPARATIVE PIPELINE
# ==============================================================================
def main():
    print("=" * 85)
    print("COMPARATIVE EVALUATION: LINEAR REGRESSION BASELINE vs. DECISION TREE")
    print("=" * 85)

    # 1. Load Environmental and Occupancy Datasets
    print("\nLoading datasets from '5. Reduced Data'...")
    df_869_train_env = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_train.csv')))
    df_869_val_env   = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_validation.csv')))
    df_869_occ       = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_occupancy.csv')))

    df_326_train_env = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_train.csv')))
    df_326_val_env   = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_validation.csv')))
    df_326_occ       = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_occupancy.csv')))

    # Room 869 matching room (strongest physical CO2 correlation)
    room_869 = '941f0352-bb69-47d0-a9c5-ac29eb4f7319'
    
    # Room 326 candidate room for transfer test
    room_326 = '227f1068-9e41-469f-bef0-275f27a0bdba'

    # Combine 326 train and val for a full deployment evaluation
    df_326_all_env = pd.concat([df_326_train_env, df_326_val_env]).sort_values('dt').reset_index(drop=True)

    # 2. Feature Engineering on Environmental Datasets
    print("Engineering environmental dynamic features...")
    feat_869_train = create_env_features(df_869_train_env)
    feat_869_val   = create_env_features(df_869_val_env)
    feat_326_test  = create_env_features(df_326_all_env)

    # 3. Asynchronous Backward Alignment
    print("Aligning environmental records with asynchronous headcount changes...")
    X_train, y_train, dt_train = align_env_with_occupancy(feat_869_train, df_869_occ, room_869)
    X_val,   y_val,   dt_val   = align_env_with_occupancy(feat_869_val,   df_869_occ, room_869)
    X_test,  y_test,  dt_test  = align_env_with_occupancy(feat_326_test,  df_326_occ, room_326)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_val   = X_val[common_cols]
    X_test  = X_test[common_cols]

    print(f"Features: {list(common_cols)}")
    print(f"Samples: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

    # ==========================================================================
    # 4. MODEL 1: LINEAR REGRESSION BASELINE
    # ==========================================================================
    print("\nTraining Linear Regression Baseline (Topic 5.1)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_preds_train = lr_model.predict(X_train)
    lr_preds_val   = lr_model.predict(X_val)
    lr_preds_test  = lr_model.predict(X_test)

    # ==========================================================================
    # 5. MODEL 2: DECISION TREE REGRESSOR
    # ==========================================================================
    print("Training Decision Tree Regressor...")
    dt_model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=15,
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    dt_preds_train = dt_model.predict(X_train)
    dt_preds_val   = dt_model.predict(X_val)
    dt_preds_test  = dt_model.predict(X_test)

    # ==========================================================================
    # 6. COMPARATIVE PERFORMANCE METRICS
    # ==========================================================================
    metrics_train_lr = compute_metrics(y_train, lr_preds_train)
    metrics_val_lr   = compute_metrics(y_val, lr_preds_val)
    metrics_test_lr  = compute_metrics(y_test, lr_preds_test)

    metrics_train_dt = compute_metrics(y_train, dt_preds_train)
    metrics_val_dt   = compute_metrics(y_val, dt_preds_val)
    metrics_test_dt  = compute_metrics(y_test, dt_preds_test)

    print("\n" + "=" * 85)
    print(f"{'EVALUATION STAGE':<26} | {'LINEAR REGRESSION (BASELINE)':<26} | {'DECISION TREE REGRESSOR':<26}")
    print(f"{'':<26} | {'MAE':<7} {'RMSE':<7} {'R^2':<8} | {'MAE':<7} {'RMSE':<7} {'R^2':<8}")
    print("-" * 85)
    print(f"{'Room 869 (Training Fit)':<26} | "
          f"{metrics_train_lr['mae']:<7.2f} {metrics_train_lr['rmse']:<7.2f} {metrics_train_lr['r2']:<+8.3f} | "
          f"{metrics_train_dt['mae']:<7.2f} {metrics_train_dt['rmse']:<7.2f} {metrics_train_dt['r2']:<+8.3f}")
    print(f"{'Room 869 (Validation)':<26} | "
          f"{metrics_val_lr['mae']:<7.2f} {metrics_val_lr['rmse']:<7.2f} {metrics_val_lr['r2']:<+8.3f} | "
          f"{metrics_val_dt['mae']:<7.2f} {metrics_val_dt['rmse']:<7.2f} {metrics_val_dt['r2']:<+8.3f}")
    print(f"{'Room 326 (Generalisation)':<26} | "
          f"{metrics_test_lr['mae']:<7.2f} {metrics_test_lr['rmse']:<7.2f} {metrics_test_lr['r2']:<+8.3f} | "
          f"{metrics_test_dt['mae']:<7.2f} {metrics_test_dt['rmse']:<7.2f} {metrics_test_dt['r2']:<+8.3f}")
    print("=" * 85)

    # 7. Linear Regression Learned Coefficients
    print("\n--- Linear Regression Learned Model Equation ---")
    print(f"Intercept (beta_0): {lr_model.intercept_:+.4f}")
    coef_df = pd.DataFrame({
        'Feature': common_cols,
        'Coefficient (beta)': lr_model.coef_
    }).sort_values(by='Coefficient (beta)', key=abs, ascending=False)
    for _, r in coef_df.iterrows():
        print(f"  {r['Feature']:<24}: {r['Coefficient (beta)']:+.5f}")

    # 8. Cross-Room Generalisation across all candidate rooms in Room 326 deployment
    print("\n" + "=" * 85)
    print("TRANSFER GENERALISATION TEST ON ROOM 326 ACROSS ALL CANDIDATE ROOMS")
    print("=" * 85)
    for fid in ['0f3f37f8-0335-41e0-ae87-a1d677dfc7d7', '227f1068-9e41-469f-bef0-275f27a0bdba', '941f0352-bb69-47d0-a9c5-ac29eb4f7319']:
        X_sub, y_sub, _ = align_env_with_occupancy(feat_326_test, df_326_occ, fid)
        p_sub_lr = lr_model.predict(X_sub[common_cols])
        p_sub_dt = dt_model.predict(X_sub[common_cols])
        m_lr = compute_metrics(y_sub, p_sub_lr)
        m_dt = compute_metrics(y_sub, p_sub_dt)
        print(f"Room {fid[:8]}... :")
        print(f"  Linear Regression Baseline : MAE = {m_lr['mae']:.2f} | RMSE = {m_lr['rmse']:.2f} | R^2 = {m_lr['r2']:+.3f}")
        print(f"  Decision Tree Regressor    : MAE = {m_dt['mae']:.2f} | RMSE = {m_dt['rmse']:.2f} | R^2 = {m_dt['r2']:+.3f}")
        print(f"  Improvement with Decision Tree: MAE reduced by {(m_lr['mae'] - m_dt['mae']):.2f} people ({(1 - m_dt['mae']/m_lr['mae'])*100:.1f}%)")

    # ==========================================================================
    # 9. DIAGNOSTIC COMPARISON VISUALIZATIONS
    # ==========================================================================
    print("\nGenerating comparative diagnostic plots...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), dpi=160)

    # Subplot 1: Bar Chart comparing MAE across datasets
    ax_bar = axes[0]
    stages = ['869 Train', '869 Validation', '326 Generalisation']
    lr_maes = [metrics_train_lr['mae'], metrics_val_lr['mae'], metrics_test_lr['mae']]
    dt_maes = [metrics_train_dt['mae'], metrics_val_dt['mae'], metrics_test_dt['mae']]

    x = np.arange(len(stages))
    width = 0.35
    b1 = ax_bar.bar(x - width/2, lr_maes, width, label='Linear Regression Baseline', color='#64748B', edgecolor='black')
    b2 = ax_bar.bar(x + width/2, dt_maes, width, label='Decision Tree Regressor', color='#2563EB', edgecolor='black')
    ax_bar.set_ylabel("MAE [People]", fontsize=10, fontweight='bold')
    ax_bar.set_title("Mean Absolute Error (MAE) Comparison: Linear Regression Baseline vs. Decision Tree", fontsize=11, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(stages, fontsize=10, fontweight='bold')
    ax_bar.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax_bar.legend(loc='upper left', fontsize=9.5)

    for bar in b1 + b2:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 2: Room 869 Validation (Zoom in on 10-day window)
    ax_val = axes[1]
    zoom_val = slice(0, min(1200, len(dt_val)))
    ax_val.plot(dt_val.iloc[zoom_val], y_val.iloc[zoom_val], color='black', linewidth=1.5, label='Actual Headcount')
    ax_val.plot(dt_val.iloc[zoom_val], lr_preds_val[zoom_val], color='#64748B', linestyle='--', linewidth=1.4, alpha=0.9, label=f'Linear Regression (MAE: {metrics_val_lr["mae"]:.2f})')
    ax_val.plot(dt_val.iloc[zoom_val], dt_preds_val[zoom_val], color='#DC2626', linewidth=1.6, alpha=0.85, label=f'Decision Tree (MAE: {metrics_val_dt["mae"]:.2f})')
    ax_val.set_title("Room 869 Validation Set: Actual vs. Linear Regression vs. Decision Tree", fontsize=11, fontweight='bold')
    ax_val.set_ylabel("Headcount", fontsize=10, fontweight='bold')
    ax_val.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_val.grid(True, linestyle=':', alpha=0.5)
    ax_val.legend(loc='upper right', fontsize=9)

    # Subplot 3: Room 326 Out-of-Sample Test (Zoom in on 10-day window)
    ax_test = axes[2]
    zoom_test = slice(0, min(1200, len(dt_test)))
    ax_test.plot(dt_test.iloc[zoom_test], y_test.iloc[zoom_test], color='black', linewidth=1.5, label='Actual Headcount')
    ax_test.plot(dt_test.iloc[zoom_test], lr_preds_test[zoom_test], color='#64748B', linestyle='--', linewidth=1.4, alpha=0.9, label=f'Linear Regression (MAE: {metrics_test_lr["mae"]:.2f})')
    ax_test.plot(dt_test.iloc[zoom_test], dt_preds_test[zoom_test], color='#16A34A', linewidth=1.6, alpha=0.85, label=f'Decision Tree (MAE: {metrics_test_dt["mae"]:.2f})')
    ax_test.set_title("Room 326 Generalisation Test: Actual vs. Linear Regression vs. Decision Tree", fontsize=11, fontweight='bold')
    ax_test.set_ylabel("Headcount", fontsize=10, fontweight='bold')
    ax_test.set_xlabel("Date", fontsize=10, fontweight='bold')
    ax_test.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_test.grid(True, linestyle=':', alpha=0.5)
    ax_test.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out_plot = os.path.join(PLOTS_DIR, 'linear_regression_vs_decision_tree.png')
    plt.savefig(out_plot, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative plot to: {out_plot}")

    print("\n" + "=" * 85)
    print("COMPARATIVE EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == '__main__':
    main()