"""Comparative Evaluation Pipeline: Linear Regression Baseline vs. Decision Tree Regressor.

This module evaluates and benchmarks an ordinary least squares (OLS) Multiple Linear
Regression baseline (Topic 5.1) against a non-linear Decision Tree Regressor for
indoor room headcount prediction.

It demonstrates why linear models suffer when predicting non-negative, count-based
occupancy metrics from environmental telemetry, and proves the generalization capacity
of non-linear decision trees across independent physical room deployments.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "6. Decision Tree/Linear Regressino Model baseline.py"
    ```
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# ==============================================================================
# 1. PATH RESOLUTION & REPOSITORY CONFIGURATION
# ==============================================================================
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '6. Decision Tree' else os.path.abspath('.')

DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, 'plots')
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, '7. model results')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 2. DATA CLEANING & TIMESTAMP PARSING
# ==============================================================================
def clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """Strips UTF-8 BOM characters, cleans column headers, and parses datetime timestamps.

    Args:
        df (pd.DataFrame): Raw DataFrame containing telemetry or camera events.

    Returns:
        pd.DataFrame: Sanitized DataFrame sorted chronologically with a valid 'dt' column.

    Raises:
        KeyError: If no recognized datetime column exists in the DataFrame headers.

    Examples:
        >>> df_clean = clean_and_parse(raw_df)
    """
    df = df.copy()
    # Strip potential UTF-8 byte order mark from header strings
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    candidates: List[str] = ['occupancystatuschangedate', 'dt', 'timestamp', 'collecteddate', 'time', 'date']
    dt_col: Optional[str] = next((c for c in df.columns if c.lower() in candidates), None)
    
    if not dt_col:
        raise KeyError(f"Could not identify datetime column in: {list(df.columns)}")
        
    # Standardize to 19-character ISO datetime string prior to conversion
    df['dt'] = pd.to_datetime(df[dt_col].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').dropna(subset=['dt']).reset_index(drop=True)


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def create_env_features(env_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers temporal indicators and dynamic physical environmental features.

    Derives lags, first differences, and rolling moving averages strictly from ambient
    environmental sensor channels (CO2, temperature, volatile organics).

    Args:
        env_df (pd.DataFrame): Environmental DataFrame with 'dt' and telemetry columns.

    Returns:
        pd.DataFrame: Enriched DataFrame with dynamic features, initial rolling NaNs dropped.

    Examples:
        >>> feat_df = create_env_features(env_df)
    """
    df = env_df.copy()
    
    # Temporal schedule features capture typical occupancy schedules
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Numerical rate of change and rolling trends
    sensor_cols: List[str] = [c for c in ['co2_ppm', 'temperature_c', 'lvoc_ppb', 'humidity_pct'] if c in df.columns]
    for col in sensor_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        # First difference: rate of change is positive when occupants exhale CO2
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_roll3'] = df[col].rolling(3).mean()
        
    return df.dropna().reset_index(drop=True)


# ==============================================================================
# 4. ASYNCHRONOUS HEADCOUNT ALIGNMENT
# ==============================================================================
def align_env_with_occupancy(
    env_features_df: pd.DataFrame, 
    occ_df: pd.DataFrame, 
    target_floorspace: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Pairs periodic environmental sensor readings with asynchronous event-driven headcount logs.

    Applies backward merge_asof to carry forward the previous occupancy state until
    a new headcount change occurs.

    Args:
        env_features_df (pd.DataFrame): Feature matrix containing 'dt'.
        occ_df (pd.DataFrame): Ground-truth occupancy records.
        target_floorspace (Optional[str], optional): Room floorspace UUID filter. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: Feature matrix X, target vector y, and timestamps dt.

    Examples:
        >>> X, y, dt = align_env_with_occupancy(feat_df, occ_df, "target_uuid")
    """
    clean_occ = occ_df.copy()
    if target_floorspace and 'floorspaceid' in clean_occ.columns:
        clean_occ = clean_occ[clean_occ['floorspaceid'] == target_floorspace]
        
    clean_occ = clean_occ[['dt', 'headcount']].sort_values('dt')
    clean_env = env_features_df.sort_values('dt')
    
    # Carry forward last known occupant count until next event occurs
    merged = pd.merge_asof(clean_env, clean_occ, on='dt', direction='backward')
    merged['headcount'] = merged['headcount'].fillna(0).astype(int)
    
    feature_cols: List[str] = [c for c in env_features_df.columns if c != 'dt']
    return merged[feature_cols], merged['headcount'], merged['dt']


# ==============================================================================
# 5. METRIC COMPUTATION
# ==============================================================================
def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Mean Absolute Error, Root Mean Squared Error, and R2 score.

    Args:
        y_true (pd.Series): Observed true headcount values.
        y_pred (np.ndarray): Predicted continuous headcount values.

    Returns:
        Dict[str, float]: Dictionary with 'mae', 'rmse', and 'r2' scores.

    Examples:
        >>> metrics = compute_metrics(pd.Series([0, 1]), np.array([0.2, 0.9]))
    """
    mae: float = float(mean_absolute_error(y_true, y_pred))
    rmse: float = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2: float = float(r2_score(y_true, y_pred))
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ==============================================================================
# 6. MAIN PIPELINE
# ==============================================================================
def main() -> None:
    """Executes the comparative training, evaluation, and visualization pipeline.

    Fits Linear Regression baseline and Decision Tree Regressor on Room 869, evaluates
    both models across in-sample train, in-room validation, and cross-room transfer sets,
    and exports summary results to CSV and diagnostic visual plots.

    Returns:
        None
    """
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

    room_869: str = '941f0352-bb69-47d0-a9c5-ac29eb4f7319'
    room_326: str = '227f1068-9e41-469f-bef0-275f27a0bdba'

    df_326_all_env = pd.concat([df_326_train_env, df_326_val_env]).sort_values('dt').reset_index(drop=True)

    # 2. Feature Engineering
    print("Engineering environmental dynamic features...")
    feat_869_train = create_env_features(df_869_train_env)
    feat_869_val   = create_env_features(df_869_val_env)
    feat_326_test  = create_env_features(df_326_all_env)

    # 3. Asynchronous Alignment
    print("Aligning environmental records with asynchronous headcount changes...")
    X_train, y_train, dt_train = align_env_with_occupancy(feat_869_train, df_869_occ, room_869)
    X_val,   y_val,   dt_val   = align_env_with_occupancy(feat_869_val,   df_869_occ, room_869)
    X_test,  y_test,  dt_test  = align_env_with_occupancy(feat_326_test,  df_326_occ, room_326)

    common_cols: List[str] = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_val   = X_val[common_cols]
    X_test  = X_test[common_cols]

    print(f"Features: {list(common_cols)}")
    print(f"Samples: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

    # 4. Model 1: Linear Regression Baseline
    print("\nTraining Linear Regression Baseline (Topic 5.1)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_preds_train = lr_model.predict(X_train)
    lr_preds_val   = lr_model.predict(X_val)
    lr_preds_test  = lr_model.predict(X_test)

    metrics_train_lr = compute_metrics(y_train, lr_preds_train)
    metrics_val_lr   = compute_metrics(y_val, lr_preds_val)
    metrics_test_lr  = compute_metrics(y_test, lr_preds_test)

    # 5. Model 2: Decision Tree Regressor
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

    metrics_train_dt = compute_metrics(y_train, dt_preds_train)
    metrics_val_dt   = compute_metrics(y_val, dt_preds_val)
    metrics_test_dt  = compute_metrics(y_test, dt_preds_test)

    # 6. Tabular Performance Output
    print("\n" + "=" * 85)
    print(f"{'EVALUATION STAGE':<26} | {'LINEAR REGRESSION (BASELINE)':<28} | {'DECISION TREE REGRESSOR':<26}")
    print(f"{'':<26} | {'MAE':<8} {'RMSE':<8} {'R^2':<8} | {'MAE':<8} {'RMSE':<8} {'R^2':<8}")
    print("-" * 85)
    print(f"{'Room 869 (Training Fit)':<26} | {metrics_train_lr['mae']:<8.2f} {metrics_train_lr['rmse']:<8.2f} {metrics_train_lr['r2']:<+8.3f} | {metrics_train_dt['mae']:<8.2f} {metrics_train_dt['rmse']:<8.2f} {metrics_train_dt['r2']:<+8.3f}")
    print(f"{'Room 869 (Validation)':<26} | {metrics_val_lr['mae']:<8.2f} {metrics_val_lr['rmse']:<8.2f} {metrics_val_lr['r2']:<+8.3f} | {metrics_val_dt['mae']:<8.2f} {metrics_val_dt['rmse']:<8.2f} {metrics_val_dt['r2']:<+8.3f}")
    print(f"{'Room 326 (Generalisation)':<26} | {metrics_test_lr['mae']:<8.2f} {metrics_test_lr['rmse']:<8.2f} {metrics_test_lr['r2']:<+8.3f} | {metrics_test_dt['mae']:<8.2f} {metrics_test_dt['rmse']:<8.2f} {metrics_test_dt['r2']:<+8.3f}")
    print("=" * 85)

    # 7. Print Linear Regression Coefficients
    print("\n--- Linear Regression Learned Model Equation ---")
    print(f"Intercept (beta_0): {lr_model.intercept_:+.4f}")
    coef_df = pd.DataFrame({
        'Feature': common_cols,
        'Coefficient (beta)': lr_model.coef_
    }).sort_values(by='Coefficient (beta)', key=abs, ascending=False)
    for _, r in coef_df.iterrows():
        print(f"  {r['Feature']:<24}: {r['Coefficient (beta)']:+.5f}")

    # 8. Transfer Generalisation Test
    print("\n" + "=" * 85)
    print("TRANSFER GENERALISATION TEST ON ROOM 326 ACROSS ALL CANDIDATE ROOMS")
    print("=" * 85)
    if 'floorspaceid' in df_326_occ.columns:
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
    else:
        m_lr = compute_metrics(y_test, lr_preds_test)
        m_dt = compute_metrics(y_test, dt_preds_test)
        print("Room 326 (Full Dataset) :")
        print(f"  Linear Regression Baseline : MAE = {m_lr['mae']:.2f} | RMSE = {m_lr['rmse']:.2f} | R^2 = {m_lr['r2']:+.3f}")
        print(f"  Decision Tree Regressor    : MAE = {m_dt['mae']:.2f} | RMSE = {m_dt['rmse']:.2f} | R^2 = {m_dt['r2']:+.3f}")
        print(f"  Improvement with Decision Tree: MAE reduced by {(m_lr['mae'] - m_dt['mae']):.2f} people ({(1 - m_dt['mae']/m_lr['mae'])*100:.1f}%)")

    # 9. Comparative Diagnostic Plots
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

    # Subplot 2: Room 869 Validation Time Series
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

    # Subplot 3: Room 326 Out-of-Sample Test Time Series
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

    # Export Comparative Baseline Summary to CSV
    baseline_summary_df = pd.DataFrame([
        {'model': 'Linear Regression Baseline', 'stage': '869 Train', 'samples': len(X_train), 'mae_people': round(metrics_train_lr['mae'], 4), 'rmse_people': round(metrics_train_lr['rmse'], 4), 'r2_score': round(metrics_train_lr['r2'], 4)},
        {'model': 'Decision Tree Regressor', 'stage': '869 Train', 'samples': len(X_train), 'mae_people': round(metrics_train_dt['mae'], 4), 'rmse_people': round(metrics_train_dt['rmse'], 4), 'r2_score': round(metrics_train_dt['r2'], 4)},
        {'model': 'Linear Regression Baseline', 'stage': '869 Validation', 'samples': len(X_val), 'mae_people': round(metrics_val_lr['mae'], 4), 'rmse_people': round(metrics_val_lr['rmse'], 4), 'r2_score': round(metrics_val_lr['r2'], 4)},
        {'model': 'Decision Tree Regressor', 'stage': '869 Validation', 'samples': len(X_val), 'mae_people': round(metrics_val_dt['mae'], 4), 'rmse_people': round(metrics_val_dt['rmse'], 4), 'r2_score': round(metrics_val_dt['r2'], 4)},
        {'model': 'Linear Regression Baseline', 'stage': '326 Generalisation Test', 'samples': len(X_test), 'mae_people': round(metrics_test_lr['mae'], 4), 'rmse_people': round(metrics_test_lr['rmse'], 4), 'r2_score': round(metrics_test_lr['r2'], 4)},
        {'model': 'Decision Tree Regressor', 'stage': '326 Generalisation Test', 'samples': len(X_test), 'mae_people': round(metrics_test_dt['mae'], 4), 'rmse_people': round(metrics_test_dt['rmse'], 4), 'r2_score': round(metrics_test_dt['r2'], 4)},
    ])
    baseline_csv_path = os.path.join(RESULTS_DIR, 'linear_regression_vs_decision_tree_summary.csv')
    baseline_summary_df.to_csv(baseline_csv_path, index=False)
    print(f"Saved comparative baseline summary to: {baseline_csv_path}")

    print("\n" + "=" * 85)
    print("COMPARATIVE EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == '__main__':
    main()
