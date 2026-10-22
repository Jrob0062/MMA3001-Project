"""Decision Tree Headcount Prediction and Cross-Room Transfer Pipeline.

This module implements the core Decision Tree regression model trained on Room 869
environmental telemetry and evaluated on both in-room temporal validation data and
out-of-sample Room 326 deployment records.

It performs data sanitization, physical respiration feature engineering, co-located
sensor discovery via CO2 correlation, asynchronous backward-fill alignment, and
diagnostic time series visualization.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "6. Decision Tree/make decision tree model.py"
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# ==============================================================================
# 1. PATH RESOLUTION & REPOSITORY STRUCTURE
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
    """Strips UTF-8 BOM characters, cleans headers, and standardises datetime.

    Args:
        df (pd.DataFrame): Raw DataFrame with sensor or occupancy records.

    Returns:
        pd.DataFrame: Sanitized DataFrame sorted chronologically with a valid 'dt' column.

    Raises:
        KeyError: If no recognisable timestamp or datetime column is found in headers.

    Examples:
        >>> df_clean = clean_and_parse(df_raw)
    """
    df = df.copy()
    # Strip leading/trailing whitespace and Byte Order Marks from columns
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    candidates: List[str] = ['occupancystatuschangedate', 'dt', 'timestamp', 'collecteddate', 'time', 'date']
    dt_col: Optional[str] = next((c for c in df.columns if c.lower() in candidates), None)
    
    if not dt_col:
        raise KeyError(f"Could not identify datetime column in: {list(df.columns)}")
        
    # Standardize string timestamp to 19 characters before conversion
    df['dt'] = pd.to_datetime(df[dt_col].astype(str).str.slice(0, 19), errors='coerce')
    return df.sort_values('dt').dropna(subset=['dt']).reset_index(drop=True)


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def create_env_features(env_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers temporal schedule indicators and rate-of-change environmental features.

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
    
    # Environmental sensor metrics: Lags, First Differences (d[metric]/dt), and Rolling Means
    sensor_cols: List[str] = [c for c in ['co2_ppm', 'temperature_c', 'lvoc_ppb', 'humidity_pct'] if c in df.columns]
    for col in sensor_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        # First difference: rate of change is positive when humans exhale CO2
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_roll3'] = df[col].rolling(3).mean()
        
    return df.dropna().reset_index(drop=True)


# ==============================================================================
# 4. DISCOVER MATCHING OCCUPANCY SENSOR (ROOM IDENTIFICATION)
# ==============================================================================
def find_matching_room(
    env_df: pd.DataFrame, 
    occ_df: pd.DataFrame, 
    sensor_name: str = "Sensor"
) -> Optional[str]:
    """Identifies which occupancy camera shares a physical room with an environmental sensor.

    Computes Pearson correlation between continuous CO2 levels and discrete camera headcounts.

    Args:
        env_df (pd.DataFrame): Environmental time series DataFrame.
        occ_df (pd.DataFrame): Occupancy event DataFrame with candidate 'floorspaceid's.
        sensor_name (str, optional): Sensor identifier for logging. Defaults to "Sensor".

    Returns:
        Optional[str]: Matching floorspace UUID string if multiple exist, else None.

    Examples:
        >>> room_id = find_matching_room(df_env, df_occ, "Room 869")
    """
    print(f"\n--- Room Matching Analysis for {sensor_name} ---")
    if 'floorspaceid' not in occ_df.columns:
        print(f" -> '{sensor_name}' occupancy dataset has single-room format without 'floorspaceid'. Using all records.")
        return None

    env_sorted = env_df.sort_values('dt').copy()
    env_sorted['co2_diff'] = env_sorted['co2_ppm'].diff().fillna(0)
    
    best_room: Optional[str] = None
    best_score: float = -999.0
    stats: List[Dict[str, Union[str, float]]] = []

    for fid, sub_occ in occ_df.groupby('floorspaceid'):
        sub_occ = sub_occ.sort_values('dt')[['dt', 'headcount']]
        # Merge backward to carry headcount state forward across asynchronous events
        merged = pd.merge_asof(env_sorted, sub_occ, on='dt', direction='backward')
        merged['headcount'] = merged['headcount'].fillna(0)
        
        corr_co2 = merged['co2_ppm'].corr(merged['headcount'])
        corr_diff = merged['co2_diff'].corr(merged['headcount'])
        
        score = corr_co2 if not np.isnan(corr_co2) else -1.0
        stats.append({
            'floorspaceid': str(fid),
            'corr_co2': corr_co2,
            'corr_diff': corr_diff,
            'mean_hc': float(merged['headcount'].mean()),
            'max_hc': float(merged['headcount'].max())
        })
        
        if score > best_score:
            best_score = score
            best_room = str(fid)

    stats_df = pd.DataFrame(stats).sort_values(by='corr_co2', ascending=False)
    for _, r in stats_df.iterrows():
        is_best = " [BEST MATCH]" if r['floorspaceid'] == best_room else ""
        print(f"  Floorspace {r['floorspaceid'][:8]}... : r(CO2, Headcount) = {r['corr_co2']:+.3f} | "
              f"r(dCO2/dt) = {r['corr_diff']:+.3f} | Max HC: {r['max_hc']:.0f}{is_best}")
              
    print(f" -> Identified matching room for {sensor_name}: {best_room}")
    return best_room


# ==============================================================================
# 5. ASYNCHRONOUS HEADCOUNT ALIGNMENT (MERGE_ASOF)
# ==============================================================================
def align_env_with_occupancy(
    env_features_df: pd.DataFrame, 
    occ_df: pd.DataFrame, 
    target_floorspace: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Pairs periodic environmental sensor readings with asynchronous event-driven headcount logs.

    Applies backward merge_asof to hold the most recent headcount constant until
    the next camera transition occurs.

    Args:
        env_features_df (pd.DataFrame): Environmental feature DataFrame containing 'dt'.
        occ_df (pd.DataFrame): Occupancy event records.
        target_floorspace (Optional[str], optional): Room UUID filter. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: Feature matrix X, target vector y, and timestamps dt.

    Examples:
        >>> X, y, dt = align_env_with_occupancy(feat_df, occ_df, "uuid")
    """
    clean_occ = occ_df.copy()
    if target_floorspace and 'floorspaceid' in clean_occ.columns:
        clean_occ = clean_occ[clean_occ['floorspaceid'] == target_floorspace]
        
    clean_occ = clean_occ[['dt', 'headcount']].sort_values('dt')
    clean_env = env_features_df.sort_values('dt')
    
    # Backward merge holds the previous headcount constant until a new reading occurs
    merged = pd.merge_asof(clean_env, clean_occ, on='dt', direction='backward')
    
    # Fill pre-deployment timestamps with zero occupants
    merged['headcount'] = merged['headcount'].fillna(0).astype(int)
    
    feature_cols: List[str] = [c for c in env_features_df.columns if c != 'dt']
    return merged[feature_cols], merged['headcount'], merged['dt']


# ==============================================================================
# 6. MODEL EVALUATION HELPER
# ==============================================================================
def evaluate_predictions(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    dataset_name: str
) -> Dict[str, Union[str, float]]:
    """Calculates quantitative performance metrics for a model evaluation stage.

    Args:
        y_true (pd.Series): Ground-truth target headcount.
        y_pred (np.ndarray): Model-predicted continuous or discrete headcount.
        dataset_name (str): Label describing the evaluation split.

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing MAE, RMSE, R2, and target means.

    Examples:
        >>> eval_dict = evaluate_predictions(y_val, preds_val, "Validation Set")
    """
    mae: float = float(mean_absolute_error(y_true, y_pred))
    rmse: float = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2: float = float(r2_score(y_true, y_pred))
    
    print(f"\n--- {dataset_name} Performance ---")
    print(f"  MAE  : {mae:.2f} people")
    print(f"  RMSE : {rmse:.2f} people")
    print(f"  R^2  : {r2:.3f}")
    print(f"  Actual Mean: {y_true.mean():.2f} | Predicted Mean: {y_pred.mean():.2f}")
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'actual_mean': round(float(y_true.mean()), 4),
        'pred_mean': round(float(y_pred.mean()), 4)
    }


# ==============================================================================
# 7. MAIN PIPELINE
# ==============================================================================
def main() -> None:
    """Executes the Decision Tree training, evaluation, and visualization pipeline.

    Loads training and validation sets, aligns data using backward merge_asof, trains
    a constrained Decision Tree Regressor, tests transferability across rooms, and
    exports summary metrics to CSV and PNG visual plots.

    Returns:
        None
    """
    print("=" * 80)
    print("DECISION TREE HEADCOUNT PREDICTION & GENERALISATION PIPELINE")
    print("=" * 80)

    # 1. Load Datasets
    print("\nLoading datasets from '5. Reduced Data'...")
    df_869_train_env = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_train.csv')))
    df_869_val_env   = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_validation.csv')))
    df_869_occ       = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_occupancy.csv')))

    df_326_train_env = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_train.csv')))
    df_326_val_env   = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_validation.csv')))
    df_326_occ       = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_occupancy.csv')))

    # 2. Discover Which Occupancy Sensor Corresponds to Room 869
    room_869 = find_matching_room(df_869_train_env, df_869_occ, "Room 869")

    # 3. Discover Which Occupancy Sensor Corresponds to Room 326
    df_326_all_env = pd.concat([df_326_train_env, df_326_val_env]).sort_values('dt').reset_index(drop=True)
    room_326 = find_matching_room(df_326_all_env, df_326_occ, "Room 326")

    # 4. Feature Engineering
    print("\nEngineering environmental features (lags, differences, rolling means, time)...")
    feat_869_train = create_env_features(df_869_train_env)
    feat_869_val   = create_env_features(df_869_val_env)
    feat_326_test  = create_env_features(df_326_all_env)

    # 5. Backward Alignment with Ground-Truth Headcount
    print("Aligning environmental records with asynchronous headcount changes...")
    X_train, y_train, dt_train = align_env_with_occupancy(feat_869_train, df_869_occ, room_869)
    X_val,   y_val,   dt_val   = align_env_with_occupancy(feat_869_val,   df_869_occ, room_869)
    X_test,  y_test,  dt_test  = align_env_with_occupancy(feat_326_test,  df_326_occ, room_326)

    common_cols: List[str] = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_val   = X_val[common_cols]
    X_test  = X_test[common_cols]

    print(f"\nAligned Datasets:")
    print(f"  Training Set (Room 869 Train) : {len(X_train):,} samples | Features: {list(common_cols)}")
    print(f"  Validation Set (Room 869 Val) : {len(X_val):,} samples")
    print(f"  Out-of-Sample Test (Room 326) : {len(X_test):,} samples")

    # 6. Train Decision Tree Regressor
    print("\nTraining DecisionTreeRegressor on Room 869 Training Data...")
    dt_model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=15,
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    # 7. Evaluate on Room 869 Training & Validation, and Room 326 Test
    preds_train = dt_model.predict(X_train)
    preds_val   = dt_model.predict(X_val)
    preds_test  = dt_model.predict(X_test)

    eval_train = evaluate_predictions(y_train, preds_train, "Room 869 (Training Set)")
    eval_val   = evaluate_predictions(y_val, preds_val, "Room 869 (Validation Set)")
    eval_test  = evaluate_predictions(y_test, preds_test, "Room 326 (Out-of-Sample Transfer Test)")

    # 8. Multi-Room Cross Generalisation Evaluation
    print("\n" + "=" * 80)
    print("CROSS-ROOM GENERALISATION TEST ACROSS ALL CANDIDATE ROOMS IN 326 DEPLOYMENT")
    print("=" * 80)
    if 'floorspaceid' in df_326_occ.columns:
        for fid in df_326_occ['floorspaceid'].unique():
            X_sub, y_sub, _ = align_env_with_occupancy(feat_326_test, df_326_occ, str(fid))
            p_sub = dt_model.predict(X_sub[common_cols])
            mae = mean_absolute_error(y_sub, p_sub)
            rmse = np.sqrt(mean_squared_error(y_sub, p_sub))
            r2 = r2_score(y_sub, p_sub)
            print(f"  Room {str(fid)[:8]}... : MAE = {mae:.2f} people | RMSE = {rmse:.2f} | R^2 = {r2:+.3f} | Actual Mean = {y_sub.mean():.2f}")
    else:
        print(f"  Room 326 Deployment (Full Dataset): MAE = {eval_test['mae']:.2f} people | RMSE = {eval_test['rmse']:.2f} | R^2 = {eval_test['r2']:+.3f} | Actual Mean = {eval_test['actual_mean']:.2f}")

    # 9. Generate Diagnostic Visualizations
    print("\nGenerating model evaluation plots...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), dpi=160)

    # Subplot 1: Feature Importances
    ax_feat = axes[0]
    importances = pd.Series(dt_model.feature_importances_, index=common_cols).sort_values(ascending=True)
    importances.plot(kind='barh', ax=ax_feat, color='#2563EB', edgecolor='black', alpha=0.85)
    ax_feat.set_title("Decision Tree Feature Importances (Trained on Room 869)", fontsize=11, fontweight='bold')
    ax_feat.set_xlabel("Relative Importance")
    ax_feat.grid(True, linestyle=':', alpha=0.5)

    # Subplot 2: Room 869 Validation Actual vs Predicted
    ax_val = axes[1]
    zoom_val = slice(0, min(1500, len(dt_val)))
    ax_val.plot(dt_val.iloc[zoom_val], y_val.iloc[zoom_val], color='black', linewidth=1.5, label='Actual Headcount')
    ax_val.plot(dt_val.iloc[zoom_val], preds_val[zoom_val], color='#DC2626', linewidth=1.5, alpha=0.85, label='Predicted Headcount (DT)')
    ax_val.set_title(f"Room 869 Validation Set: Actual vs. Predicted Headcount (MAE: {eval_val['mae']:.2f} people)", fontsize=11, fontweight='bold')
    ax_val.set_ylabel("Headcount")
    ax_val.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_val.grid(True, linestyle=':', alpha=0.5)
    ax_val.legend(loc='upper right')

    # Subplot 3: Room 326 Out-of-Sample Transfer Test
    ax_test = axes[2]
    zoom_test = slice(0, min(1500, len(dt_test)))
    ax_test.plot(dt_test.iloc[zoom_test], y_test.iloc[zoom_test], color='black', linewidth=1.5, label='Actual Headcount')
    ax_test.plot(dt_test.iloc[zoom_test], preds_test[zoom_test], color='#16A34A', linewidth=1.5, alpha=0.85, label='Predicted Headcount (Transfer DT)')
    ax_test.set_title(f"Room 326 Generalisation Test: Actual vs. Predicted Headcount (MAE: {eval_test['mae']:.2f} people)", fontsize=11, fontweight='bold')
    ax_test.set_ylabel("Headcount")
    ax_test.set_xlabel("Date")
    ax_test.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_test.grid(True, linestyle=':', alpha=0.5)
    ax_test.legend(loc='upper right')

    plt.tight_layout()
    out_plot = os.path.join(PLOTS_DIR, 'decision_tree_headcount_model.png')
    plt.savefig(out_plot, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plot to: {out_plot}")

    # Export Baseline Summary to CSV
    dt_eval_df = pd.DataFrame([
        {'model': 'Decision Tree Regressor (depth=5, leaf=15)', 'dataset': '869 Training', 'samples': len(X_train), 'mae_people': eval_train['mae'], 'rmse_people': eval_train['rmse'], 'r2_score': eval_train['r2'], 'actual_mean': eval_train['actual_mean'], 'pred_mean': eval_train['pred_mean']},
        {'model': 'Decision Tree Regressor (depth=5, leaf=15)', 'dataset': '869 Validation', 'samples': len(X_val), 'mae_people': eval_val['mae'], 'rmse_people': eval_val['rmse'], 'r2_score': eval_val['r2'], 'actual_mean': eval_val['actual_mean'], 'pred_mean': eval_val['pred_mean']},
        {'model': 'Decision Tree Regressor (depth=5, leaf=15)', 'dataset': '326 Generalisation Test', 'samples': len(X_test), 'mae_people': eval_test['mae'], 'rmse_people': eval_test['rmse'], 'r2_score': eval_test['r2'], 'actual_mean': eval_test['actual_mean'], 'pred_mean': eval_test['pred_mean']},
    ])
    dt_csv_path = os.path.join(RESULTS_DIR, 'decision_tree_baseline_summary.csv')
    dt_eval_df.to_csv(dt_csv_path, index=False)
    print(f"Saved Decision Tree baseline summary to: {dt_csv_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
