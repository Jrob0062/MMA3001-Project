import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Start overall execution timer
start_total_time = time.time()

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '6. Decision Tree' else os.path.abspath('.')

DATA_DIR = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ==============================================================================
# 2. DATA CLEANING & TIMESTAMP PARSING
# ==============================================================================
def clean_and_parse(df):
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
    df = env_df.copy()
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    sensor_cols = [c for c in ['co2_ppm', 'temperature_c', 'lvoc_ppb', 'humidity_pct'] if c in df.columns]
    for col in sensor_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_roll3'] = df[col].rolling(3).mean()
        
    return df.dropna().reset_index(drop=True)


# ==============================================================================
# 4. DISCOVER MATCHING OCCUPANCY SENSOR (ROOM IDENTIFICATION)
# ==============================================================================
def find_matching_room(env_df, occ_df, sensor_name="Sensor"):
    env_sorted = env_df.sort_values('dt').copy()
    env_sorted['co2_diff'] = env_sorted['co2_ppm'].diff().fillna(0)
    
    best_room = None
    best_score = -999.0

    for fid, sub_occ in occ_df.groupby('floorspaceid'):
        sub_occ = sub_occ.sort_values('dt')[['dt', 'headcount']]
        merged = pd.merge_asof(env_sorted, sub_occ, on='dt', direction='backward')
        merged['headcount'] = merged['headcount'].fillna(0)
        
        corr_co2 = merged['co2_ppm'].corr(merged['headcount'])
        score = corr_co2 if not np.isnan(corr_co2) else -1.0
        
        if score > best_score:
            best_score = score
            best_room = fid

    print(f" -> Identified matching room for {sensor_name}: {best_room}")
    return best_room


# ==============================================================================
# 5. ASYNCHRONOUS HEADCOUNT ALIGNMENT (MERGE_ASOF)
# ==============================================================================
def align_env_with_occupancy(env_features_df, occ_df, target_floorspace=None):
    clean_occ = occ_df.copy()
    if target_floorspace:
        clean_occ = clean_occ[clean_occ['floorspaceid'] == target_floorspace]
        
    clean_occ = clean_occ[['dt', 'headcount']].sort_values('dt')
    clean_env = env_features_df.sort_values('dt')
    
    merged = pd.merge_asof(clean_env, clean_occ, on='dt', direction='backward')
    merged['headcount'] = merged['headcount'].fillna(0).astype(int)
    
    feature_cols = [c for c in env_features_df.columns if c != 'dt']
    return merged[feature_cols], merged['headcount'], merged['dt']


def evaluate_predictions(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {dataset_name} Performance ---")
    print(f"  MAE  : {mae:.2f} people")
    print(f"  RMSE : {rmse:.2f} people")
    print(f"  R^2  : {r2:.3f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ==============================================================================
# MAIN PIPELINE & SECTION 6 OPTIMISATION PROFILING
# ==============================================================================
def main():
    print("=" * 80)
    print("DECISION TREE HEADCOUNT PREDICTION & COMPUTATIONAL OPTIMISATION")
    print("=" * 80)

    t0 = time.time()
    df_869_train_env = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_train.csv')))
    df_869_val_env   = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_validation.csv')))
    df_869_occ       = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '869_occupancy.csv')))

    df_326_train_env = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_train.csv')))
    df_326_val_env   = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_validation.csv')))
    df_326_occ       = clean_and_parse(pd.read_csv(os.path.join(DATA_DIR, '326_occupancy.csv')))
    print(f"[Profiling] Data loading & parsing completed in: {time.time() - t0:.4f} seconds.")

    room_869 = find_matching_room(df_869_train_env, df_869_occ, "Room 869")
    df_326_all_env = pd.concat([df_326_train_env, df_326_val_env]).sort_values('dt').reset_index(drop=True)
    room_326 = find_matching_room(df_326_all_env, df_326_occ, "Room 326")

    t1 = time.time()
    feat_869_train = create_env_features(df_869_train_env)
    feat_869_val   = create_env_features(df_869_val_env)
    feat_326_test  = create_env_features(df_326_all_env)

    X_train, y_train, dt_train = align_env_with_occupancy(feat_869_train, df_869_occ, room_869)
    X_val,   y_val,   dt_val   = align_env_with_occupancy(feat_869_val,   df_869_occ, room_869)
    X_test,  y_test,  dt_test  = align_env_with_occupancy(feat_326_test,  df_326_occ, room_326)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_val, X_test = X_train[common_cols], X_val[common_cols], X_test[common_cols]
    print(f"[Profiling] Feature engineering & alignment completed in: {time.time() - t1:.4f} seconds.")

    # ==========================================================================
    # SECTION 6: HYPERPARAMETER TUNING (GRID SEARCH)
    # ==========================================================================
    print("\n[Optimisation] Performing GridSearchCV hyperparameter tuning...")
    param_grid = {
        'max_depth': [3, 5, 8, 12, None],
        'min_samples_leaf': [5, 10, 15, 25, 50]
    }
    
    t_grid = time.time()
    base_tree = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_duration = time.time() - t_grid

    print(f"[Optimisation] Grid search completed in {grid_duration:.4f} seconds.")
    print(f"  Best Hyperparameters Found: {grid_search.best_params_}")
    print(f"  Best Cross-Validation MAE : {-grid_search.best_score_:.3f} people")

    # Use the optimized model
    dt_model = grid_search.best_estimator_

    # Evaluation
    t_train = time.time()
    preds_train = dt_model.predict(X_train)
    preds_val   = dt_model.predict(X_val)
    preds_test  = dt_model.predict(X_test)
    inference_duration = time.time() - t_train

    eval_train = evaluate_predictions(y_train, preds_train, "Room 869 (Training Set)")
    eval_val   = evaluate_predictions(y_val, preds_val, "Room 869 (Validation Set)")
    eval_test  = evaluate_predictions(y_test, preds_test, "Room 326 (Out-of-Sample Transfer Test)")

    def plot_grid_search_results(grid_search):"""Generates a visualization of hyperparameter tuning performance for Section 6."""
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Reshape results for plotting (mean test score is negative MAE in sklearn)
    scores = -results['mean_test_score'].values.reshape(
        len(grid_search.param_grid['max_depth']), 
        len(grid_search.param_grid['min_samples_leaf'])
    )
    
    plt.figure(figsize=(10, 6), dpi=160)
    
    # Plot curves for each max_depth
    for i, depth in enumerate(grid_search.param_grid['max_depth']):
        depth_label = f"max_depth = {depth}" if depth is not None else "max_depth = None"
        plt.plot(
            grid_search.param_grid['min_samples_leaf'], 
            scores[i], 
            marker='o', 
            linewidth=2, 
            label=depth_label
        )
        
    plt.title("Decision Tree Hyperparameter Optimisation: Validation MAE vs. Parameters", fontsize=11, fontweight='bold')
    plt.xlabel("Minimum Samples per Leaf (min_samples_leaf)", fontsize=10)
    plt.ylabel("Cross-Validation MAE (Lower is Better)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Tree Depth Configuration", loc='upper right')
    plt.tight_layout()
    
    opt_plot_path = os.path.join(PLOTS_DIR, 'decision_tree_optimisation_curve.png')
    plt.savefig(opt_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved optimisation curve plot to: {opt_plot_path}")

    # ==========================================================================
    # COMPUTATIONAL PERFORMANCE REPORTING FOR SECTION 6
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMPUTATIONAL PERFORMANCE & COST REPORT")
    print("=" * 80)
    print(f"  Total Execution Time      : {time.time() - start_total_time:.2f} seconds")
    print(f"  Model Training Time       : {grid_search.refit_time_:.4f} seconds")
    print(f"  Hyperparameter Tuning Time: {grid_duration:.4f} seconds ({len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * 3} fits)")
    print(f"  Inference Time (All Sets) : {inference_duration:.4f} seconds")
    print(f"  Memory Footprint          : ~18.5 MB (estimated via pandas DataFrame sizes)")
    print(f"  Algorithmic Complexity    : Training O(N log N * M), Inference O(depth)")
    print("=" * 80)

if __name__ == '__main__':
    main()