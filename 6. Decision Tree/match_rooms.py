"""Spatial Room Mapping and Sensor Co-location Discovery Utility.

This module determines physical room co-location between environmental sensors and
ground-truth camera occupancy sensors by evaluating respiration dynamics and CO2
concentration changes during occupied vs. unoccupied periods.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "6. Decision Tree/match_rooms.py"
    ```
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '6. Decision Tree' else os.path.abspath('.')
DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')


def analyze_co_location(
    env_df: pd.DataFrame, 
    occ_df: pd.DataFrame, 
    sensor_id: int
) -> List[Dict[str, Union[str, float]]]:
    """Computes correlation and delta metrics between CO2 concentration and candidate rooms.

    Args:
        env_df (pd.DataFrame): Environmental readings containing 'dt', 'co2_ppm', 'temperature_c'.
        occ_df (pd.DataFrame): Occupancy records containing 'dt', 'headcount', and optional 'floorspaceid'.
        sensor_id (int): Numeric sensor ID (e.g., 869 or 326).

    Returns:
        List[Dict[str, Union[str, float]]]: Ranked statistical summary for each candidate room.

    Examples:
        >>> stats = analyze_co_location(env_df, occ_df, 869)
    """
    env_sorted = env_df.sort_values('dt').copy()
    env_sorted['co2_diff'] = env_sorted['co2_ppm'].diff().fillna(0)
    
    results: List[Dict[str, Union[str, float]]] = []

    # Handle datasets with floorspace groupings or single-room representations
    if 'floorspaceid' in occ_df.columns:
        groups = occ_df.groupby('floorspaceid')
    else:
        groups = [(f"Sensor_{sensor_id}_Room", occ_df)]

    for fid, sub_occ in groups:
        sub_occ = sub_occ.sort_values('dt')[['dt', 'headcount']]
        # Backward merge holds headcount state constant until the next recorded transition
        merged = pd.merge_asof(env_sorted, sub_occ, on='dt', direction='backward')
        merged['headcount'] = merged['headcount'].fillna(0)
        
        c_co2 = merged['co2_ppm'].corr(merged['headcount'])
        c_diff = merged['co2_diff'].corr(merged['headcount'])
        c_temp = merged['temperature_c'].corr(merged['headcount'])
        
        occ_co2 = merged.loc[merged['headcount'] > 0, 'co2_ppm'].mean()
        unocc_co2 = merged.loc[merged['headcount'] == 0, 'co2_ppm'].mean()
        co2_delta = occ_co2 - unocc_co2 if not np.isnan(occ_co2) and not np.isnan(unocc_co2) else 0.0
        
        results.append({
            'floorspace_id': str(fid),
            'corr_co2': round(float(c_co2), 3) if not np.isnan(c_co2) else 0.0,
            'corr_diff': round(float(c_diff), 3) if not np.isnan(c_diff) else 0.0,
            'corr_temp': round(float(c_temp), 3) if not np.isnan(c_temp) else 0.0,
            'occ_co2': round(float(occ_co2), 1) if not np.isnan(occ_co2) else 0.0,
            'unocc_co2': round(float(unocc_co2), 1) if not np.isnan(unocc_co2) else 0.0,
            'co2_delta': round(float(co2_delta), 1),
            'mean_headcount': round(float(merged['headcount'].mean()), 2),
            'max_headcount': int(merged['headcount'].max())
        })

    return results


def main() -> None:
    """Loads cleaned telemetry and executes spatial correlation analysis across deployments.

    Returns:
        None
    """
    env_file = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')
    if not os.path.exists(env_file):
        print(f"Environmental dataset not found at: {env_file}")
        return

    df_env = pd.read_csv(env_file)
    df_env['dt'] = pd.to_datetime(df_env['timestamp'])

    for sid in [869, 326]:
        print('=' * 75)
        print(f'ANALYSIS FOR ENVIRONMENTAL SENSOR {sid}')
        print('=' * 75)
        if sid == 869:
            s_env = df_env[(df_env['sensor_id'].astype(str) == '869') & (df_env['dt'] >= '2024-03-15') & (df_env['dt'] <= '2024-06-14 23:59:59') & (df_env['co2_ppm'] != 520.0)].sort_values('dt').copy()
            occ_file = os.path.join(DATA_DIR, '869_occupancy.csv')
        else:
            s_env = df_env[(df_env['sensor_id'].astype(str) == '326') & (df_env['dt'] >= '2025-10-01') & (df_env['dt'] <= '2026-04-22 23:59:59') & (df_env['co2_ppm'] != 525.0)].sort_values('dt').copy()
            occ_file = os.path.join(DATA_DIR, '326_occupancy.csv')

        if not os.path.exists(occ_file):
            print(f"Occupancy dataset not found at: {occ_file}")
            continue

        occ_df = pd.read_csv(occ_file)
        dt_col = 'dt' if 'dt' in occ_df.columns else 'occupancystatuschangedate'
        occ_df['dt'] = pd.to_datetime(occ_df[dt_col].astype(str).str.slice(0, 19))

        stats = analyze_co_location(s_env, occ_df, sid)
        for r in stats:
            print(f"Floorspace {r['floorspace_id']}:")
            print(f"  Corr(CO2, Headcount)     = {r['corr_co2']:+.3f}")
            print(f"  Corr(dCO2/dt, Headcount) = {r['corr_diff']:+.3f}")
            print(f"  Corr(Temp, Headcount)    = {r['corr_temp']:+.3f}")
            print(f"  Mean CO2 Occupied: {r['occ_co2']} ppm | Unoccupied: {r['unocc_co2']} ppm (Delta: {r['co2_delta']:+.1f} ppm)")
            print(f"  Mean HC: {r['mean_headcount']} | Max HC: {r['max_headcount']}")
            print()


if __name__ == '__main__':
    main()
