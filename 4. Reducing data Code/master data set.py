"""Master Aligned Environmental and Occupancy Dataset Creator.

This module aligns environmental sensor readings with room occupancy counts using
backward-fill chronological merging (`merge_asof`), engineers temporal and rate of
change dynamics, and generates binary occupancy indicators.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "4. Reducing data Code/master data set.py"
    ```
"""

import os
from typing import Optional

import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')
DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')


def align_master_dataset(
    env_filepath: str, 
    occ_filepath: str, 
    output_prefix: str = "master_aligned_dataset"
) -> Optional[pd.DataFrame]:
    """Pairs environmental sensor readings with ground-truth occupancy headcounts.

    Args:
        env_filepath (str): Path to environmental sensor CSV dataset.
        occ_filepath (str): Path to occupancy Excel or CSV dataset.
        output_prefix (str, optional): Base filename for output artifacts.
            Defaults to "master_aligned_dataset".

    Returns:
        Optional[pd.DataFrame]: Aligned master DataFrame if files exist, else None.

    Raises:
        KeyError: If required timestamp or feature columns are missing from either file.

    Examples:
        >>> df_aligned = align_master_dataset("env.csv", "occ.csv")
    """
    if not os.path.exists(env_filepath) or not os.path.exists(occ_filepath):
        print(f"Notice: Source files not found ({env_filepath} or {occ_filepath}). Skipping alignment.")
        return None

    print(f"Loading environmental telemetry from: {env_filepath}")
    env_df = pd.read_csv(env_filepath)
    env_df['dt'] = pd.to_datetime(env_df['timestamp']).dt.tz_localize(None)

    print(f"Loading occupancy data from: {occ_filepath}")
    if occ_filepath.endswith('.xlsx') or occ_filepath.endswith('.xls'):
        occ_df = pd.read_excel(occ_filepath)
    else:
        occ_df = pd.read_csv(occ_filepath)

    time_col = 'dt' if 'dt' in occ_df.columns else ('collecteddate' if 'collecteddate' in occ_df.columns else 'occupancystatuschangedate')
    occ_df['dt'] = pd.to_datetime(occ_df[time_col].astype(str).str.slice(0, 19))

    # Filter for default co-located room if floorspace column exists
    if 'floorspaceid' in occ_df.columns:
        selected_room = '941f0352-bb69-47d0-a9c5-ac29eb4f7319'
        occ_room = occ_df[occ_df['floorspaceid'] == selected_room].sort_values('dt').copy()
    else:
        occ_room = occ_df.sort_values('dt').copy()

    env_sorted = env_df.sort_values('dt').copy()

    # Pair environmental timestamps with most recent headcount state change
    master = pd.merge_asof(
        env_sorted,
        occ_room[['dt', 'headcount']],
        on='dt',
        direction='backward'
    )

    master['headcount'] = master['headcount'].fillna(0).astype(int)

    # Physics and schedule feature engineering
    master['co2_rate_of_change'] = master['co2_ppm'].diff().fillna(0)
    master['hour'] = master['dt'].dt.hour
    master['day_of_week'] = master['dt'].dt.dayofweek
    master['is_weekend'] = master['day_of_week'].isin([5, 6]).astype(int)
    master['is_occupied'] = (master['headcount'] > 0).astype(int)

    out_csv = os.path.join(DATA_DIR, f"{output_prefix}.csv")
    master.to_csv(out_csv, index=False)
    print(f"Successfully aligned {len(master):,} rows to: {out_csv}")
    return master


def main() -> None:
    """Executes master environmental and occupancy alignment pipeline.

    Returns:
        None
    """
    default_env = os.path.join(DATA_DIR, '869_train.csv')
    default_occ = os.path.join(DATA_DIR, '869_occupancy.csv')
    align_master_dataset(default_env, default_occ)


if __name__ == '__main__':
    main()
