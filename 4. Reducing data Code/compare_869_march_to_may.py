"""Sensor 869 Telemetry Analysis and ASHRAE Comfort Comparison.

This module evaluates environmental telemetry for Sensor 869 during the active window
from March 1 to May 5, 2024, computing statistical summaries, cross-correlation matrices,
and multi-panel visual comparisons against ASHRAE thermal comfort and indoor air quality standards.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "4. Reducing data Code/compare_869_march_to_may.py"
    ```
"""

import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '4. Reducing data Code' else os.path.abspath('.')
DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, 'plots')
DEFAULT_CSV: str = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')
DEFAULT_PLOT: str = os.path.join(PLOTS_DIR, 'sensor_869_march_to_may_comparison.png')


def analyze_sensor_869(
    csv_path: str = DEFAULT_CSV, 
    output_plot_path: str = DEFAULT_PLOT
) -> Optional[pd.DataFrame]:
    """Generates statistical summaries and comparative IAQ visualizations for Sensor 869.

    Args:
        csv_path (str, optional): Path to cleaned tabular sensor CSV. Defaults to DEFAULT_CSV.
        output_plot_path (str, optional): Destination for generated plot. Defaults to DEFAULT_PLOT.

    Returns:
        Optional[pd.DataFrame]: Filtered DataFrame if successful, else None.

    Examples:
        >>> df_sub = analyze_sensor_869("data.csv", "plot.png")
    """
    if not os.path.exists(csv_path):
        print(f"Notice: Cleaned environmental CSV not found at: {csv_path}. Skipping.")
        return None

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    df['sensor_id'] = df['sensor_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    start_date = "2024-03-01"
    end_date = "2024-05-05 23:59:59"

    mask = (df['sensor_id'] == '869') & (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    sub_df = df[mask].sort_values('timestamp').reset_index(drop=True)

    print(f"Total matching records found for Sensor 869: {len(sub_df):,}")
    if sub_df.empty:
        return None

    cols = ['temperature_c', 'co2_ppm', 'lvoc_ppb']
    cols_present = [c for c in cols if c in sub_df.columns]
    summary = sub_df[cols_present].describe().round(2)
    print("\n--- Summary Statistics (March 1 - May 5, 2024) ---")
    print(summary)

    corr_matrix = sub_df[cols_present].corr().round(3)
    print("\n--- Pearson Correlation Matrix ---")
    print(corr_matrix)

    # Multi-panel visualization
    fig = plt.figure(figsize=(16, 12), dpi=160)
    gs = GridSpec(3, 2, width_ratios=[1.7, 1.0], hspace=0.32, wspace=0.25)

    # Temperature
    ax_temp = fig.add_subplot(gs[0, 0])
    ax_temp.plot(sub_df['timestamp'], sub_df['temperature_c'], '.', markersize=2.5, color='#DC2626', alpha=0.35, label='Temp Raw')
    roll_t = sub_df.set_index('timestamp')['temperature_c'].rolling('24h', min_periods=10).mean()
    ax_temp.plot(roll_t.index, roll_t.values, color='#991B1B', linewidth=2.0, label='24h Rolling Mean')
    ax_temp.axhspan(20, 24, color='#16A34A', alpha=0.15, label='ASHRAE Comfort Zone (20-24°C)')
    ax_temp.set_title("Sensor 869: Temperature (°C) Over Time", fontsize=11, fontweight='bold', loc='left')
    ax_temp.set_ylabel("Temp [°C]", fontsize=10, fontweight='bold')
    ax_temp.grid(True, linestyle=':', alpha=0.5)
    ax_temp.legend(loc='upper right', fontsize=8.5, framealpha=0.9, ncol=3)
    ax_temp.set_xticklabels([])

    # CO2
    ax_co2 = fig.add_subplot(gs[1, 0])
    ax_co2.plot(sub_df['timestamp'], sub_df['co2_ppm'], '.', markersize=2.5, color='#2563EB', alpha=0.35, label='CO2 Raw')
    roll_co2 = sub_df.set_index('timestamp')['co2_ppm'].rolling('24h', min_periods=10).mean()
    ax_co2.plot(roll_co2.index, roll_co2.values, color='#1E40AF', linewidth=2.0, label='24h Rolling Mean')
    ax_co2.axhline(1000, color='#DC2626', linestyle='--', linewidth=1.5, label='ASHRAE Threshold (1,000 ppm)')
    ax_co2.set_title("Sensor 869: Carbon Dioxide (CO2 ppm) Over Time", fontsize=11, fontweight='bold', loc='left')
    ax_co2.set_ylabel("CO2 [ppm]", fontsize=10, fontweight='bold')
    ax_co2.grid(True, linestyle=':', alpha=0.5)
    ax_co2.legend(loc='upper right', fontsize=8.5, framealpha=0.9, ncol=3)
    ax_co2.set_xticklabels([])

    # LVOC
    ax_voc = fig.add_subplot(gs[2, 0])
    ax_voc.plot(sub_df['timestamp'], sub_df['lvoc_ppb'], '.', markersize=2.5, color='#F59E0B', alpha=0.35, label='LVOC Raw')
    roll_voc = sub_df.set_index('timestamp')['lvoc_ppb'].rolling('24h', min_periods=10).mean()
    ax_voc.plot(roll_voc.index, roll_voc.values, color='#B45309', linewidth=2.0, label='24h Rolling Mean')
    ax_voc.axhline(500, color='#DC2626', linestyle='--', linewidth=1.5, label='Elevated Threshold (500 ppb)')
    ax_voc.set_title("Sensor 869: Light Volatile Organic Compounds (LVOC ppb) Over Time", fontsize=11, fontweight='bold', loc='left')
    ax_voc.set_ylabel("LVOC [ppb]", fontsize=10, fontweight='bold')
    ax_voc.set_xlabel("Date", fontsize=10, fontweight='bold')
    ax_voc.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_voc.grid(True, linestyle=':', alpha=0.5)
    ax_voc.legend(loc='upper right', fontsize=8.5, framealpha=0.9, ncol=3)

    # Right Column: Histograms / KDE
    ax_thist = fig.add_subplot(gs[0, 1])
    ax_thist.hist(sub_df['temperature_c'].dropna(), bins=40, color='#DC2626', edgecolor='black', alpha=0.7)
    ax_thist.set_title("Temperature Distribution", fontsize=11, fontweight='bold')
    ax_thist.set_xlabel("°C")

    ax_chist = fig.add_subplot(gs[1, 1])
    ax_chist.hist(sub_df['co2_ppm'].dropna(), bins=40, color='#2563EB', edgecolor='black', alpha=0.7)
    ax_chist.set_title("CO2 Distribution", fontsize=11, fontweight='bold')
    ax_chist.set_xlabel("ppm")

    ax_vhist = fig.add_subplot(gs[2, 1])
    ax_vhist.hist(sub_df['lvoc_ppb'].dropna(), bins=40, color='#F59E0B', edgecolor='black', alpha=0.7)
    ax_vhist.set_title("LVOC Distribution", fontsize=11, fontweight='bold')
    ax_vhist.set_xlabel("ppb")

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic plot to: {output_plot_path}")
    return sub_df


def main() -> None:
    """Executes the statistical and visual comparison for Sensor 869.

    Returns:
        None
    """
    analyze_sensor_869()


if __name__ == '__main__':
    main()
