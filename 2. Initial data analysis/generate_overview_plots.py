"""Exploratory Data Analysis: Multi-Variable Overview and IAQ Correlation Visualizer.

This module generates comprehensive architectural and environmental overview figures
from the cleaned IoT telemetry dataset, including:
1. Particulate matter spectrum analysis (PM1.0, PM2.5, PM4.0, PM10) and size ratios.
2. ASHRAE Standard 55 thermal comfort compliance scatter and seasonal drift.
3. Pearson cross-correlation heatmap across all measured physical variables.
4. Diurnal weekday vs. weekend 24-hour occupancy waves (CO2, LVOC, PM2.5, Temperature).

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "2. Initial data analysis/generate_overview_plots.py"
    ```
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '2. Initial data analysis' else os.path.abspath('.')

DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, '3. Initial plots')
DEFAULT_CSV: str = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')


def generate_overview_plots(
    csv_path: str = DEFAULT_CSV, 
    out_dir: str = PLOTS_DIR
) -> None:
    """Generates overview diagnostic plots for thermal comfort, particulate matter, and correlations.

    Args:
        csv_path (str, optional): Path to cleaned tabular sensor CSV. Defaults to DEFAULT_CSV.
        out_dir (str, optional): Destination directory for PNG plots. Defaults to PLOTS_DIR.

    Returns:
        None

    Raises:
        FileNotFoundError: If the source CSV file does not exist.

    Examples:
        >>> generate_overview_plots("data.csv", "plots/")
    """
    if not os.path.exists(csv_path):
        print(f"Notice: Cleaned telemetry CSV not found at: {csv_path}. Skipping overview plots.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading sensor telemetry from: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['sensor_id'] = df['sensor_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    # Standardize short sensor ID aliases (e.g. '869' or '6012002000869')
    sid_869_mask = df['sensor_id'].str.endswith('869')
    df_869 = df[sid_869_mask].sort_values('timestamp')

    # 1. Particulate Matter Breakdown
    if 'pm10_ugm3' in df.columns and not df_869.empty:
        print("Generating Particulate Matter comparison plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=160, gridspec_kw={'height_ratios': [1.5, 1]})
        df_869_sub = df_869.iloc[::max(1, len(df_869) // 2500)]

        ax1.plot(df_869_sub['timestamp'], df_869_sub['pm10_ugm3'], label='PM10 (Coarse, <10µm)', color='#DC2626', alpha=0.7, linewidth=1.2)
        ax1.plot(df_869_sub['timestamp'], df_869_sub['pm4_ugm3'], label='PM4.0 (Thoracic, <4µm)', color='#EA580C', alpha=0.75, linewidth=1.2)
        ax1.plot(df_869_sub['timestamp'], df_869_sub['pm25_ugm3'], label='PM2.5 (Fine, <2.5µm)', color='#059669', alpha=0.8, linewidth=1.2)
        ax1.plot(df_869_sub['timestamp'], df_869_sub['pm1_ugm3'], label='PM1.0 (Ultrafine, <1µm)', color='#2563EB', alpha=0.85, linewidth=1.2)

        ax1.axhline(15, color='#059669', linestyle='--', linewidth=1.5, label='WHO PM2.5 24h Guideline (15 µg/m³)')
        ax1.axhline(45, color='#DC2626', linestyle='--', linewidth=1.5, label='WHO PM10 24h Guideline (45 µg/m³)')

        ax1.set_title("Particulate Matter Spectrum (PM1.0, PM2.5, PM4.0, PM10) Over Time [Sensor ...869]", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Concentration [µg/m³]", fontsize=11)
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.legend(loc='upper right', fontsize=9.5, framealpha=0.9, ncol=3)

        ratio_hourly = df_869.groupby('hour')[['pm1_ugm3', 'pm25_ugm3', 'pm4_ugm3', 'pm10_ugm3']].mean()
        ax2.plot(ratio_hourly.index, ratio_hourly['pm10_ugm3'], 'o-', label='PM10 Mean', color='#DC2626', linewidth=2)
        ax2.plot(ratio_hourly.index, ratio_hourly['pm4_ugm3'], 's-', label='PM4.0 Mean', color='#EA580C', linewidth=2)
        ax2.plot(ratio_hourly.index, ratio_hourly['pm25_ugm3'], '^-', label='PM2.5 Mean', color='#059669', linewidth=2)
        ax2.plot(ratio_hourly.index, ratio_hourly['pm1_ugm3'], 'd-', label='PM1.0 Mean', color='#2563EB', linewidth=2)

        ax2.set_title("24-Hour Diurnal Profile of Particulate Matter Sizes", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Hour of Day (00:00 - 23:00 Local Time)", fontsize=10)
        ax2.set_ylabel("Hourly Mean [µg/m³]", fontsize=10)
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(loc='best', fontsize=9, ncol=4)

        plt.tight_layout()
        pm_path = os.path.join(out_dir, "overview_particulate_matter.png")
        plt.savefig(pm_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Particulate Matter overview to: {pm_path}")

    # 2. Thermal Comfort (ASHRAE 55)
    if 'temperature_c' in df.columns and 'humidity_pct' in df.columns:
        print("Generating Thermal Comfort plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=160)

        df_dyn = df[df['sensor_id'].str.endswith('869') | df['sensor_id'].str.endswith('326')].dropna(subset=['temperature_c', 'humidity_pct'])
        df_dyn_sub = df_dyn.sample(min(len(df_dyn), 8000), random_state=42)

        for sid, grp in df_dyn_sub.groupby('sensor_id'):
            c = '#2563EB' if sid.endswith('869') else '#10B981'
            lbl = 'Sensor 869 (2024)' if sid.endswith('869') else 'Sensor 326 (2025/26)'
            ax1.scatter(grp['temperature_c'], grp['humidity_pct'], alpha=0.25, s=15, color=c, label=lbl)

        comfort_box = Rectangle((20, 30), 4, 30, linewidth=2, edgecolor='#16A34A', facecolor='#DCFCE7', alpha=0.35, label='ASHRAE 55 Comfort Zone (20-24°C, 30-60%)')
        ax1.add_patch(comfort_box)
        ax1.axhline(70, color='#DC2626', linestyle='--', linewidth=1.2, label='High Humidity / Mold Risk (>70%)')
        ax1.axhline(30, color='#D97706', linestyle=':', linewidth=1.2, label='Dry Air Threshold (<30%)')

        ax1.set_title("Thermal Comfort Scatter: Temperature vs Humidity", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Temperature [°C]", fontsize=11)
        ax1.set_ylabel("Relative Humidity [% RH]", fontsize=11)
        ax1.set_xlim(10, 38)
        ax1.set_ylim(15, 85)
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=8.5)

        if not df_869.empty:
            monthly = df_869.set_index('timestamp').resample('ME')[['temperature_c', 'humidity_pct']].mean().dropna()
            ax2_twin = ax2.twinx()
            months_str = [m.strftime('%b %Y') for m in monthly.index]
            p1 = ax2.plot(months_str, monthly['temperature_c'], 'o-', color='#DC2626', linewidth=2.5, label='Monthly Mean Temperature')
            p2 = ax2_twin.plot(months_str, monthly['humidity_pct'], 's--', color='#2563EB', linewidth=2.5, label='Monthly Mean Humidity')
            ax2.set_title("Seasonal Trajectory: Mean Temp vs Humidity [Sensor ...869]", fontsize=13, fontweight='bold')
            ax2.set_xlabel("Month", fontsize=11)
            ax2.set_ylabel("Temperature [°C]", color='#DC2626', fontsize=11)
            ax2_twin.set_ylabel("Relative Humidity [% RH]", color='#2563EB', fontsize=11)
            ax2.grid(True, linestyle=':', alpha=0.5)
            lines = p1 + p2
            labs = [l.get_label() for l in lines]
            ax2.legend(lines, labs, loc='upper left', fontsize=9.5)

        plt.tight_layout()
        tc_path = os.path.join(out_dir, "overview_thermal_comfort.png")
        plt.savefig(tc_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Thermal Comfort overview to: {tc_path}")

    # 3. Correlation Matrix Heatmap
    print("Generating Correlation Matrix Heatmap...")
    cols_corr = ['temperature_c', 'humidity_pct', 'co2_ppm', 'lvoc_ppb', 'formaldehyde_ugm3', 
                 'pm1_ugm3', 'pm25_ugm3', 'pm4_ugm3', 'pm10_ugm3', 'pressure_mb', 'battery_v']
    avail_cols = [c for c in cols_corr if c in df.columns]

    if len(avail_cols) > 2 and not df_869.empty:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
        corr_matrix = df_869[avail_cols].corr()

        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson Correlation Coefficient (r)')

        ax.set_xticks(range(len(avail_cols)))
        ax.set_yticks(range(len(avail_cols)))
        ax.set_xticklabels(avail_cols, rotation=45, ha='right', fontsize=9.5)
        ax.set_yticklabels(avail_cols, fontsize=9.5)

        for i in range(len(avail_cols)):
            for j in range(len(avail_cols)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > 0.45 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8.5, fontweight='bold' if abs(val) > 0.5 else 'normal')

        ax.set_title("Environmental Variable Correlation Matrix [Sensor ...869 Active Window]", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        corr_path = os.path.join(out_dir, "overview_correlation_matrix.png")
        plt.savefig(corr_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Correlation Heatmap to: {corr_path}")


def main() -> None:
    """Executes the multi-variable overview plotting pipeline.

    Returns:
        None
    """
    generate_overview_plots()


if __name__ == '__main__':
    main()
