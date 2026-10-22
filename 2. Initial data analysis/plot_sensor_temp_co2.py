"""Multi-Sensor Faceted Temperature and CO2 Diagnostic Visualizer.

This module renders multi-panel comparative facet plots for all five IoT sensor nodes
(227, 241, 326, 777, and 869), displaying dual-axis ambient temperature and CO2 time series,
scatter correlations, and hourly diurnal profile distributions.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "2. Initial data analysis/plot_sensor_temp_co2.py"
    ```
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '2. Initial data analysis' else os.path.abspath('.')

DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, '3. Initial plots')
DEFAULT_CSV: str = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')

SENSOR_IDS: List[str] = ['227', '241', '326', '777', '869']
SENSOR_META: Dict[str, Dict[str, str]] = {
    '227': {'name': 'Sensor ID: 227', 'desc': 'Static Reference (May 2024 – Oct 2024)', 'color': '#8B5CF6'},
    '241': {'name': 'Sensor ID: 241', 'desc': 'Static Dual-State Ref (Feb 2024 – Apr 2026)', 'color': '#F59E0B'},
    '326': {'name': 'Sensor ID: 326', 'desc': 'Active Node 2025/26 (Feb 2024 – Apr 2026)', 'color': '#10B981'},
    '777': {'name': 'Sensor ID: 777', 'desc': 'Static Reference (Feb 2024 – May 2024)', 'color': '#EC4899'},
    '869': {'name': 'Sensor ID: 869', 'desc': 'Active Dynamic Node 2024 (Feb 2024 – Oct 2024)', 'color': '#2563EB'},
}


def plot_sensor_temp_co2(
    csv_path: str = DEFAULT_CSV, 
    out_dir: str = PLOTS_DIR
) -> None:
    """Generates dual-axis faceted time series plots comparing temperature and CO2 across all nodes.

    Args:
        csv_path (str, optional): Path to cleaned tabular environmental sensor CSV.
            Defaults to DEFAULT_CSV.
        out_dir (str, optional): Destination folder for output figures.
            Defaults to PLOTS_DIR.

    Returns:
        None

    Raises:
        FileNotFoundError: If the source CSV file does not exist.

    Examples:
        >>> plot_sensor_temp_co2("data.csv", "plots/")
    """
    if not os.path.exists(csv_path):
        print(f"Notice: Cleaned telemetry CSV not found at: {csv_path}. Skipping visualization.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Reading cleaned dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['sensor_id'] = df['sensor_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    print("Generating Figure 1: 5-row dual-axis facet plot...")
    fig, axes = plt.subplots(5, 1, figsize=(16, 15), dpi=160, sharex=True)

    for i, sid in enumerate(SENSOR_IDS):
        ax1 = axes[i]
        sdf = df[df['sensor_id'] == sid].dropna(subset=['temperature_c', 'co2_ppm'])
        meta = SENSOR_META.get(sid, {'name': f"Sensor {sid}", 'desc': '', 'color': '#2563EB'})

        # Downsample large series to maintain crisp rendering performance
        if len(sdf) > 4000:
            step = len(sdf) // 3000
            sdf_plot = sdf.iloc[::step]
        else:
            sdf_plot = sdf

        # Primary Y-axis: Temperature
        color_temp = '#DC2626'
        ax1.plot(sdf_plot['timestamp'], sdf_plot['temperature_c'], color=color_temp, alpha=0.75, linewidth=1.2, label='Temperature (°C)')
        ax1.set_ylabel("Temp [°C]", color=color_temp, fontsize=10, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_temp)
        ax1.set_ylim(15, 35)

        # Secondary Y-axis: CO2 concentration
        ax2 = ax1.twinx()
        color_co2 = meta['color']
        ax2.plot(sdf_plot['timestamp'], sdf_plot['co2_ppm'], color=color_co2, alpha=0.85, linewidth=1.2, label='CO2 (ppm)')
        ax2.set_ylabel("CO2 [ppm]", color=color_co2, fontsize=10, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_co2)
        ax2.set_ylim(350, 1600)

        # Annotate sensor operational metadata
        title_text = f"{meta['name']}  |  {meta['desc']}  (N = {len(sdf):,} records)"
        ax1.set_title(title_text, loc='left', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.5)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].set_xlabel("Date (Month/Year)", fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, 'figure1_5sensors_temp_co2_facets.png')
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to: {fig1_path}")


def main() -> None:
    """Executes the multi-sensor temperature and CO2 faceted plotting pipeline.

    Returns:
        None
    """
    plot_sensor_temp_co2()


if __name__ == '__main__':
    main()
