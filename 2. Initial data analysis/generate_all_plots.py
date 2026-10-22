"""Comprehensive Variable Dashboard and Diurnal Diagnostic Plot Generator.

This module iterates across all eleven recorded environmental variables (temperature,
relative humidity, CO2, LVOC, formaldehyde, particulate matter fractions PM1.0, PM2.5,
PM4.0, PM10, atmospheric pressure, and battery voltage), rendering multi-panel diagnostic
dashboards with long-term time series, 24-hour diurnal diurnal cycles, and cross-sensor
distribution boxplots.

Typical Usage:
    Run directly from terminal or IDE:
    ```bash
    python "2. Initial data analysis/generate_all_plots.py"
    ```
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == '2. Initial data analysis' else os.path.abspath('.')

DATA_DIR: str = os.path.join(PROJECT_ROOT, '5. Reduced Data')
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, '3. Initial plots')
DEFAULT_CSV: str = os.path.join(DATA_DIR, '5EnvSensor_Cleaned_Tabular.csv')

SENSOR_COLORS: Dict[str, str] = {
    '6012002000869': '#2563EB',
    '869': '#2563EB',
    '6012002000326': '#10B981',
    '326': '#10B981',
    '6012002000241': '#F59E0B',
    '241': '#F59E0B',
    '6012002000227': '#8B5CF6',
    '227': '#8B5CF6',
    '6012002000777': '#EC4899',
    '777': '#EC4899',
}

SENSOR_LABELS: Dict[str, str] = {
    '6012002000869': 'Sensor ...869 (Active 2024)',
    '869': 'Sensor ...869 (Active 2024)',
    '6012002000326': 'Sensor ...326 (Active 2025/26)',
    '326': 'Sensor ...326 (Active 2025/26)',
    '6012002000241': 'Sensor ...241 (Static Ref)',
    '241': 'Sensor ...241 (Static Ref)',
    '6012002000227': 'Sensor ...227 (Static Ref)',
    '227': 'Sensor ...227 (Static Ref)',
    '6012002000777': 'Sensor ...777 (Static Ref)',
    '777': 'Sensor ...777 (Static Ref)',
}

VARIABLE_CONFIGS: Dict[str, Dict] = {
    'temperature_c': {
        'name': 'Temperature',
        'unit': '°C',
        'ref_lines': [(20, 24, 'green', 'Comfort Zone (20-24°C)')],
        'desc': 'Ambient air temperature measured by the sensor nodes.'
    },
    'humidity_pct': {
        'name': 'Relative Humidity',
        'unit': '% RH',
        'ref_lines': [(30, 60, 'teal', 'ASHRAE Comfort Zone (30-60%)'), (70, 70, 'red', 'Mold Growth Risk (>70%)')],
        'desc': 'Relative humidity of the surrounding air.'
    },
    'co2_ppm': {
        'name': 'Carbon Dioxide (CO2)',
        'unit': 'ppm',
        'ref_lines': [(420, 420, 'gray', 'Outdoor Ambient (~420 ppm)'), (800, 800, 'orange', 'Recommended Indoor (<800 ppm)'), (1000, 1000, 'red', 'Ventilation Limit (1000 ppm)')],
        'desc': 'Carbon dioxide concentration, key indicator of occupancy and ventilation.'
    },
    'lvoc_ppb': {
        'name': 'Light Volatile Organic Compounds (LVOC / TVOC)',
        'unit': 'ppb',
        'ref_lines': [(65, 65, 'green', 'Good Quality (<65 ppb)'), (220, 220, 'orange', 'Moderate Quality (<220 ppb)')],
        'desc': 'Total volatile organic compound concentrations in parts per billion.'
    },
    'formaldehyde_ugm3': {
        'name': 'Formaldehyde (HCHO)',
        'unit': 'µg/m³',
        'ref_lines': [(30, 30, 'green', 'Typical Clean Indoor (<30 µg/m³)'), (100, 100, 'red', 'WHO 30-min Guideline (100 µg/m³)')],
        'desc': 'Airborne formaldehyde concentration, a common indoor air toxicant.'
    },
    'pm1_ugm3': {
        'name': 'Particulate Matter PM1.0',
        'unit': 'µg/m³',
        'ref_lines': [(10, 10, 'gray', 'Indicative Baseline (<10 µg/m³)')],
        'desc': 'Ultrafine suspended particulate matter smaller than 1.0 micrometer.'
    },
    'pm25_ugm3': {
        'name': 'Particulate Matter PM2.5',
        'unit': 'µg/m³',
        'ref_lines': [(15, 15, 'orange', 'WHO 24h Guideline (15 µg/m³)'), (25, 25, 'red', 'UK Daily Air Quality Limit (25 µg/m³)')],
        'desc': 'Fine respirable particles penetrating deep into pulmonary alveoli.'
    },
    'pm4_ugm3': {
        'name': 'Particulate Matter PM4.0',
        'unit': 'µg/m³',
        'ref_lines': [(25, 25, 'orange', 'Thoracic Fraction Guideline')],
        'desc': 'Thoracic dust particles capable of passing the larynx.'
    },
    'pm10_ugm3': {
        'name': 'Particulate Matter PM10',
        'unit': 'µg/m³',
        'ref_lines': [(45, 45, 'red', 'WHO 24h Guideline (45 µg/m³)')],
        'desc': 'Inhalable coarse particles such as mineral dust and pollen.'
    },
    'pressure_mb': {
        'name': 'Barometric Pressure',
        'unit': 'mbar',
        'ref_lines': [(1013.25, 1013.25, 'gray', 'Standard Sea Level (1013.25 mbar)')],
        'desc': 'Atmospheric barometric pressure in millibars (hPa).'
    },
    'battery_v': {
        'name': 'Battery Voltage',
        'unit': 'V',
        'ref_lines': [(3.6, 3.6, 'green', 'Nominal Lithium-Thionyl (3.6V)'), (3.0, 3.0, 'red', 'Depletion Cutoff (3.0V)')],
        'desc': 'Node internal battery state indicating sensor power health.'
    }
}


def plot_variable_dashboard(
    vdf: pd.DataFrame, 
    col: str, 
    conf: Dict, 
    out_dir: str
) -> None:
    """Renders a 3-panel diagnostic dashboard for a single environmental telemetry variable.

    Args:
        vdf (pd.DataFrame): Filtered DataFrame containing timestamps and column data.
        col (str): Column name in DataFrame to plot.
        conf (Dict): Configuration dictionary specifying unit, name, and thresholds.
        out_dir (str): Output directory for saving the figure.

    Returns:
        None

    Examples:
        >>> plot_variable_dashboard(df, "co2_ppm", VARIABLE_CONFIGS["co2_ppm"], "plots/")
    """
    fig = plt.figure(figsize=(16, 10), dpi=160)
    gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.22)

    # Panel 1: Full Longitudinal Time Series
    ax_time = fig.add_subplot(gs[0, :])
    for sid, group in vdf.groupby('sensor_id'):
        color = SENSOR_COLORS.get(str(sid), '#666666')
        lbl = SENSOR_LABELS.get(str(sid), f"Sensor {sid}")
        group_sorted = group.sort_values('timestamp')
        sub_group = group_sorted.iloc[::max(1, len(group_sorted) // 2500)]
        ax_time.plot(sub_group['timestamp'], sub_group[col], '.', markersize=2.0, color=color, alpha=0.35)

        if len(group_sorted) > 100:
            roll = group_sorted.set_index('timestamp')[col].rolling('24h', min_periods=10).mean()
            ax_time.plot(roll.index, roll.values, color=color, linewidth=2.0, alpha=0.9, label=f"{lbl} [24h Trend]")

    for ref in conf.get('ref_lines', []):
        if len(ref) == 4 and ref[0] != ref[1]:
            ax_time.axhspan(ref[0], ref[1], color=ref[2], alpha=0.12, label=ref[3])
        elif len(ref) >= 3:
            ax_time.axhline(ref[0], color=ref[2], linestyle='--', linewidth=1.2, alpha=0.7, label=ref[3])

    ax_time.set_title(f"Time Series: {conf['name']} ({conf['unit']}) Across All 5 Sensors", fontsize=13, fontweight='bold', pad=10)
    ax_time.set_ylabel(f"{conf['name']} [{conf['unit']}]", fontsize=11)
    ax_time.grid(True, linestyle=':', alpha=0.5)
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_time.legend(loc='upper right', fontsize=8.5, framealpha=0.9, ncol=2)

    # Panel 2: 24-Hour Diurnal Cycle
    ax_diurnal = fig.add_subplot(gs[1, 0])
    for sid in ['869', '6012002000869', '326', '6012002000326']:
        s_data = vdf[vdf['sensor_id'] == sid]
        if len(s_data) > 100 and s_data[col].std() > 0.001:
            color = SENSOR_COLORS.get(str(sid), '#2563EB')
            lbl = SENSOR_LABELS.get(str(sid), f"Sensor {sid}")
            hourly = s_data.groupby('hour')[col].agg(['mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
            hourly.columns = ['mean', 'q25', 'q75']
            hours = hourly.index
            ax_diurnal.plot(hours, hourly['mean'], 'o-', color=color, linewidth=2.2, label=f"{lbl} Mean")
            ax_diurnal.fill_between(hours, hourly['q25'], hourly['q75'], color=color, alpha=0.18)

    ax_diurnal.set_title("Daily Diurnal Cycle (24-Hour Profile)", fontsize=11, fontweight='bold')
    ax_diurnal.set_xlabel("Hour of Day (00:00 - 23:00 Local Time)", fontsize=10)
    ax_diurnal.set_ylabel(f"{conf['name']} [{conf['unit']}]", fontsize=10)
    ax_diurnal.set_xticks(range(0, 24, 2))
    ax_diurnal.grid(True, linestyle=':', alpha=0.5)
    ax_diurnal.legend(loc='best', fontsize=8.5, framealpha=0.9)

    # Panel 3: Distribution Boxplot
    ax_dist = fig.add_subplot(gs[1, 1])
    sensor_groups, sensor_names_list, colors_list = [], [], []
    for sid in sorted(vdf['sensor_id'].unique()):
        s_vals = vdf[vdf['sensor_id'] == sid][col].dropna()
        if len(s_vals) > 0:
            sensor_groups.append(s_vals)
            sensor_names_list.append(f"...{str(sid)[-3:]}")
            colors_list.append(SENSOR_COLORS.get(str(sid), '#666666'))

    if sensor_groups:
        bp = ax_dist.boxplot(sensor_groups, patch_artist=True, labels=sensor_names_list, showfliers=False, medianprops=dict(color='black', linewidth=1.8))
        for patch, c in zip(bp['boxes'], colors_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)

    ax_dist.set_title("Distribution & Spread Across Sensors", fontsize=11, fontweight='bold')
    ax_dist.set_xlabel("Sensor Node (Last 3 Digits)", fontsize=10)
    ax_dist.set_ylabel(f"{conf['name']} [{conf['unit']}]", fontsize=10)
    ax_dist.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f"MMA3001 Environmental Sensor Analysis: {conf['name']}", fontsize=15, fontweight='bold', y=0.98)
    fig_path = os.path.join(out_dir, f"var_{col}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved dashboard figure to: {fig_path}")


def generate_all_variable_plots(
    csv_path: str = DEFAULT_CSV, 
    out_dir: str = PLOTS_DIR
) -> None:
    """Iterates through all environmental telemetry variables and generates dashboard plots.

    Args:
        csv_path (str, optional): Path to cleaned tabular sensor CSV. Defaults to DEFAULT_CSV.
        out_dir (str, optional): Target directory for saving PNG figures. Defaults to PLOTS_DIR.

    Returns:
        None

    Raises:
        FileNotFoundError: If the source CSV dataset does not exist.

    Examples:
        >>> generate_all_variable_plots("cleaned.csv", "plots/")
    """
    if not os.path.exists(csv_path):
        print(f"Notice: Cleaned telemetry CSV not found at: {csv_path}. Skipping plot generation.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Starting plot generation script from: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['sensor_id'] = df['sensor_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    for col, conf in VARIABLE_CONFIGS.items():
        if col in df.columns:
            vdf = df[['sensor_id', 'timestamp', 'hour', col]].dropna(subset=[col])
            if not vdf.empty:
                print(f"Generating dashboard for: {conf['name']} ({col})...")
                plot_variable_dashboard(vdf, col, conf, out_dir)


def main() -> None:
    """Executes the full variable plotting suite.

    Returns:
        None
    """
    generate_all_variable_plots()


if __name__ == '__main__':
    main()
