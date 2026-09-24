import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

print("Starting plot generation script...")
t0 = time.time()

csv_path = r'C:\Users\gregr\.gemini\antigravity\brain\fa95884d-3e0c-4586-8e63-03ba1bc71dea\scratch\cleaned_sensor_data.csv'
out_dir = r'C:\Users\gregr\.gemini\antigravity\brain\fa95884d-3e0c-4586-8e63-03ba1bc71dea\plots'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date

# Sensor color mapping
sensor_colors = {
    '6012002000869': '#2563EB', # Vibrant Blue (2024 active)
    '6012002000326': '#10B981', # Emerald Green (2025/2026 active)
    '6012002000241': '#F59E0B', # Amber Orange
    '6012002000227': '#8B5CF6', # Violet
    '6012002000777': '#EC4899', # Pink / Rose
}

sensor_labels = {
    '6012002000869': 'Sensor ...869 (Active 2024)',
    '6012002000326': 'Sensor ...326 (Active 2025/26)',
    '6012002000241': 'Sensor ...241 (Static Ref)',
    '6012002000227': 'Sensor ...227 (Static Ref)',
    '6012002000777': 'Sensor ...777 (Static Ref)',
}

# Variable configurations
var_configs = {
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
        'name': 'Carbon Dioxide (CO₂)',
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
        'ref_lines': [(10, 10, 'green', 'Low PM1 Level (<10 µg/m³)')],
        'desc': 'Ultrafine airborne particles with aerodynamic diameter < 1.0 µm.'
    },
    'pm25_ugm3': {
        'name': 'Particulate Matter PM2.5',
        'unit': 'µg/m³',
        'ref_lines': [(5, 5, 'green', 'WHO Annual Guideline (5 µg/m³)'), (15, 15, 'orange', 'WHO 24h Guideline (15 µg/m³)')],
        'desc': 'Fine inhalable airborne particles with diameter < 2.5 µm.'
    },
    'pm4_ugm3': {
        'name': 'Particulate Matter PM4.0',
        'unit': 'µg/m³',
        'ref_lines': [(25, 25, 'green', 'Moderate Benchmark (25 µg/m³)')],
        'desc': 'Thoracic airborne particles with diameter < 4.0 µm.'
    },
    'pm10_ugm3': {
        'name': 'Particulate Matter PM10',
        'unit': 'µg/m³',
        'ref_lines': [(15, 15, 'green', 'WHO Annual Guideline (15 µg/m³)'), (45, 45, 'orange', 'WHO 24h Guideline (45 µg/m³)')],
        'desc': 'Coarse inhalable particles with diameter < 10 µm (dust, pollen, smoke).'
    },
    'pressure_mb': {
        'name': 'Barometric Pressure',
        'unit': 'mb (hPa)',
        'ref_lines': [(1013.25, 1013.25, 'gray', 'Standard Atmospheric Pressure (1013.25 mb)')],
        'desc': 'Atmospheric air pressure in millibars (hectopascals).'
    },
    'battery_v': {
        'name': 'Battery Voltage',
        'unit': 'V',
        'ref_lines': [(4.2, 4.2, 'green', 'Full Charge (4.2V)'), (3.7, 3.7, 'orange', 'Nominal (3.7V)'), (3.5, 3.5, 'red', 'Cutoff / Low (3.5V)')],
        'desc': 'Node internal battery voltage monitoring power health.'
    }
}

print("Plotting individual variables...")

for col, conf in var_configs.items():
    print(f"Generating plot for {conf['name']} ({col})...")
    fig = plt.figure(figsize=(16, 11), dpi=160)
    gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
    
    # Non-null data for this variable
    vdf = df.dropna(subset=[col]).copy()
    if len(vdf) == 0:
        plt.close(fig)
        continue
        
    # Panel 1: Full Time Series (Top Left & Right spanning or Top Panel)
    # Let's make Top Panel span both columns: gs[0, :]
    ax_time = fig.add_subplot(gs[0, :])
    
    # Plot each sensor
    for sid, group in vdf.groupby('sensor_id'):
        color = sensor_colors.get(str(sid), '#666666')
        lbl = sensor_labels.get(str(sid), f"Sensor {sid}")
        
        # Subsample for display if very dense
        if len(group) > 5000:
            sub = group.iloc[::max(1, len(group)//3000)]
        else:
            sub = group
            
        ax_time.plot(sub['timestamp'], sub[col], '.', markersize=2, alpha=0.35, color=color, label=lbl)
        
        # Compute and plot 24h rolling median for active sensors
        if sid in ['6012002000869', '6012002000326']:
            group_sorted = group.sort_values('timestamp')
            roll = group_sorted.set_index('timestamp')[col].rolling('24h', min_periods=10).mean()
            ax_time.plot(roll.index, roll.values, color=color, linewidth=2.0, alpha=0.9, label=f"{lbl} [24h Trend]")
            
    # Reference lines
    for ref in conf['ref_lines']:
        if len(ref) == 4 and ref[0] != ref[1]: # Zone
            ax_time.axhspan(ref[0], ref[1], color=ref[2], alpha=0.12, label=ref[3])
        elif len(ref) >= 3:
            ax_time.axhline(ref[0], color=ref[2], linestyle='--', linewidth=1.2, alpha=0.7, label=ref[3])
            
    ax_time.set_title(f"Time Series: {conf['name']} ({conf['unit']}) Across All 5 Sensors", fontsize=13, fontweight='bold', pad=10)
    ax_time.set_ylabel(f"{conf['name']} [{conf['unit']}]", fontsize=11)
    ax_time.grid(True, linestyle=':', alpha=0.5)
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_time.legend(loc='upper right', fontsize=8.5, framealpha=0.9, ncol=2)
    
    # Panel 2: Diurnal Cycle (24-Hour Hour-of-Day Profile) for Dynamic Sensors
    ax_diurnal = fig.add_subplot(gs[1, 0])
    dynamic_sids = ['6012002000869', '6012002000326']
    has_diurnal = False
    
    for sid in dynamic_sids:
        s_data = vdf[vdf['sensor_id'] == sid]
        if len(s_data) > 100 and s_data[col].std() > 0.001:
            has_diurnal = True
            color = sensor_colors.get(str(sid), '#2563EB')
            lbl = sensor_labels.get(str(sid), f"Sensor {sid}")
            hourly = s_data.groupby('hour')[col].agg(['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
            hourly.columns = ['mean', 'std', 'q25', 'q75']
            
            hours = hourly.index
            ax_diurnal.plot(hours, hourly['mean'], 'o-', color=color, linewidth=2.2, label=f"{lbl} Mean")
            ax_diurnal.fill_between(hours, hourly['q25'], hourly['q75'], color=color, alpha=0.18, label=f"{lbl} IQR (25-75%)")
            
    if not has_diurnal:
        # Fallback to all data if none dynamic
        hourly = vdf.groupby('hour')[col].mean()
        ax_diurnal.plot(hourly.index, hourly.values, 'o-', color='#2563EB', linewidth=2.2, label='Dataset Mean')
        
    ax_diurnal.set_title(f"Daily Diurnal Cycle (24-Hour Profile)", fontsize=11, fontweight='bold')
    ax_diurnal.set_xlabel("Hour of Day (00:00 - 23:00 Local Time)", fontsize=10)
    ax_diurnal.set_ylabel(f"{conf['name']} [{conf['unit']}]", fontsize=10)
    ax_diurnal.set_xticks(range(0, 24, 2))
    ax_diurnal.grid(True, linestyle=':', alpha=0.5)
    ax_diurnal.legend(loc='best', fontsize=8.5, framealpha=0.9)
    
    # Panel 3: Distribution & Boxplot / Histogram
    ax_dist = fig.add_subplot(gs[1, 1])
    
    # Create grouped boxplot by sensor
    sensor_groups = []
    sensor_names_list = []
    colors_list = []
    
    for sid in sorted(vdf['sensor_id'].unique()):
        s_vals = vdf[vdf['sensor_id'] == sid][col].dropna()
        if len(s_vals) > 0:
            sensor_groups.append(s_vals)
            short_id = f"...{str(sid)[-3:]}"
            sensor_names_list.append(short_id)
            colors_list.append(sensor_colors.get(str(sid), '#666666'))
            
    bp = ax_dist.boxplot(sensor_groups, patch_artist=True, labels=sensor_names_list, showfliers=False, medianprops=dict(color='black', linewidth=1.8))
    for patch, c in zip(bp['boxes'], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        
    ax_dist.set_title(f"Distribution & Spread Across Sensors", fontsize=11, fontweight='bold')
    ax_dist.set_xlabel("Sensor Node (Last 3 Digits)", fontsize=10)
    ax_dist.set_ylabel(f"{conf['name']} [{conf['unit']}]", fontsize=10)
    ax_dist.grid(True, linestyle=':', alpha=0.5)
    
    # Add stats summary text box
    q05 = vdf[col].quantile(0.05)
    q50 = vdf[col].median()
    q95 = vdf[col].quantile(0.95)
    vmean = vdf[col].mean()
    vstd = vdf[col].std()
    
    stats_text = (f"Overall Stats:\n"
                  f"Mean: {vmean:.2f} {conf['unit']}\n"
                  f"Std: {vstd:.2f}\n"
                  f"Median: {q50:.2f}\n"
                  f"5th-95th%: [{q05:.1f}, {q95:.1f}]")
    ax_dist.text(0.97, 0.95, stats_text, transform=ax_dist.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='#cccccc'), fontsize=8.5)
                 
    plt.suptitle(f"MMA3001 Environmental Sensor Analysis: {conf['name']}", fontsize=15, fontweight='bold', y=0.98)
    
    fig_filename = f"var_{col}.png"
    fig_path = os.path.join(out_dir, fig_filename)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fig_path}")

print(f"All 11 variable plots generated in {time.time()-t0:.1f}s!")
