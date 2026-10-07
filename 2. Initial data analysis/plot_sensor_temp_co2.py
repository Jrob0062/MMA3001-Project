import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

csv_path = r'c:\Users\gregr\Documents\Year_3\MMA3001\5EnvSensor_Cleaned_Tabular.csv'
plot_dir_user = r'c:\Users\gregr\Documents\Year_3\MMA3001\plots'
plot_dir_artifact = r'C:\Users\gregr\.gemini\antigravity\brain\fa95884d-3e0c-4586-8e63-03ba1bc71dea'

os.makedirs(plot_dir_user, exist_ok=True)
os.makedirs(plot_dir_artifact, exist_ok=True)

print("Reading cleaned dataset...")
df = pd.read_csv(csv_path)
df['sensor_id'] = df['sensor_id'].astype(str)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

sensor_ids = ['227', '241', '326', '777', '869']
sensor_meta = {
    '227': {'name': 'Sensor ID: 227', 'desc': 'Static Reference (21,363 rows | May 2024 – Oct 2024)', 'color': '#8B5CF6'},
    '241': {'name': 'Sensor ID: 241', 'desc': 'Static Dual-State Ref (58,908 rows | Feb 2024 – Apr 2026)', 'color': '#F59E0B'},
    '326': {'name': 'Sensor ID: 326', 'desc': 'Active Node 2025/26 (58,909 rows | Feb 2024 – Apr 2026)', 'color': '#10B981'},
    '777': {'name': 'Sensor ID: 777', 'desc': 'Static Reference (10,926 rows | Feb 2024 – May 2024)', 'color': '#EC4899'},
    '869': {'name': 'Sensor ID: 869', 'desc': 'Active Dynamic Node 2024 (29,341 rows | Feb 2024 – Oct 2024)', 'color': '#2563EB'},
}

# -------------------------------------------------------------
# FIGURE 1: 5-Row Facet Plot (Each Sensor ID with Dual Y-Axes: Temp & CO2)
# -------------------------------------------------------------
print("Generating Figure 1: 5-row dual-axis facet plot...")
fig, axes = plt.subplots(5, 1, figsize=(16, 15), dpi=160, sharex=True)

for i, sid in enumerate(sensor_ids):
    ax1 = axes[i]
    sdf = df[df['sensor_id'] == sid].dropna(subset=['temperature_c', 'co2_ppm'])
    meta = sensor_meta[sid]
    
    # Subsample if large for crisp rendering
    if len(sdf) > 4000:
        sdf_plot = sdf.iloc[::max(1, len(sdf)//3000)]
    else:
        sdf_plot = sdf
        
    ax2 = ax1.twinx()
    
    # Plot Temperature on left axis (Coral / Red)
    p1 = ax1.plot(sdf_plot['timestamp'], sdf_plot['temperature_c'], color='#E11D48', alpha=0.75, linewidth=1.2, label=f"Temp (°C)")
    ax1.set_ylabel("Temp [°C]", color='#E11D48', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#E11D48')
    
    # Plot CO2 on right axis (Teal / Blue)
    p2 = ax2.plot(sdf_plot['timestamp'], sdf_plot['co2_ppm'], color='#0284C7', alpha=0.75, linewidth=1.2, label=f"CO₂ (ppm)")
    ax2.set_ylabel("CO₂ [ppm]", color='#0284C7', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#0284C7')
    
    # Add stats summary text in top left
    t_min, t_max, t_mean = sdf['temperature_c'].min(), sdf['temperature_c'].max(), sdf['temperature_c'].mean()
    c_min, c_max, c_mean = sdf['co2_ppm'].min(), sdf['co2_ppm'].max(), sdf['co2_ppm'].mean()
    
    summary_box = (f"{meta['name']} — {meta['desc']}\n"
                   f"Temp: min={t_min:.1f}°C, mean={t_mean:.1f}°C, max={t_max:.1f}°C  |  "
                   f"CO₂: min={c_min:.0f}ppm, mean={c_mean:.0f}ppm, max={c_max:.0f}ppm")
                   
    ax1.set_title(summary_box, fontsize=10.5, fontweight='bold', loc='left', pad=6,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.9))
                  
    ax1.grid(True, linestyle=':', alpha=0.5)

axes[-1].set_xlabel("Date", fontsize=11, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.suptitle("Temperature & CO₂ Over Time for Each Sensor ID (Dual Y-Axis Time Series)", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

f1_user = os.path.join(plot_dir_user, "each_sensor_temp_and_co2_dual_axis.png")
f1_art = os.path.join(plot_dir_artifact, "each_sensor_temp_and_co2_dual_axis.png")
plt.savefig(f1_user, bbox_inches='tight')
plt.savefig(f1_art, bbox_inches='tight')
plt.close()
print(f"Saved {f1_user}")

# -------------------------------------------------------------
# FIGURE 2: Multi-Sensor Comparison (Temp on top, CO2 on bottom)
# -------------------------------------------------------------
print("Generating Figure 2: Comparative overlay plot...")
fig, (ax_temp, ax_co2) = plt.subplots(2, 1, figsize=(16, 10), dpi=160, sharex=True)

for sid in sensor_ids:
    sdf = df[df['sensor_id'] == sid].dropna(subset=['temperature_c', 'co2_ppm'])
    meta = sensor_meta[sid]
    color = meta['color']
    
    # Subsample for smooth line
    if len(sdf) > 5000:
        sdf_plot = sdf.iloc[::max(1, len(sdf)//3000)]
    else:
        sdf_plot = sdf
        
    ax_temp.plot(sdf_plot['timestamp'], sdf_plot['temperature_c'], '.', markersize=2, alpha=0.4, color=color, label=f"ID {sid} ({meta['name'].split(': ')[1]})")
    ax_co2.plot(sdf_plot['timestamp'], sdf_plot['co2_ppm'], '.', markersize=2, alpha=0.4, color=color, label=f"ID {sid} ({meta['name'].split(': ')[1]})")
    
    # Add rolling mean for dynamic sensors
    if sid in ['869', '326']:
        roll_t = sdf.set_index('timestamp')['temperature_c'].rolling('48h', min_periods=20).mean()
        roll_c = sdf.set_index('timestamp')['co2_ppm'].rolling('48h', min_periods=20).mean()
        ax_temp.plot(roll_t.index, roll_t.values, color=color, linewidth=2.0, alpha=0.95)
        ax_co2.plot(roll_c.index, roll_c.values, color=color, linewidth=2.0, alpha=0.95)

# Guidelines
ax_temp.axhspan(20, 24, color='#16A34A', alpha=0.12, label='ASHRAE Comfort Zone (20–24°C)')
ax_temp.set_title("Comparative Temperature (°C) Over Time by Sensor ID", fontsize=12, fontweight='bold')
ax_temp.set_ylabel("Temperature [°C]", fontsize=11)
ax_temp.grid(True, linestyle=':', alpha=0.5)
ax_temp.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)

ax_co2.axhline(420, color='gray', linestyle='--', linewidth=1.2, label='Outdoor Ambient (~420 ppm)')
ax_co2.axhline(800, color='orange', linestyle='--', linewidth=1.2, label='Indoor Recommended (<800 ppm)')
ax_co2.axhline(1000, color='red', linestyle='--', linewidth=1.2, label='Ventilation Guideline Limit (1000 ppm)')
ax_co2.set_title("Comparative Carbon Dioxide (CO₂) (ppm) Over Time by Sensor ID", fontsize=12, fontweight='bold')
ax_co2.set_ylabel("CO₂ [ppm]", fontsize=11)
ax_co2.set_xlabel("Date", fontsize=11)
ax_co2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_co2.grid(True, linestyle=':', alpha=0.5)
ax_co2.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)

plt.suptitle("Comparison of All 5 Sensor IDs: Temperature and CO₂ Over Time", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

f2_user = os.path.join(plot_dir_user, "all_ids_temp_and_co2_comparison.png")
f2_art = os.path.join(plot_dir_artifact, "all_ids_temp_and_co2_comparison.png")
plt.savefig(f2_user, bbox_inches='tight')
plt.savefig(f2_art, bbox_inches='tight')
plt.close()
print(f"Saved {f2_user}")

# -------------------------------------------------------------
# FIGURE 3: Zoomed-in 2024 Campaign (Where all 5 nodes were active/present)
# -------------------------------------------------------------
print("Generating Figure 3: Zoomed 2024 Campaign...")
df_2024 = df[(df['timestamp'] >= '2024-03-01') & (df['timestamp'] <= '2024-10-15')].copy()

fig, (ax_z_t, ax_z_c) = plt.subplots(2, 1, figsize=(16, 9), dpi=160, sharex=True)

for sid in sensor_ids:
    sdf = df_2024[df_2024['sensor_id'] == sid].dropna(subset=['temperature_c', 'co2_ppm'])
    meta = sensor_meta[sid]
    color = meta['color']
    
    if len(sdf) > 3000:
        sdf_plot = sdf.iloc[::max(1, len(sdf)//2000)]
    else:
        sdf_plot = sdf
        
    ax_z_t.plot(sdf_plot['timestamp'], sdf_plot['temperature_c'], '.', markersize=2.5, alpha=0.45, color=color, label=f"ID {sid}")
    ax_z_c.plot(sdf_plot['timestamp'], sdf_plot['co2_ppm'], '.', markersize=2.5, alpha=0.45, color=color, label=f"ID {sid}")
    
    if sid == '869':
        roll_t = sdf.set_index('timestamp')['temperature_c'].rolling('24h', min_periods=10).mean()
        roll_c = sdf.set_index('timestamp')['co2_ppm'].rolling('24h', min_periods=10).mean()
        ax_z_t.plot(roll_t.index, roll_t.values, color=color, linewidth=2.2, label=f"ID 869 (24h Trend)")
        ax_z_c.plot(roll_c.index, roll_c.values, color=color, linewidth=2.2, label=f"ID 869 (24h Trend)")

ax_z_t.axhspan(20, 24, color='#16A34A', alpha=0.12, label='ASHRAE Comfort Zone (20–24°C)')
ax_z_t.set_title("2024 Main Campaign: Temperature by Sensor ID", fontsize=12, fontweight='bold')
ax_z_t.set_ylabel("Temperature [°C]", fontsize=11)
ax_z_t.grid(True, linestyle=':', alpha=0.5)
ax_z_t.legend(loc='upper right', fontsize=9, ncol=3)

ax_z_c.axhline(420, color='gray', linestyle='--', linewidth=1.2, label='Outdoor Baseline (420 ppm)')
ax_z_c.axhline(800, color='orange', linestyle='--', linewidth=1.2, label='Indoor Limit (<800 ppm)')
ax_z_c.set_title("2024 Main Campaign: Carbon Dioxide (CO₂) by Sensor ID", fontsize=12, fontweight='bold')
ax_z_c.set_ylabel("CO₂ [ppm]", fontsize=11)
ax_z_c.set_xlabel("Date (March – October 2024)", fontsize=11)
ax_z_c.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax_z_c.grid(True, linestyle=':', alpha=0.5)
ax_z_c.legend(loc='upper right', fontsize=9, ncol=3)

plt.suptitle("Detailed 2024 Campaign View: Temperature and CO₂ Across All 5 Sensor IDs", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

f3_user = os.path.join(plot_dir_user, "zoomed_2024_ids_temp_co2.png")
f3_art = os.path.join(plot_dir_artifact, "zoomed_2024_ids_temp_co2.png")
plt.savefig(f3_user, bbox_inches='tight')
plt.savefig(f3_art, bbox_inches='tight')
plt.close()
print(f"Saved {f3_user}")

print("All plots generated successfully!")
