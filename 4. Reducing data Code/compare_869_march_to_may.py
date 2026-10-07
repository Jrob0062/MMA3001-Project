"""
MMA3001 - Environmental Sensor Analysis
Compare Sensor 869: Temperature, CO2, and LVOC (March 1 to May 5, 2024)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 1. Load Cleaned Dataset
csv_file = "5EnvSensor_Cleaned_Tabular.csv"
print(f"Loading {csv_file}...")
df = pd.read_csv(csv_file)

# Ensure proper data types
df['sensor_id'] = df['sensor_id'].astype(str)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Filter for Sensor 869 and Date Range: March 1 to May 5, 2024
start_date = "2024-03-01"
end_date = "2024-05-05 23:59:59"

mask = (df['sensor_id'] == '869') & (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
sub_df = df[mask].sort_values('timestamp').reset_index(drop=True)

print(f"Total matching records found for Sensor 869: {len(sub_df)}")

# 3. Statistical Summary & Correlation
cols = ['temperature_c', 'co2_ppm', 'lvoc_ppb']
summary = sub_df[cols].describe().round(2)
print("\n--- Summary Statistics (March 1 - May 5, 2024) ---")
print(summary)

corr_matrix = sub_df[cols].corr().round(3)
print("\n--- Pearson Correlation Matrix ---")
print(corr_matrix)

# 4. Create Multi-Panel Comparative Visualizations
fig = plt.figure(figsize=(16, 12), dpi=160)
gs = GridSpec(3, 2, width_ratios=[1.7, 1.0], hspace=0.32, wspace=0.25)

# --- LEFT COLUMN: 3 Synchronized Time Series Stacks ---
# Subplot 1: Temperature
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

# Subplot 2: CO2
ax_co2 = fig.add_subplot(gs[1, 0], sharex=ax_temp)
ax_co2.plot(sub_df['timestamp'], sub_df['co2_ppm'], '.', markersize=2.5, color='#0284C7', alpha=0.35, label='CO₂ Raw')
roll_c = sub_df.set_index('timestamp')['co2_ppm'].rolling('24h', min_periods=10).mean()
ax_co2.plot(roll_c.index, roll_c.values, color='#0369A1', linewidth=2.0, label='24h Rolling Mean')
ax_co2.axhline(420, color='gray', linestyle='--', linewidth=1.2, label='Outdoor Baseline (420 ppm)')
ax_co2.axhline(800, color='orange', linestyle='--', linewidth=1.2, label='Recommended Indoor (<800 ppm)')
ax_co2.set_title("Sensor 869: Carbon Dioxide (CO₂) (ppm) Over Time", fontsize=11, fontweight='bold', loc='left')
ax_co2.set_ylabel("CO₂ [ppm]", fontsize=10, fontweight='bold')
ax_co2.grid(True, linestyle=':', alpha=0.5)
ax_co2.legend(loc='upper right', fontsize=8.5, framealpha=0.9, ncol=2)
ax_co2.set_xticklabels([])

# Subplot 3: LVOC
ax_lvoc = fig.add_subplot(gs[2, 0], sharex=ax_temp)
ax_lvoc.plot(sub_df['timestamp'], sub_df['lvoc_ppb'], color='#D97706', linewidth=1.2, alpha=0.85, label='LVOC (ppb)')
ax_lvoc.axhline(65, color='green', linestyle=':', linewidth=1.2, label='Good Quality Threshold (<65 ppb)')
ax_lvoc.set_title("Sensor 869: Light Volatile Organic Compounds (LVOC) (ppb) Over Time", fontsize=11, fontweight='bold', loc='left')
ax_lvoc.set_ylabel("LVOC [ppb]", fontsize=10, fontweight='bold')
ax_lvoc.set_xlabel("Date (March 1 – May 5, 2024)", fontsize=11, fontweight='bold')
ax_lvoc.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax_lvoc.grid(True, linestyle=':', alpha=0.5)
ax_lvoc.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

# --- RIGHT COLUMN: Diurnal Profile & Cross Correlations ---
# Subplot 4: 24-Hour Diurnal Cycle for Temp & CO2
ax_diurnal = fig.add_subplot(gs[0, 1])
sub_df['hour'] = sub_df['timestamp'].dt.hour
hourly = sub_df.groupby('hour')[['temperature_c', 'co2_ppm', 'lvoc_ppb']].mean()

ax_diurnal_twin = ax_diurnal.twinx()
p1 = ax_diurnal.plot(hourly.index, hourly['temperature_c'], 'o-', color='#DC2626', linewidth=2.2, label='Mean Temp (°C)')
p2 = ax_diurnal_twin.plot(hourly.index, hourly['co2_ppm'], 's--', color='#0284C7', linewidth=2.2, label='Mean CO₂ (ppm)')

ax_diurnal.set_title("24-Hour Diurnal Rhythm (Hourly Averages)", fontsize=11, fontweight='bold')
ax_diurnal.set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=9.5)
ax_diurnal.set_ylabel("Temp [°C]", color='#DC2626', fontsize=10, fontweight='bold')
ax_diurnal_twin.set_ylabel("CO₂ [ppm]", color='#0284C7', fontsize=10, fontweight='bold')
ax_diurnal.tick_params(axis='y', labelcolor='#DC2626')
ax_diurnal_twin.tick_params(axis='y', labelcolor='#0284C7')
ax_diurnal.set_xticks(range(0, 24, 4))
ax_diurnal.grid(True, linestyle=':', alpha=0.5)

lines = p1 + p2
labels = [l.get_label() for l in lines]
ax_diurnal.legend(lines, labels, loc='upper left', fontsize=8.5)

# Subplot 5: Scatter of Temperature vs CO2
ax_scatter1 = fig.add_subplot(gs[1, 1])
sc = ax_scatter1.scatter(sub_df['temperature_c'], sub_df['co2_ppm'], c=sub_df['hour'], cmap='viridis', alpha=0.4, s=15)
cbar = plt.colorbar(sc, ax=ax_scatter1, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Hour of Day', fontsize=8.5)
ax_scatter1.set_title(f"Temperature vs. CO₂ Scatter (r = {corr_matrix.loc['temperature_c', 'co2_ppm']:.2f})", fontsize=11, fontweight='bold')
ax_scatter1.set_xlabel("Temperature [°C]", fontsize=9.5)
ax_scatter1.set_ylabel("CO₂ [ppm]", fontsize=9.5)
ax_scatter1.grid(True, linestyle=':', alpha=0.5)

# Subplot 6: Scatter of CO2 vs LVOC
ax_scatter2 = fig.add_subplot(gs[2, 1])
ax_scatter2.scatter(sub_df['co2_ppm'], sub_df['lvoc_ppb'], color='#7C3AED', alpha=0.35, s=15)
ax_scatter2.set_title(f"CO₂ vs. LVOC Scatter (r = {corr_matrix.loc['co2_ppm', 'lvoc_ppb']:.2f})", fontsize=11, fontweight='bold')
ax_scatter2.set_xlabel("CO₂ [ppm]", fontsize=9.5)
ax_scatter2.set_ylabel("LVOC [ppb]", fontsize=9.5)
ax_scatter2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Sensor 869 Analysis: Temperature, CO₂, and LVOC (March 1 – May 5, 2024)", fontsize=14, fontweight='bold', y=0.99)

# 5. Save High-Res PNG Output
os.makedirs("plots", exist_ok=True)
out_fig_user = os.path.join("plots", "sensor869_march_to_may_comparison.png")
plt.savefig(out_fig_user, bbox_inches='tight')
plt.close()
print(f"\nPlot successfully saved to: {out_fig_user}")
