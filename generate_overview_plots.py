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

print("Starting overview plots generation...")
t0 = time.time()

csv_path = r'C:\Users\gregr\.gemini\antigravity\brain\fa95884d-3e0c-4586-8e63-03ba1bc71dea\scratch\cleaned_sensor_data.csv'
out_dir = r'C:\Users\gregr\.gemini\antigravity\brain\fa95884d-3e0c-4586-8e63-03ba1bc71dea\plots'

sensor_colors = {
    '6012002000869': '#2563EB',
    '6012002000326': '#10B981',
    '6012002000241': '#F59E0B',
    '6012002000227': '#8B5CF6',
    '6012002000777': '#EC4899',
}

sensor_labels = {
    '6012002000869': 'Sensor ...869 (Active 2024)',
    '6012002000326': 'Sensor ...326 (Active 2025/26)',
    '6012002000241': 'Sensor ...241 (Static Ref)',
    '6012002000227': 'Sensor ...227 (Static Ref)',
    '6012002000777': 'Sensor ...777 (Static Ref)',
}

df = pd.read_csv(csv_path)
df['sensor_id'] = df['sensor_id'].astype(str)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# ----------------------------------------------------
# 1. Particulate Matter Breakdown (PM1 vs PM2.5 vs PM4 vs PM10)
# ----------------------------------------------------
print("Generating Particulate Matter comparison plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=160, gridspec_kw={'height_ratios': [1.5, 1]})

# Use active dynamic sensor 869
df_869 = df[df['sensor_id'] == '6012002000869'].sort_values('timestamp')
df_869_sub = df_869.iloc[::max(1, len(df_869)//2500)]

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

# Ratios / fractions: PM2.5/PM10 ratio indicates whether pollution is fine combustion smoke vs coarse dust
ratio_25_10 = (df_869['pm25_ugm3'] / (df_869['pm10_ugm3'].replace(0, np.nan))).dropna()
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
plt.savefig(os.path.join(out_dir, "overview_particulate_matter.png"), bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# 2. Thermal Comfort (Temperature vs Relative Humidity)
# ----------------------------------------------------
print("Generating Thermal Comfort plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=160)

# Scatter of Temp vs RH for dynamic periods
df_dyn = df[df['sensor_id'].isin(['6012002000869', '6012002000326'])].dropna(subset=['temperature_c', 'humidity_pct'])
# Subsample for smooth scatter
df_dyn_sub = df_dyn.sample(min(len(df_dyn), 8000), random_state=42)

for sid, grp in df_dyn_sub.groupby('sensor_id'):
    c = '#2563EB' if sid == '6012002000869' else '#10B981'
    lbl = 'Sensor ...869 (2024)' if sid == '6012002000869' else 'Sensor ...326 (2025/26)'
    ax1.scatter(grp['temperature_c'], grp['humidity_pct'], alpha=0.25, s=15, color=c, label=lbl)

# ASHRAE 55 Comfort Zone box
from matplotlib.patches import Rectangle
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

# Monthly seasonal temperature and humidity progression
monthly = df[df['sensor_id'] == '6012002000869'].set_index('timestamp').resample('ME')[['temperature_c', 'humidity_pct']].mean().dropna()
ax2_twin = ax2.twinx()

months_str = [m.strftime('%b %Y') for m in monthly.index]
p1 = ax2.plot(months_str, monthly['temperature_c'], 'o-', color='#DC2626', linewidth=2.5, label='Monthly Mean Temperature')
p2 = ax2_twin.plot(months_str, monthly['humidity_pct'], 's--', color='#2563EB', linewidth=2.5, label='Monthly Mean Humidity')

ax2.set_title("Seasonal Trajectory: Mean Temp vs Humidity [Sensor ...869]", fontsize=13, fontweight='bold')
ax2.set_xlabel("Month", fontsize=11)
ax2.set_ylabel("Temperature [°C]", color='#DC2626', fontsize=11)
ax2_twin.set_ylabel("Relative Humidity [% RH]", color='#2563EB', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.5)

# Combined legend
lines = p1 + p2
labs = [l.get_label() for l in lines]
ax2.legend(lines, labs, loc='upper left', fontsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "overview_thermal_comfort.png"), bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# 3. Correlation Matrix Heatmap
# ----------------------------------------------------
print("Generating Correlation Matrix Heatmap...")
fig, ax = plt.subplots(figsize=(11, 9), dpi=160)

cols_corr = ['temperature_c', 'humidity_pct', 'co2_ppm', 'lvoc_ppb', 'formaldehyde_ugm3', 
             'pm1_ugm3', 'pm25_ugm3', 'pm4_ugm3', 'pm10_ugm3', 'pressure_mb', 'battery_v']
corr_labels = ['Temperature', 'Humidity', 'CO₂', 'LVOC', 'Formaldehyde', 
               'PM 1.0', 'PM 2.5', 'PM 4.0', 'PM 10', 'Pressure', 'Battery']

# Compute correlation on dynamic sensor 869
corr_matrix = df[df['sensor_id'] == '6012002000869'][cols_corr].corr()

im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson Correlation Coefficient (r)')

ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9.5)
ax.set_yticklabels(corr_labels, fontsize=9.5)

# Add correlation text in each cell
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.iloc[i, j]
        if not np.isnan(val):
            text_color = 'white' if abs(val) > 0.45 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8.5, fontweight='bold' if abs(val)>0.5 else 'normal')

ax.set_title("Environmental Variable Correlation Matrix [Sensor ...869 Active Window]", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "overview_correlation_matrix.png"), bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# 4. Air Quality & Occupancy Diurnal Waves (CO2, LVOC, PM2.5, Temp)
# ----------------------------------------------------
print("Generating IAQ Occupancy Diurnal Waves...")
fig, axes = plt.subplots(4, 1, figsize=(15, 11), dpi=160, sharex=True)

iaq_vars = [
    ('co2_ppm', 'Carbon Dioxide (CO₂)', 'ppm', '#DC2626', 800),
    ('lvoc_ppb', 'Light VOCs (LVOC)', 'ppb', '#D97706', 65),
    ('pm25_ugm3', 'Particulate Matter PM2.5', 'µg/m³', '#059669', 15),
    ('temperature_c', 'Temperature', '°C', '#2563EB', 24)
]

for ax, (var, title, unit, color, thresh) in zip(axes, iaq_vars):
    hourly_mean = df_869.groupby('hour')[var].mean()
    hourly_q25 = df_869.groupby('hour')[var].quantile(0.25)
    hourly_q75 = df_869.groupby('hour')[var].quantile(0.75)
    
    ax.plot(hourly_mean.index, hourly_mean.values, 'o-', color=color, linewidth=2.2, label=f"Mean {title}")
    ax.fill_between(hourly_mean.index, hourly_q25.values, hourly_q75.values, color=color, alpha=0.18, label="Interquartile Range (25-75%)")
    if thresh:
        ax.axhline(thresh, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label=f"Threshold ({thresh} {unit})")
    ax.set_ylabel(f"{unit}", fontsize=10, fontweight='bold')
    ax.set_title(f"{title}", fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

axes[-1].set_xlabel("Hour of the Day (00:00 - 23:00 Local Time)", fontsize=11)
axes[-1].set_xticks(range(0, 24, 1))

plt.suptitle("Indoor Air Quality (IAQ) & Occupancy Diurnal Signatures [Sensor ...869]", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "overview_iaq_occupancy.png"), bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# 5. Sensor Timeline & Health Dashboard
# ----------------------------------------------------
print("Generating Sensor Timeline & Health Dashboard...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=160, sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})

# Daily packet count per sensor
daily_counts = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'sensor_id']).size().unstack(fill_value=0)
for sid in daily_counts.columns:
    c = sensor_colors.get(str(sid), '#666666')
    lbl = sensor_labels.get(str(sid), f"Sensor {sid}")
    ax1.plot(daily_counts.index, daily_counts[sid], label=lbl, color=c, linewidth=1.5, alpha=0.85)

ax1.set_title("Transmission Reliability: Daily Packet Counts per Sensor (Expected: ~135/day)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Packets / Day", fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right', fontsize=8.5, ncol=3)

# Battery voltage over time
for sid, grp in df.groupby('sensor_id'):
    c = sensor_colors.get(str(sid), '#666666')
    lbl = sensor_labels.get(str(sid), f"Sensor {sid}")
    grp_sub = grp.iloc[::max(1, len(grp)//1500)]
    ax2.plot(grp_sub['timestamp'], grp_sub['battery_v'], '.', markersize=2, alpha=0.4, color=c, label=lbl)

ax2.axhline(4.2, color='green', linestyle=':', label='Full Battery (4.2V)')
ax2.axhline(3.7, color='orange', linestyle=':', label='Nominal (3.7V)')
ax2.set_title("Node Power Health: Battery Voltage (V) Over Time", fontsize=12, fontweight='bold')
ax2.set_ylabel("Battery [V]", fontsize=10)
ax2.set_xlabel("Date", fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='lower left', fontsize=8.5, ncol=3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "overview_sensor_health.png"), bbox_inches='tight')
plt.close()

print(f"All overview plots generated successfully in {time.time()-t0:.1f}s!")
