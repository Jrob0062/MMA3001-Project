import pandas as pd
import numpy as np

# Load raw datasets
env_df = pd.read_csv('sensor_869_occupancy_overlap.csv')
occ_df = pd.read_excel('reduced occupancy data.xlsx')

# Convert timestamps
env_df['dt'] = pd.to_datetime(env_df['timestamp']).dt.tz_localize(None)
occ_df['dt'] = pd.to_datetime(occ_df['collecteddate'].str.slice(0, 19))

# Select room 941f0352 (full March-May coverage) or 3e5a14df (May 1-6 overlap)
# We will use room 941f0352 for full 2-month training
selected_room = '941f0352-bb69-47d0-a9c5-ac29eb4f7319'
occ_room = occ_df[occ_df['floorspaceid'] == selected_room].copy()

# Set index for time resampling
env_indexed = env_df.set_index('dt').sort_index()
occ_indexed = occ_room.set_index('dt').sort_index()

# Resample to 15-minute grid
env_15m = env_indexed[['co2_ppm', 'temperature_c', 'humidity_pct', 'lvoc_ppb', 'pm25_ugm3']].resample('15min').mean().interpolate(method='time')
occ_15m = occ_indexed[['headcount']].resample('15min').mean().interpolate(method='time')

# Merge environmental and occupancy data
master = env_15m.join(occ_15m, how='inner').dropna().reset_index()

# Save master dataset
master.to_csv('master_aligned_dataset.csv', index=False)
print(f"Master dataset saved! ({len(master):,} rows from {master['dt'].min()} to {master['dt'].max()})")