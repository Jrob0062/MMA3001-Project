import pandas as pd

# 1. Load Environmental CSV Data
env_df = pd.read_csv('sensor_869_occupancy_overlap.csv')
env_df['dt'] = pd.to_datetime(env_df['timestamp']).dt.tz_localize(None)

# 2. Load Occupancy Excel Data
occ_df = pd.read_excel('reduced occupancy data.xlsx')
occ_df['dt'] = pd.to_datetime(occ_df['collecteddate'].str.slice(0, 19))

# Filter for the target room
selected_room = '941f0352-bb69-47d0-a9c5-ac29eb4f7319'
occ_room = occ_df[occ_df['floorspaceid'] == selected_room].sort_values('dt').copy()

# Sort both datasets chronologically
env_sorted = env_df.sort_values('dt').copy()

# 3. Pair environmental timestamps with the most recent headcount change
master = pd.merge_asof(
    env_sorted,
    occ_room[['dt', 'headcount']],
    on='dt',
    direction='backward'  # Holds the previous headcount constant until a new reading occurs
)

# Fill any early missing values before the first camera event with 0
master['headcount'] = master['headcount'].fillna(0).astype(int)

# 4. Feature Engineering (Rate of Change & Time features)
master['co2_rate_of_change'] = master['co2_ppm'].diff().fillna(0)
master['hour'] = master['dt'].dt.hour
master['day_of_week'] = master['dt'].dt.dayofweek
master['is_weekend'] = master['day_of_week'].isin([5, 6]).astype(int)

# 5. Create binary occupied target
master['is_occupied'] = (master['headcount'] > 0).astype(int)

# Save merged dataset directly to Excel / CSV
master.to_excel('master_aligned_dataset.xlsx', index=False)
master.to_csv('master_aligned_dataset.csv', index=False)

print(f"Successfully aligned {len(master):,} rows while keeping exact integer headcounts!")