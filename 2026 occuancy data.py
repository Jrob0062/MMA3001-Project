import pandas as pd

# 1. Load Sensor 326 Environmental Data (2025–2026)
env_326 = pd.read_csv('5occupancySensor_MayToDec2024_9MRows.csv')

# Clean leading/trailing spaces and BOM characters from column names
env_326.columns = env_326.columns.str.strip().str.replace('\ufeff', '')

# Print available columns to confirm the exact header name
print("Cleaned column names:", env_326.columns.tolist())

# 2. Identify the date column and convert
raw_date_column = 'dt' if 'dt' in env_326.columns else 'collecteddate'

# Overwrite or convert to datetime
env_326['dt'] = pd.to_datetime(env_326[raw_date_column]).dt.tz_localize(None)

# 3. Filter for late 2025 onwards
env_326 = env_326[env_326['dt'] >= '2025-09-01'].sort_values('dt')

# 2. Load Occupancy Data (Filtered for >= Sept 2025)
# If using chunking on the large CSV:
chunk_size = 500000
occ_chunks = []

for chunk in pd.read_csv('full_occupancy_data.csv', chunksize=chunk_size):
    chunk['dt'] = pd.to_datetime(chunk['collecteddate'].str.slice(0, 19))
    sub = chunk[(chunk['dt'] >= '2025-09-01') & (chunk['floorspaceid'] == '941f0352-bb69-47d0-a9c5-ac29eb4f7319')]
    if not sub.empty:
        occ_chunks.append(sub)

occ_326_timeline = pd.concat(occ_chunks, ignore_index=True).sort_values('dt')

# 3. Pair environmental readings with constant headcount state (merge_asof)
master_326 = pd.merge_asof(
    env_326,
    occ_326_timeline[['dt', 'headcount']],
    on='dt',
    direction='backward'  # Holds previous headcount constant until next change
)

master_326['headcount'] = master_326['headcount'].fillna(0).astype(int)
master_326['is_occupied'] = (master_326['headcount'] > 0).astype(int)
print ("done")