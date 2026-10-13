import os
import pandas as pd

DATA_DIR = os.path.join('.', '5. Reduced Data')

def clean_and_format_occupancy(input_filename, output_filename, room_id_val):
    """
    Loads raw occupancy data, parses timestamps, adds a sensor/room ID, 
    and saves a clean standalone occupancy file.
    """
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)
    
    print(f"Processing {input_filename} for Room {room_id_val}...")
    df = pd.read_csv(input_path)
    
    # Clean column headers
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    # 1. Standardise Timestamp Column
    dt_candidates = ['timestamp', 'dt', 'collecteddate', 'time', 'date', 'datetime']
    dt_col = next((c for c in df.columns if c.lower() in dt_candidates), None)
    if not dt_col:
        raise KeyError(f"Could not find timestamp in {input_filename}. Columns: {list(df.columns)}")
    
    df['dt'] = pd.to_datetime(df[dt_col].astype(str).str.slice(0, 19), errors='coerce')
    
    # 2. Standardise Headcount / Occupancy Column
    occ_candidates = ['headcount', 'occupancy', 'head_count', 'count', 'people', 'occupants']
    occ_col = next((c for c in df.columns if c.lower() in occ_candidates), None)
    if not occ_col:
        raise KeyError(f"Could not find headcount column in {input_filename}. Columns: {list(df.columns)}")
    
    # 3. Build Clean Standalone DataFrame
    clean_df = pd.DataFrame()
    clean_df['dt'] = df['dt']
    clean_df['sensor_id'] = room_id_val  # Explicitly add sensor/room identifier
    clean_df['headcount'] = pd.to_numeric(df[occ_col], errors='coerce')
    
    # Drop rows with missing values and sort chronologically
    clean_df = clean_df.dropna().sort_values('dt').reset_index(drop=True)
    
    # Save to CSV
    clean_df.to_csv(output_path, index=False)
    print(f"Successfully saved clean file to: {output_path} ({len(clean_df)} rows)\n")

# ==============================================================================
# EXECUTE REBUILD FOR ROOM 869 & ROOM 326
# ==============================================================================

# 1. Rebuild Room 869 Occupancy
# (Change 'sensor_869_occupancy_overlap.csv' to whatever raw source file you use for 869 occupancy)
try:
    clean_and_format_occupancy(
        input_filename='sensor_869_occupancy_overlap.csv', 
        output_filename='869_occupancy.csv', 
        room_id_val='Room_869'
    )
except Exception as e:
    print(f"Room 869 processing note: {e}")

# 2. Rebuild Room 326 Occupancy
# (Using 'occupancy_data_2026_only.csv' or your master occupancy log)
try:
    clean_and_format_occupancy(
        input_filename='occupancy_data_2026_only.csv', 
        output_filename='326_occupancy.csv', 
        room_id_val='Room_326'
    )
except Exception as e:
    print(f"Room 326 processing note: {e}")