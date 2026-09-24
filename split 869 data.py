import pandas as pd

def extract_sensor_869_datasets(input_filepath='5EnvSensor_Cleaned_Tabular.csv'):
    print("Loading environmental dataset...")
    # Read original tabular CSV dataset
    df = pd.read_csv(input_filepath)
    
    # Convert timestamp column to datetime
    df['dt'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    # 1. Full dataset for Sensor 869
    s869_full = df[df['sensor_id'] == 869].copy()
    
    # 2. Valid active window (excluding the flatlined 520.0 ppm CO2 period)
    s869_valid = df[(df['sensor_id'] == 869) & (df['co2_ppm'] != 520.0)].copy()
    
    # 3. Ground-truth occupancy overlap window (March 1, 2024 to May 6, 2024)
    s869_overlap = df[
        (df['sensor_id'] == 869) & 
        (df['dt'] >= '2024-03-01') & 
        (df['dt'] <= '2024-05-06')
    ].copy()
    
    # Save datasets to CSV without the temporary helper column
    s869_overlap.drop(columns=['dt']).to_csv('sensor_869_occupancy_overlap.csv', index=False)
    s869_valid.drop(columns=['dt']).to_csv('sensor_869_valid_window.csv', index=False)
    s869_full.drop(columns=['dt']).to_csv('sensor_869_full.csv', index=False)
    
    print("\nFiles successfully created in your directory:")
    print(f" - sensor_869_occupancy_overlap.csv ({len(s869_overlap):,} rows)")
    print(f" - sensor_869_valid_window.csv      ({len(s869_valid):,} rows)")
    print(f" - sensor_869_full.csv              ({len(s869_full):,} rows)")

if __name__ == '__main__':
    extract_sensor_869_datasets()