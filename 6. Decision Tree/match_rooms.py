import pandas as pd
import numpy as np

# Load environmental data
df_env = pd.read_csv('5. Reduced Data/5EnvSensor_Cleaned_Tabular.csv')
df_env['dt'] = pd.to_datetime(df_env['timestamp'])

for sid in [869, 326]:
    print('='*75)
    print(f'ANALYSIS FOR ENVIRONMENTAL SENSOR {sid}')
    print('='*75)
    if sid == 869:
        s_env = df_env[(df_env['sensor_id'] == 869) & (df_env['dt'] >= '2024-03-15') & (df_env['dt'] <= '2024-06-14 23:59:59') & (df_env['co2_ppm'] != 520.0)].sort_values('dt').copy()
        occ_file = '5. Reduced Data/869_occupancy.csv'
    else:
        s_env = df_env[(df_env['sensor_id'] == 326) & (df_env['dt'] >= '2025-10-01') & (df_env['dt'] <= '2026-04-22 23:59:59') & (df_env['co2_ppm'] != 525.0)].sort_values('dt').copy()
        occ_file = '5. Reduced Data/326_occupancy.csv'

    s_env['co2_diff'] = s_env['co2_ppm'].diff().fillna(0)
    occ_df = pd.read_csv(occ_file)
    occ_df['dt'] = pd.to_datetime(occ_df['occupancystatuschangedate'].astype(str).str.slice(0, 19))

    for fid, sub_occ in occ_df.groupby('floorspaceid'):
        sub_occ = sub_occ.sort_values('dt')[['dt', 'headcount']]
        merged = pd.merge_asof(s_env, sub_occ, on='dt', direction='backward')
        merged['headcount'] = merged['headcount'].fillna(0)
        
        c_co2 = merged['co2_ppm'].corr(merged['headcount'])
        c_diff = merged['co2_diff'].corr(merged['headcount'])
        c_temp = merged['temperature_c'].corr(merged['headcount'])
        
        # When occupied vs unoccupied mean CO2
        occ_co2 = merged.loc[merged['headcount'] > 0, 'co2_ppm'].mean()
        unocc_co2 = merged.loc[merged['headcount'] == 0, 'co2_ppm'].mean()
        co2_delta = occ_co2 - unocc_co2
        
        print(f"Floorspace {fid}:")
        print(f"  Corr(CO2, Headcount)     = {c_co2:+.3f}")
        print(f"  Corr(dCO2/dt, Headcount) = {c_diff:+.3f}")
        print(f"  Corr(Temp, Headcount)    = {c_temp:+.3f}")
        print(f"  Mean CO2 Occupied: {occ_co2:.1f} ppm | Unoccupied: {unocc_co2:.1f} ppm (Delta: {co2_delta:+.1f} ppm)")
        print(f"  Mean HC: {merged['headcount'].mean():.2f} | Max HC: {merged['headcount'].max():.0f}")
        print()
