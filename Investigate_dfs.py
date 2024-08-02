import pandas as pd
from PARAMETERS import *
import numpy as np

path = r"D:\attack_injected_data\REAL_SUB_DFS\v1712_IP21361_10a_t1234\2017-07-06.csv"

df = pd.read_csv(path)
df['datetime'] = pd.to_datetime(df['@timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
df.set_index('datetime', inplace=True)

def filter_by_day(df, start_time, end_time):
    time_slice = pd.date_range(start=start_time, end=end_time, freq='T')
    filtered_index = df.index.intersection(time_slice)
    return df.loc[filtered_index]

start_time = df.index.min()
end_time = start_time + pd.Timedelta(hours=24)
df_ = filter_by_day(df, start_time, end_time)

# LABELLING PROCESS ###############################
src_ip_attack = '10.0.2.186'
target_ips = '10.0.0'
df['Label'] = np.where((df['IPV4_SRC_ADDR'] == src_ip_attack) & (df['IPV4_DST_ADDR'].str.contains(target_ips)), 1, 0)
output = path[:-4] + '_Labelled.csv'
df.to_csv(output, ignore_index=False)



filtered_df = df_[df_['IPV4_SRC_ADDR'] == src_ip_attack]
print(f'The shape of single source ip {src_ip_attack}: {filtered_df.shape}')
print(f'Sttarting from {filtered_df.index.min()} to {filtered_df.index.max()}')
print(f'Number of destination ips:{filtered_df.IPV4_DST_ADDR.nunique()}')


hourly_counts = filtered_df.resample('H').size()
print(f'The maximum number of folws from {src_ip_attack} is: {hourly_counts.max()}, in the timestamp: {hourly_counts.idxmax()}')



print('done')

