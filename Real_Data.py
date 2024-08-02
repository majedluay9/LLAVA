import pandas as pd
import os
from PARAMETERS import *

path = r"D:\attack_injected_data\v1712_IP21361_10a_t1234.csv"
Title = 'v1712_IP21361_10a_t1234'
df = pd.read_csv(path)

print(df.shape)
print(df.columns)
start_time = df['@timestamp'].min()
end_time = df['@timestamp'].max()
df['datetime'] = pd.to_datetime(df['FIRST_SWITCHED'], unit='s')
ips = df.IPV4_SRC_ADDR.unique().tolist()
print(ips.__contains__('10.0.2.242'))

duration = df['datetime'].max() - df['datetime'].min()
print(f'df period is: {duration}')
print(start_time)
print(end_time)
#
#
# # Filter Start
# my_timestamp = pd.Timestamp('2017-07-07 16:00:0')                   # For a specific day
# filter_delta = df['datetime'].min() + pd.Timedelta(hours=24)
# result_df = df[df['datetime'] <= filter_delta]
#
# # Filter End
# end_time = my_timestamp + pd.Timedelta(hours=1)
# result_df_ = result_df[result_df['datetime'] <= end_time]
#
# duration = result_df_['datetime'].max() - result_df_['datetime'].min()
# print(f'New df period is: {duration}')
# print(result_df_['datetime'].min())
# print(result_df_['datetime'].max() )
#
#
# output =  path[:-4] + '_DoS_Day8' + path[-4:]
# result_df_.to_csv(output)
# print('done')



grouped = df.groupby(pd.Grouper(key='datetime', freq='D'))
output_directory = rf"D:\attack_injected_data\REAL_SUB_DFS\{Title}"
os.makedirs(output_directory, exist_ok=True)

for name, group in grouped:
    if not group.empty:
        filename = name.strftime('%Y-%m-%d') + '.csv'
        group.to_csv(os.path.join(output_directory, filename), index=False)

print("All daily data files have been saved successfully.")