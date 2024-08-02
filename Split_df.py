import PARAMETERS
import pandas as pd
import Functions
import os
import numpy as np
import gc


if __name__ == '__main__':
    FNAME = PARAMETERS.BoT_4June
    Heading = 'Benign_UNSW_Niloo'

    df_main = pd.read_csv(FNAME)
    dfs = []
    W_SIZE = 3600 # frame length in seconds
    S_SIZE = 600 # steps in seconds
    folder_name = f'{W_SIZE}_Seconds_{S_SIZE}sec_{Heading}_NoLimit'
    df = Functions.convert_to_datetime(df_main, 'FIRST_SWITCHED', 'timestamp').copy()
    df.sort_values('timestamp', inplace=True)
    Duration = df['timestamp'].max() - df['timestamp'].min()
    print(f'Duration of {Heading} dataset is: {Duration}')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    save_dir = rf"D:\Frames\LIAM\{Heading}"
    os.makedirs(save_dir, exist_ok=True)

    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    window_size = pd.Timedelta(seconds=W_SIZE)
    step_size = pd.Timedelta(seconds=S_SIZE)

    current_time = start_time
    while current_time + window_size <= end_time:
        window_data = df[(df['timestamp'] >= current_time) & (df['timestamp'] < current_time + window_size)]
        # Benign, Attack = window_data.ATTACK.value_counts()
        if not window_data.empty:
            dfs.append(window_data)
        next_time = current_time + step_size
        if df[(df['timestamp'] >= next_time) & (df['timestamp'] < next_time + window_size)].empty:
            next_time = df[df['timestamp'] > current_time + window_size].timestamp.min()

        current_time = next_time if next_time else end_time  # Avoid infinite loop
        gc.collect()

    dfs_dir = f"D:\python datasets\Liam\\{Heading}_{W_SIZE}_{S_SIZE}"
    os.makedirs(dfs_dir, exist_ok=True)
    for i, df in enumerate(dfs):
        print(f'Shape of Window{i}: {df.shape} , Classes: {df.ATTACK.unique()}')
        df.to_csv(os.path.join(dfs_dir, f'df_{i}.csv'), index=False)
