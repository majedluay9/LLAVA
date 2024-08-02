import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import gc
import os
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import itertools
import PARAMETERS
import utils
from Functions import single_attack_filtering, convert_to_datetime


def apply_kpca(data, features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    kpca = KernelPCA(n_components=2, kernel='rbf')
    transformed_data = kpca.fit_transform(scaled_data)
    return transformed_data

def kpca_frame_generation(dataset, features, window_size_sec=60, step_size_sec=10):
    df = convert_to_datetime(dataset, 'FIRST_SWITCHED', 'timestamp')
    df.sort_values('timestamp', inplace=True)
    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f'Duration of this dataset is: {duration}')

    # Take a subset only
    # interval = 3
    # filter = df['timestamp'].min() + pd.Timedelta(hours=interval)
    # sub_df = df.copy()
    # df = sub_df[sub_df['timestamp'] <= filter]
    # print(f'Only the first {interval} hours is taken ...')
    x_full, y_full = [],[]
    n_timesteps = 0

    if 'ATTACK' in dataset.columns:
        att_list = dataset['ATTACK'].unique().tolist()
        print(f'The classes included are: {att_list}')
        att_list.remove('Benign')
        for att in att_list:
            df_att = single_attack_filtering(df, att, True)
            df_att.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_att.dropna(inplace=True)

            start_time = df_att['timestamp'].min()
            end_time = df_att['timestamp'].max()
            window_size = pd.Timedelta(seconds=window_size_sec)
            step_size = pd.Timedelta(seconds=step_size_sec)

            current_time = start_time
            while current_time + window_size <= end_time:
                window_data = df_att[
                    (df_att['timestamp'] >= current_time) & (df_att['timestamp'] < current_time + window_size)]
                if not window_data.empty:
                    window_data = window_data[features] # Chosen features only
                    transformed_data = apply_kpca(window_data, features)
                    x_full.append(transformed_data[:, 0])
                    y_full.append(transformed_data[:, 1])
                    n_timesteps += 1
                current_time += step_size
                gc.collect()
    else:
        print('This dataset doesnt have labels')
        df_att = dataset.copy()
        df_att.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_att.dropna(inplace=True)

        start_time = df_att['timestamp'].min()
        end_time = df_att['timestamp'].max()
        window_size = pd.Timedelta(seconds=window_size_sec)
        step_size = pd.Timedelta(seconds=step_size_sec)

        current_time = start_time
        while current_time + window_size <= end_time:
            window_data = df_att[
                (df_att['timestamp'] >= current_time) & (df_att['timestamp'] < current_time + window_size)]
            if not window_data.empty:
                window_data = window_data[features]  # Chosen features only
                transformed_data = apply_kpca(window_data, features)
                x_full.append(transformed_data[:, 0])
                y_full.append(transformed_data[:, 1])
                n_timesteps += 1
            next_time = current_time + step_size
            if df_att[(df_att['timestamp'] >= next_time) & (df_att['timestamp'] < next_time + window_size)].empty:
                next_time = df_att[df_att['timestamp'] > current_time + window_size].timestamp.min()

            current_time = next_time if next_time else end_time  # Avoid infinite loop
            gc.collect()


    return x_full, y_full, n_timesteps

def log_transform_feature(data, small_constant=1):
    data = np.array(data)

    # Apply the logarithmic transformation with a small constant
    transformed_data = np.log(data + small_constant)

    return transformed_data

# Example usage
if __name__ == '__main__':
    df = pd.read_csv(PARAMETERS.SUB_REAL1)

    # df = single_attack_filtering(df,'DoS',True)

    window_size = 120
    step_size = 60
    consistient_scale = True
    Heading = f'SUB_REAL1_{window_size}_{step_size}'

    if not consistient_scale:
        Heading += '_NoScale'
    features_to_be_scalled = ['IN_PKTS', 'OUT_PKTS', 'IN_BYTES', 'OUT_BYTES']  # Define your full feature set
    Used_Features = features_to_be_scalled #+ ['SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV','DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV']

    for feature_ in features_to_be_scalled:
        df[feature_] = log_transform_feature(df[feature_])
        print(f'df[{feature_}].max after log scalling = {df[feature_].max()} ')


    x_full, y_full, n_timesteps = kpca_frame_generation(df, Used_Features, window_size, step_size)  # Adjust the window and step sizes as needed
    # print(f'X_Full shape = {x_full.shape} \n Y_Full shape = {y_full.shape}')
    print(f'The number of timestamps (Frames) = {n_timesteps}')

    # Squish the data, so we can work out the bins, the next 2 lines is converting into 1D array for binning
    bin_count = 20                                                                                                          # Do I need to define better bins !
    x_full_squished = np.concatenate([arr.flatten() for arr in x_full])
    y_full_squished = np.concatenate([arr.flatten() for arr in y_full])
    print(f'X_Full_squished shape = {x_full_squished.shape} \nY_Full_squished shape = {y_full_squished.shape}')

    x_bins = np.histogram_bin_edges(x_full_squished, bins=bin_count)
    y_bins = np.histogram_bin_edges(y_full_squished, bins=bin_count)
    print(f'X_bins shape = {x_bins.shape} \n Y_bin shape = {y_bins.shape}')


    # Function to create a histogram using pre-determined bin edges
    def create_histogram(x, y, x_bins, y_bins):
        hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        return hist


    all_timestep_histograms = []
    for timestep_i in range(n_timesteps):
        x_timestep = x_full[timestep_i]
        y_timestep = y_full[timestep_i]

        # Create histogram of the full data using fixed bins
        hist_timestep = create_histogram(x_timestep, y_timestep, x_bins, y_bins)
        all_timestep_histograms.append(hist_timestep)

        # Optional: Visualize the original histogram
        plt.figure(figsize=(8, 6))
        plt.imshow(hist_timestep, interpolation='nearest', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
        plt.colorbar()
        plt.title(f'{Heading} - 2D Histogram with {bin_count} Bins, in timestamp {timestep_i+1}')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.show()
        save_dir = rf"D:\Frames\Histogram_To_OnePoint\{Heading}\Histograms"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'frame_{timestep_i+1}.png'))
        plt.close()
    print('Done generating the histogram frames. Now processing K-PCA again...')
    all_timestep_histograms = np.array(all_timestep_histograms)

    # Flatten the histogram to use in KPCA
    hist_flattened = all_timestep_histograms.flatten().reshape(n_timesteps, -1)

    kpca = KernelPCA(n_components=2, kernel='rbf')
    transformed_timesteps = kpca.fit_transform(hist_flattened)

    # Consistient frame sizes
    x_min, x_max , y_min, y_max =utils.const_Xlim_Ylim((transformed_timesteps))

    for i in range(1, n_timesteps):
        plt.figure(figsize=(10, 8))
        if i > 1:
            plt.scatter(transformed_timesteps[:i - 1, 0], transformed_timesteps[:i - 1, 1], c='gray',
                        label='Previous Points')
        plt.scatter(transformed_timesteps[i - 1:i, 0], transformed_timesteps[i - 1:i, 1], c='red',
                    label='Current Point')

        if consistient_scale:
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
        plt.xlabel('KPCA 1st Component')
        plt.ylabel('KPCA 2nd Component')
        plt.title(f'Timestamp {i}')
        plt.grid()
        # Save the figure
        save_dir = rf"D:\Frames\Histogram_To_OnePoint\{Heading}\OnePoint"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'frame_{i}.png'))
        plt.close()

