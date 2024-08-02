import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import numpy as np
import PARAMETERS
from Functions import *

H_IP_SRC        = 'IPV4_SRC_ADDR'
H_IP_DST        = 'IPV4_DST_ADDR'
H_PORT_SRC      = 'L4_SRC_PORT'
H_PORT_DST      = 'L4_DST_PORT'
H_LAST_SWITCHED = 'LAST_SWITCHED'
H_IN_PKTS       = 'IN_PKTS'
H_OUT_PKTS      = 'OUT_PKTS'
H_IN_BYTES      = 'IN_BYTES'
H_OUT_BYTES     = 'OUT_BYTES'

Features = [H_IN_PKTS,H_OUT_PKTS,H_IN_BYTES,H_OUT_BYTES]

def create_color_palette(attacks):
    # Create a consistent color palette
    palette = sns.color_palette("hsv", len(attacks))
    color_map = {attack: color for attack, color in zip(attacks, palette)}
    return color_map

def apply_pca_and_plot(data, current_time, save_dir, features, color_map, B, A,x_lim=None, y_lim=None):

    if not data.empty:
        kpca = KernelPCA(n_components=2, kernel='rbf')  # Using RBF kernel
        pca_results = kpca.fit_transform(data[features])


        # Color mapping
        unique_attacks = data['ATTACK'].unique()
        palette = [color_map[attack] for attack in unique_attacks if attack in color_map]
        color_map = {attack: color_map[attack] for attack in unique_attacks if attack in color_map}
        colors = data['ATTACK'].map(color_map)


        # Visualization
        plt.figure(figsize=(8, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors, alpha=0.5)
        plt.xlabel('Kernel PCA Dimension 1')
        plt.ylabel('Kernel PCA Dimension 2')
        plt.title(f'{current_time} - Benign:{B}, Attack:{A}')

        # Adding legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5) for color in palette]
        plt.legend(handles, color_map.keys(), title="Attack Types")

        plt.savefig(os.path.join(save_dir, f'frame_{current_time}.png'))
        plt.close()

def determine_global_pca_limits(df, features):

    scaler = StandardScaler()
    data_Scale = scaler.fit_transform(df[features])
    kpca = KernelPCA(n_components=2, kernel='rbf')
    pca_results = kpca.fit_transform(df[features])
    x_min, x_max = pca_results[:, 0].min(), pca_results[:, 0].max()
    y_min, y_max = pca_results[:, 1].min(), pca_results[:, 1].max()
    return x_min, x_max, y_min, y_max

# Main function to generate frames
def pca_frame_generation(dataset, title, window_size_sec=60, step_size_sec=10):
    features = Features
    df = convert_to_datetime(dataset, 'FIRST_SWITCHED', 'timestamp').copy()
    df.sort_values('timestamp', inplace=True)
    att_list = dataset['ATTACK'].unique().tolist()
    color_map = create_color_palette(att_list)

    for att in att_list:
        if att != 'Benign':
            df_att = single_attack_filtering(df, att, True).copy()
            Duration = df_att['timestamp'].max() - df_att['timestamp'].min()
            print(f'Now Working on {att} and its Duration: {Duration}')
            df_att.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_att.dropna(inplace=True)


            save_dir = rf"D:\Frames\PCA\{title}\{att}"
            os.makedirs(save_dir, exist_ok=True)

            start_time = df_att['timestamp'].min()
            end_time = df_att['timestamp'].max()
            window_size = pd.Timedelta(seconds=window_size_sec)
            step_size = pd.Timedelta(seconds=step_size_sec)

            current_time = start_time
            while current_time + window_size <= end_time:
                window_data = df_att[(df_att['timestamp'] >= current_time) & (df_att['timestamp'] < current_time + window_size)]
                # DETAILS ABOUT CLASSES
                counts = window_data.ATTACK.value_counts()
                if 'Benign' in counts.index:
                    Benign = counts['Benign']
                if att in counts.index:
                    Attack = counts[att]
                # _____________________________________________________
                if not window_data.empty:
                    apply_pca_and_plot(window_data, current_time.strftime("%Y-%m-%d %H-%M-%S"), save_dir, features, color_map, Benign, Attack)
                next_time = current_time + step_size
                if df_att[(df_att['timestamp'] >= next_time) & (df_att['timestamp'] < next_time + window_size)].empty:
                    next_time = df_att[df_att['timestamp'] > current_time + window_size].timestamp.min()

                current_time = next_time if next_time else end_time  # Avoid infinite loop
                gc.collect()

if __name__ == '__main__':

    df = pd.read_csv(PARAMETERS.UNSW_Niloo)
    Heading = 'UNSW_Niloo'

    w = 1800 # frame length in seconds
    s = 120 # steps in seconds
    folder_name = f'{w}_Seconds_{s}sec_{Heading}_KPCA'
    pca_frame_generation(df, folder_name, w, s)

    print('Done')
