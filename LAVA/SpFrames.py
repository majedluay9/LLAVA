import os
import pandas as pd
import numpy as np
import matplotlib
from PARAMETERS import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
from matplotlib.colors import LogNorm
import Functions
from VideoGeneration import create_video_from_frames




class CsvProcessor:
    def __init__(self, fname, idx=[H_IP_SRC, H_IP_DST], attributes=[H_IN_BYTES]):
        self.fname = fname
        self.idx = idx
        self.attributes = attributes
        self.df = pd.read_csv(fname, usecols=[H_LAST_SWITCHED] + self.attributes + self.idx + ['Label'])
        self.df[H_LAST_SWITCHED] = pd.to_datetime(self.df[H_LAST_SWITCHED], unit='s')
        if idx==[H_IP_SRC, H_IP_DST]:
            self.ip_map = {ip: idx for idx, ip in enumerate(set(self.df[H_IP_SRC].unique()).union(set(self.df[H_IP_DST].unique())))}
            self.num_nodes = len(self.ip_map)
            self.src_idx = H_IP_SRC
            self.dst_idx = H_IP_DST
        elif idx==[H_PORT_SRC, H_PORT_DST]:
            self.ip_map = {ip: idx for idx, ip in enumerate(set(self.df[H_PORT_SRC].unique()).union(set(self.df[H_PORT_DST].unique())))}
            self.num_nodes = len(self.ip_map)
            self.src_idx = H_PORT_SRC
            self.dst_idx = H_PORT_DST


    def bin_by_resolution(self, window_size, shift_size):
        start_time = self.df[H_LAST_SWITCHED].min()
        end_time = self.df[H_LAST_SWITCHED].max()
        time_bins = []

        print('Now counting the windows ...')
        current_start = start_time
        while current_start + timedelta(seconds=window_size) <= end_time:
            time_bins.append(current_start)
            current_start += timedelta(seconds=shift_size)

        sparse_matrices = []
        time_labels = []
        attacks = []
        attack_label = {}                                                                                                    # FOR LABELLING PURPOSEES
        print(f'Now iterating over {len(time_bins)} windows ...')
        for start in tqdm(time_bins, total=len(time_bins), desc="Processing windows"):
            end = start + timedelta(seconds=window_size)
            mask = (self.df[H_LAST_SWITCHED] >= start) & (self.df[H_LAST_SWITCHED] < end)
            filtered_df = self.df.loc[mask]
            matrix = lil_matrix((self.num_nodes, self.num_nodes), dtype=np.int64)
            label_count = 0
            for _, row in filtered_df.iterrows():
                label = row['Label']
                src_idx = self.ip_map[row[self.src_idx]]
                dst_idx = self.ip_map[row[self.dst_idx]]
                matrix[src_idx, dst_idx] += row[H_IN_BYTES]  # Incrementally update matrix
                if label == 1:
                    attack_label[src_idx, dst_idx] = 1
                    label_count +=1
                # Convert the LIL matrix to CSR after all updates are done for efficient arithmetic operations later
            if label_count > 0:
                print(f'{label_count} points were labelled in window {start}')
            sparse_matrices.append(matrix.tocsr())
            time_labels.append(start)
            attacks.append(attack_label)
        return sparse_matrices, time_labels, attacks

    def plot_matrices(self, matrices, time_labels, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, matrix in enumerate(matrices):
            fig, ax = plt.figure(figsize=(10, 10)), plt.gca()
            # Find non-zero locations and values
            src_indices, dst_indices = matrix.nonzero()

            if src_indices.size == 0 or dst_indices.size == 0:
                print(f"No data available for plotting at time {time_labels[i].strftime('%Y-%m-%d %H:%M:%S')}")
                continue  # Skip plotting if there are no data points

            values = matrix[src_indices, dst_indices].A.flatten()  # Proper conversion to 1D array for plotting

            # Use both color and size to represent values
            sc = ax.scatter(src_indices, dst_indices, c=values, s=values / values.max() * 100,
                            # Adjust size scaling as needed
                            cmap='viridis', norm=LogNorm(), alpha=0.6)
            # plt.colorbar(sc, ax=ax, label='Traffic Volume (bytes)')
            plt.title(f'Time: {time_labels[i].strftime("%Y-%m-%d %H:%M:%S")}')
            plt.xlabel('Source IP Index')
            plt.ylabel('Destination IP Index')
            ax.set_xlim([0, self.num_nodes])
            ax.set_ylim([0, self.num_nodes])
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.savefig(f'{output_dir}/matrix_{i}.png')
            plt.close()

        print(f"Plots saved in {output_dir}")



    def plot_matrices_Binary(self, matrices, time_labels, output_dir, attacks):
        output_dir += '_Binary'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, matrix in enumerate(matrices):
            fig, ax = plt.figure(figsize=(10, 10)), plt.gca()
            # Find non-zero locations and values
            src_indices, dst_indices = matrix.nonzero()

            if src_indices.size == 0 or dst_indices.size == 0:
                print(f"No data available for plotting at time {time_labels[i].strftime('%Y-%m-%d %H:%M:%S')}")
                continue  # Skip plotting if there are no data points

            values = matrix[src_indices, dst_indices].A.flatten()  # Proper conversion to 1D array for plotting
            colors = ['red' if attacks[i].get((src, dst), 0) == 1 else 'gray' for src, dst in
                      zip(src_indices, dst_indices)]

            # Use both color and size to represent values
            sc = ax.scatter(src_indices, dst_indices, c=colors, s=50,
                            cmap='viridis', norm=LogNorm(), alpha=0.6)
            plt.title(f'Time: {time_labels[i].strftime("%Y-%m-%d %H:%M:%S")}')
            plt.xlabel('Source IP Index')
            plt.ylabel('Destination IP Index')
            ax.set_xlim([0, self.num_nodes])
            ax.set_ylim([0, self.num_nodes])
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.savefig(f'{output_dir}/matrix_{i}.png')
            plt.close()

        print(f"Plots saved in {output_dir}")


if __name__ == '__main__':
    FNAME = r"D:\attack_injected_data\REAL_SUB_DFS\v1712_IP21361_10a_t1234\2017-07-06_Labelled.csv"
    Window_len = 3600  # Temporal resolution in seconds
    Shifting = 60
    Heading = f'REAL_SIAMAK_DATA_2017-07-06_IN_BYTES_Labelled_Friday'


    output_dir = rf"D:\Frames\LLAVA\{Heading}\IPs_matrix"
    csv_processor_ip = CsvProcessor(FNAME,idx=[H_IP_SRC, H_IP_DST])
    print('Now Binning IPs...')
    matrices, time_labels, attacks = csv_processor_ip.bin_by_resolution(Window_len,Shifting)
    print('Now Plotting IPs...')
    csv_processor_ip.plot_matrices_Binary(matrices, time_labels, output_dir,attacks)
    csv_processor_ip.plot_matrices(matrices, time_labels, output_dir)

    create_video_from_frames(output_dir, f'{output_dir}_IPs.mp4', frame_rate=10)
    create_video_from_frames(output_dir + '_Binary', f'{output_dir}_IPs_Binary.mp4', frame_rate=10)
    Functions.empty_folder(output_dir)
    Functions.empty_folder(output_dir + '_Binary')



    output_dir = rf"D:\Frames\LLAVA\{Heading}\Ports_matrix"
    csv_processor_ports = CsvProcessor(FNAME,idx=[H_PORT_SRC, H_PORT_DST])
    print('Now Binning Ports...')
    matrices, time_labels = csv_processor_ports.bin_by_resolution(Window_len,Shifting)
    print('Now Plotting Ports...')
    csv_processor_ports.plot_matrices_Binary(matrices, time_labels, output_dir)
    csv_processor_ports.plot_matrices(matrices, time_labels, output_dir)

    create_video_from_frames(output_dir, f'{output_dir}_PORTs.mp4', frame_rate=10)
    create_video_from_frames(output_dir+ '_Binary', f'{output_dir}_PORTs_Binary.mp4', frame_rate=10)

    Functions.empty_folder(output_dir)
    Functions.empty_folder(output_dir+ '_Binary')




    print('Done')