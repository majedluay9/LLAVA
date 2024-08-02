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



class CsvProcessor:
    def __init__(self, fname, idx=[H_IP_SRC, H_IP_DST], attributes=[H_IN_BYTES]):
        self.fname = fname
        self.idx = idx
        self.attributes = attributes
        self.df = pd.read_csv(fname, usecols=[H_LAST_SWITCHED] + self.attributes + self.idx)
        self.df[H_LAST_SWITCHED] = pd.to_datetime(self.df[H_LAST_SWITCHED], unit='s')
        self.ip_map = {ip: idx for idx, ip in enumerate(set(self.df[H_IP_SRC].unique()).union(set(self.df[H_IP_DST].unique())))}
        self.num_nodes = len(self.ip_map)

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
        print(f'Now iterating over {len(time_bins)} windows ...')
        for start in tqdm(time_bins, total=len(time_bins), desc="Processing windows"):
            end = start + timedelta(seconds=window_size)
            mask = (self.df[H_LAST_SWITCHED] >= start) & (self.df[H_LAST_SWITCHED] < end)
            filtered_df = self.df.loc[mask]
            matrix = lil_matrix((self.num_nodes, self.num_nodes), dtype=np.int64)

            for _, row in filtered_df.iterrows():
                src_idx = self.ip_map[row[H_IP_SRC]]
                dst_idx = self.ip_map[row[H_IP_DST]]
                matrix[src_idx, dst_idx] += row[H_IN_BYTES]  # Incrementally update matrix

            # Convert the LIL matrix to CSR after all updates are done for efficient arithmetic operations later
            sparse_matrices.append(matrix.tocsr())
            time_labels.append(start)

        return sparse_matrices, time_labels

    def plot_matrices(self, matrices, time_labels, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, matrix in enumerate(matrices):
            plt.figure(figsize=(10, 10))
            plt.spy(matrix, markersize=5)
            plt.title(f'Time: {datetime.utcfromtimestamp(time_labels[i].timestamp()).strftime("%Y-%m-%d %H:%M:%S")}')
            plt.xlabel('Source IP Index')
            plt.ylabel('Destination IP Index')
            plt.savefig(f'{output_dir}/matrix_{i}.png')
            plt.close()


if __name__ == '__main__':
    FNAME = UNSW_Niloo
    RES_SMALL = 3600  # Temporal resolution in seconds
    Heading = f'UNSW_Niloo_{RES_SMALL}S_GPT'
    output_dir = rf"D:\Frames\LLAVA\{Heading}"


    csv_processor = CsvProcessor(FNAME)
    print('Now Binning ...')
    matrices, time_labels = csv_processor.bin_by_resolution(RES_SMALL,60)
    print('Now Plotting ...')
    csv_processor.plot_matrices(matrices, time_labels, output_dir)
