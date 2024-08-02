import os
import pandas as pd
import matplotlib
import Functions
from VideoGeneration import create_video_from_frames
from PARAMETERS import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
import scipy.spatial
import numpy as np
import sklearn.decomposition
from datetime import datetime
import utils as utils

class CsvProcessor(object):
    def __init__(self, fname,
                 attributes=[H_IN_BYTES,H_OUT_BYTES,H_IN_PKTS,H_OUT_PKTS],
                 idx=[H_IP_SRC, H_IP_DST]):
        self.fname = fname
        self.idx = idx
        self.aggregation = 'sum'
        # READ THE FILE INTO A PANDAS DATAFRAME
        self.df = pd.read_csv(fname, usecols=[H_LAST_SWITCHED] + attributes + idx)
        self.df[H_LAST_SWITCHED] = self.df[H_LAST_SWITCHED].astype('int64')
        self.num_nodes = max(self.df[idx[0]].nunique(),
                             self.df[idx[1]].nunique())
        self.attributes = attributes
        self.start_time = self.df[H_LAST_SWITCHED].min()
        self.end_time = self.df[H_LAST_SWITCHED].max()
        # self.log_transform_attributes()                                                                                   # I need to put numerical features into log + 1 scale

    def log_transform_attributes(self):
        for attr in self.attributes:
            if attr in [H_IN_PKTS, H_OUT_PKTS, H_IN_BYTES, H_OUT_BYTES]:  # Ensure the column is numeric
                self.df[attr] = self.log_transform_feature(self.df[attr])

    def log_transform_feature(self, data, small_constant=1):
        data = np.array(data)
        transformed_data = np.log(data + small_constant)
        return np.int64(transformed_data)
    def bin_by_resolution(self, res_small,res_big):
        binned_df = self.df.copy()
        bins = list(range(self.start_time - res_small,
                          self.end_time + res_small, res_small - (res_small - res_big)))                                        # Here I modify the code to implement shifting
        times = list(range(
            len(bins) - 1))  # times in units (res seconds)                         ## use for labels in the next line
        binned_df[H_BIN] = pd.cut(self.df[H_LAST_SWITCHED], bins=bins,
                                  labels=times)  ## output categorical list and then be a new column of the df

        # DROP UNECESSARY SWITCHING TIMES
        binned_df = binned_df.drop(columns=[H_LAST_SWITCHED])

        # SORT BY IP TO GET AN ORDERING OF IPS
        binned_df = binned_df.sort_values(by=[self.idx[0]])

        # CONVERT FROM IP STRINGS FROM OBJECTS TO CATEGORIES
        for col in self.idx:
            binned_df[col] = binned_df[col].astype('category')
        cat_columns = binned_df.select_dtypes(['category']).columns
        binned_df[cat_columns] = binned_df[cat_columns].apply(
            lambda x: x.cat.codes)  ## return int related to categorical value, I might use it directly without lambda

        if self.aggregation == 'sum':
            binned_df = binned_df.groupby([H_BIN, self.idx[0], self.idx[1]]).sum().reset_index()                                ## Aggregation and I can change the sum to max or avg
        elif self.aggregation == 'avg':
            binned_df = binned_df.groupby([H_BIN, self.idx[0], self.idx[1]]).avg().reset_index()
        elif self.aggregation == 'max':
            binned_df = binned_df.groupby([H_BIN, self.idx[0], self.idx[1]]).max().reset_index()
        return binned_df

    def add_new_data(self, fname, old_sparse, res_small, res_big):
        """
        Add new data stored in fname csv to the existing pandas data.
        Also return the new sparse array with the complete history, including
        the new data
        """
        num_nodes_before_new_data = self.num_nodes
        new_df = pd.read_csv(fname,
                             usecols=[H_LAST_SWITCHED] + self.attributes + self.idx)
        self.df = pd.concat([self.df, new_df])

        # Update start and end times, number of nodes
        old_end_time = self.end_time
        old_start_time = self.start_time
        self.end_time = max(self.end_time, new_df[H_LAST_SWITCHED].max())
        self.start_time = min(self.start_time, new_df[H_LAST_SWITCHED].min())
        self.num_nodes = max(self.df[self.idx[0]].nunique(),
                             self.df[self.idx[1]].nunique())

        # If new nodes have been added to the graph, we need to modify
        # the number of columns stored in old data matrix
        if self.num_nodes != num_nodes_before_new_data:
            num_new_nodes = self.num_nodes - num_nodes_before_new_data
            num_new_features = num_new_nodes * num_nodes_before_new_data * 2 + \
                               num_new_nodes ** 2
            num_new_features = num_new_features * len(self.attributes)
            expansion = sp.csr_matrix((old_sparse.shape[0], num_new_features))

            old_sparse = sp.hstack((old_sparse, expansion))

        new_sparse = self.get_sparse_array(res_small, res_big,
                                           old_array=old_sparse)
        return new_sparse

    def plot_by_resolution(self, res, name=''):
        binned = self.bin_by_resolution(res)
        duration = binned[H_BIN].max()

        for t in range(duration):
            snapshot = binned[binned[H_BIN] == t]
            snapshot = snapshot.values[:, 1:]  # Ignore the time column

            posix_time = self.start_time + t * res

            plt.plot(snapshot[:, 0], snapshot[:, 1], 's')  # Plot only src and dst
            plt.xlim([0, self.num_nodes])
            plt.ylim([0, self.num_nodes])
            plt.title(datetime.utcfromtimestamp(posix_time). \
                      strftime('%Y-%m-%d %H:%M:%S'))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(name + str(t) + '.png')
            plt.close()

    def get_sparse_array(self, res_small, res_big, old_array=None):
        # data = self._load_npz_if_exists(res_small, res_big)
        # if not (data is None):
        #    return data

        if not (old_array is None):                                                                                         ### Condition to start sparsing from zero or after last array
            start_from = old_array.shape[0]
        else:
            start_from = 0

        binned = self.bin_by_resolution(res_small,res_big)                                                                          ## Result in aggregated df
        duration = binned[H_BIN].max()

        data = sp.lil_matrix((duration - start_from, self.num_nodes ** 2 * len(self.attributes)))                                            ## Sparse matrix generation

        # Data is ordered in time
        for t in range(start_from, duration):
            snapshot = binned[binned[H_BIN] == t]
            snapshot = snapshot.values[:, 1:]                                                                              # Ignore the time column

            for row_idx in range(snapshot.shape[0]):
                row = snapshot[row_idx, :]
                # Note: Even if some of the attributes take a value of 0,
                # we explicitly store them. This is desired.
                for attribute in range(len(self.attributes)):
                    idx = self.src_dst_att_to_idx(row[0], row[1], attribute)
                    data[t - start_from, idx] = row[attribute]

        data = sp.csr_matrix(data)

        if not (old_array is None):
            data = sp.vstack((old_array, data))

        # sp.save_npz(self._gen_save_fname(res_small, res_big), data)

        return data

    def _gen_save_fname(self, res_small, res_big):
        return \
                self.fname[:-4] + '_' + self.idx[0] + '_' + str(res_small) + '_' + str(res_big) + \
                '.npz'

    def _load_npz_if_exists(self, res_small, res_big):
        fname = self._gen_save_fname(res_small, res_big)
        try:
            return sp.load_npz(fname)
        except:
            return None

    def src_dst_att_to_idx(self, src, dst, a):
        """
        Convert 3 dimensional index (source, destination, attribute) to 1
        dimension.
        """
        return (self.num_nodes * src + dst) * (a + 1)


if __name__ == '__main__':

    FNAME = v1712_IP21361_12a_t1234
    FNAME = FNAME[:-4] + '_DoS_Day8' + FNAME[-4:]
    # FNAME = FNAME[:-4] + '_LogScalled' + FNAME[-4:]

    # FNAME2 = PARAMETERS.REAL2

    RES_SMALL = 3600 #60 * 60  # Temporal resolution of averaging (in seconds)
    RES_BIG = 60 # Number of temporal resolution units per timestep ( in Seconds)
    consistient_scale = False
    Heading = f'v1712_IP21361_12a_t1234_{RES_SMALL}S_{RES_BIG}S_DoS_Day8_SUM'

    if FNAME.__contains__('_LogScalled'):
        Heading += '_LogScalled'

    csv_processor_ip = CsvProcessor(FNAME, idx=[H_IP_SRC, H_IP_DST])
    # csv_processor_ip.plot_by_resolution(RES_SMALL*RES_BIG, name='ip')
    data_ip = csv_processor_ip.get_sparse_array(RES_SMALL, RES_BIG)  ## data_ip is sparse lil matrix
    # data_ip = csv_processor_ip.add_new_data(FNAME2, data_ip, RES_SMALL,                                                       # Uncomment these to add extra data
    #                                         RES_BIG)  ## Read new df and make the first one as a history
    n_features_ip = \
        csv_processor_ip.num_nodes ** 2 * len(csv_processor_ip.attributes)
    data_ip = utils.drop_sparse_cols(data_ip)  ## Drop zeros
    data_ip = utils.moving_average(data_ip, RES_BIG)
    data_ip = sklearn.preprocessing.scale(data_ip,
                                          with_mean=False)  ## scale the values and with means=false because it is a sparse matrix
    output_dir = rf"D:\Frames\LIAM\{Heading}"
    output_dir_ip = os.path.join(output_dir, 'ip')
    print('Now Generating IPs frames ... ')
    kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', gamma=1 / (n_features_ip))
    transformed_ip = kpca.fit_transform(data_ip)
    utils.plot_transformed_eachwindow(transformed_ip, csv_processor_ip.start_time, RES_SMALL, output_dir_ip,
                                      consistient_scale, RES_BIG)
    create_video_from_frames(output_dir_ip, f'{output_dir}_IPs.mp4', frame_rate=10)




    csv_processor_port = CsvProcessor(FNAME, idx=[H_PORT_SRC, H_PORT_DST])  ## Same process with ports instead of IPs
    # csv_processor_port.plot_by_resolution(RES_SMALL*RES_BIG, name='port')
    data_port = csv_processor_port.get_sparse_array(RES_SMALL, RES_BIG)


    # data_port = csv_processor_port.add_new_data(FNAME2, data_port, RES_SMALL, RES_BIG)


    n_features_port = \
        csv_processor_port.num_nodes ** 2 * len(csv_processor_port.attributes)


    data_port = utils.drop_sparse_cols(data_port)
    data_port = utils.moving_average(data_port, RES_BIG)


    data_port = sklearn.preprocessing.scale(data_port, with_mean=False)
    data_all = sp.hstack((data_ip, data_port))


    output_dir_port = os.path.join(output_dir, 'port')
    output_dir_all = os.path.join(output_dir, 'all')

    # Execute KPCA and plot each step

    print('Now Generating Ports frames ... ')
    kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', gamma=1 / (n_features_port))
    transformed_port = kpca.fit_transform(data_port)
    utils.plot_transformed_eachwindow(transformed_port, csv_processor_port.start_time, RES_SMALL, output_dir_port,consistient_scale,RES_BIG)
    print('Now Generating all frames ... ')
    kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', gamma=1 / ((n_features_ip + n_features_port)))
    transformed_all = kpca.fit_transform(data_all)
    utils.plot_transformed_eachwindow(transformed_all, csv_processor_ip.start_time, RES_SMALL, output_dir_all,consistient_scale,RES_BIG)

    # print('Now Generating Videos ... ')
    # Generate Videos
    create_video_from_frames(output_dir_port, f'{output_dir}_PORTs.mp4', frame_rate=10)
    create_video_from_frames(output_dir_all, f'{output_dir}_ALL.mp4', frame_rate=10)


    # Empty the folders after Generating the videos
    print('Now deleting all frames after generating the videos ... ')
    Functions.empty_folder(output_dir_ip)
    Functions.empty_folder(output_dir_port)
    Functions.empty_folder(output_dir_all)

    try:
        os.rmdir(output_dir)
        print(f"Directory '{output_dir}' has been deleted successfully")
    except OSError as e:
        print(f"Error: {e.strerror}")

    print('Done')