import matplotlib
matplotlib.use('Agg')
import scipy.sparse as sp
import matplotlib.cm as cm
from datetime import timedelta
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



def plot_transformed(transformed, start_time, res, name='embedding.png'):

    #dist = scipy.spatial.distance_matrix(transformed, transformed)
    duration = transformed.shape[0]

    fig, ax = plt.subplots()
    plt.scatter(transformed[:,0], transformed[:,1], c=range(duration), 
            cmap=cm.viridis, s=2)
    plt.colorbar(orientation = 'horizontal', pad=0.2)
    """
    for i, t in enumerate(range(duration)): 
        if np.amax(dist[i,:]) == np.amax(dist):
            txt = time_idx_to_time(t, res, start_time)
            ax.annotate(txt, (transformed[i,0], transformed[i,1]))
    """
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.7)
    plt.title(f'{name}')
    plt.axis('equal')
    #plt.xlim([-2, 2])
    #plt.ylim([-2, 2])
    plt.savefig(name)
    # plt.show()
    plt.close()

def plot_transformed_eachwindow(transformed_data, start_time, resolution, output_folder,consistient_scale:bool,Shifting, Benign=None,Attack=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    num_frames = transformed_data.shape[0]
    attack_times = ['10:28', '10:48', '11:09', '11:39']
    color_states = ['blue'] * num_frames
    # Consistient frame sizes
    x_min, x_max, y_min, y_max = const_Xlim_Ylim((transformed_data))


    for i in range(num_frames):
        plt.figure(figsize=(10, 8))
        start_time += Shifting
        W_start = datetime.utcfromtimestamp(start_time).strftime('%H:%M')

        current_color = 'red' if W_start in attack_times else 'blue'
        color_states[i] = current_color  # Update the color state for the current point
        plt.scatter(transformed_data[:i+1, 0], transformed_data[:i+1, 1], c=color_states[:i+1])
        #
        # if i > 0:
        #     plt.scatter(transformed_data[:i, 0], transformed_data[:i, 1], c='gray')
        # # Plot current point in blue
        # plt.scatter(transformed_data[i, 0], transformed_data[i, 1], c=colors)

        if consistient_scale:
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
        if Benign is None:
            plt.title(f'Window starts at {W_start}')
        else:
            plt.title('KPCA Transformation at ' + datetime.utcfromtimestamp(start_time + i * resolution ).strftime(
                '%H:%M:') + f'Benign:{Benign}, Attacks:{Attack}')

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.xlabel('')
        plt.ylabel('')
        plt.grid(True)

        plt.savefig(os.path.join(output_folder, f'frame_{i + 1}.png'))
        plt.close()


    print("All frames saved in:", output_folder)





# Example usage for const_Xlim_Ylim (assuming this function returns consistent limits)
def const_Xlim_Ylim(data):
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    return x_min, x_max, y_min, y_max


def plot_transformed_eachwindow_label(transformed_data, timestamps_attacks, start_time,end_time, resolution, output_folder, consistient_scale, Shifting):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    num_frames = transformed_data.shape[0]

    for i in range(num_frames):
        plt.figure(figsize=(10, 8))

        current_time = start_time + i * Shifting
        end_time = current_time + 3600
        W_start = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        if isinstance(start_time, (int, float)):
            current_time = pd.to_datetime(start_time, unit='s') + i * Shifting
        else:
            current_time = start_time

        # Convert attack timestamps to datetime objects if they're not already
        attack_times = [pd.to_datetime(ts, unit='s') if isinstance(ts, (int, float)) else ts for ts in
                        timestamps_attacks]

        # Plot all previous points in gray
        if i > 0:
            plt.scatter(transformed_data[:i, 0], transformed_data[:i, 1], c='gray')
        # Plot current point in blue or red depending on if it's an attack
        is_attack_time = any(current_time <= attack_time + timedelta(hours=1) and current_time >= attack_time - timedelta(hours=1) for attack_time in attack_times)
        point_color = 'red' if current_time in timestamps_attacks else 'blue'
        plt.scatter(transformed_data[i, 0], transformed_data[i, 1], c=point_color)


        plt.title(f'KPCA Transformation at {W_start}, Status: {"Attack" if point_color == "red" else "Benign"}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)

        plt.savefig(os.path.join(output_folder, f'frame_{i + 1}.png'))
        plt.close()

    print("All frames saved in:", output_folder)


def plot_transformed_eachwindow_Threshold(transformed_data, start_time, resolution, output_folder,consistient_scale:bool,Shifting):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    num_frames = transformed_data.shape[0]


    # Initialize mean and standard deviation calculations
    mean_distance = 0
    sum_squared_distances = 0
    num_points = 0
    threshold_multiplier = 3  # You can adjust this multiplier

    for i in range(num_frames):
        plt.figure(figsize=(10, 8))
        start_time += Shifting
        W_start = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

        distance = np.sqrt(np.sum((transformed_data[i] - transformed_data[i - 1]) ** 2))
        num_points += 1
        delta = distance - mean_distance
        mean_distance += delta / num_points
        delta2 = distance - mean_distance
        sum_squared_distances += delta * delta2
        std_dev = np.sqrt(sum_squared_distances / num_points) if num_points > 1 else 0
        threshold = mean_distance + threshold_multiplier * std_dev


        color = 'red' if distance > threshold else 'blue'

        if i > 0:
            plt.scatter(transformed_data[:i, 0], transformed_data[:i, 1], c='gray')
        # Plot current point in blue
        plt.scatter(transformed_data[i, 0], transformed_data[i, 1], c=color)
        if consistient_scale:
            x_min, x_max, y_min, y_max = const_Xlim_Ylim((transformed_data))
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])

        plt.title(f'From {W_start}')


        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.xlabel('')
        plt.ylabel('')
        plt.grid(True)

        plt.savefig(os.path.join(output_folder, f'frame_{i + 1}.png'))
        plt.close()


    print("All frames saved in:", output_folder)

def remove_old_data(data, res_big, limit=2800):
    """
    Keep the last RES_BIG datapoints, and make up the rest so that the shape[0]
    of data is less than limit (default 2800).
    """
    n = data.shape[0]
    if n <= limit:
        return data
    else:
        last_data = data[-res_big:,:]

    extra_data_size = limit - res_big
    idx = list(np.unique(np.linspace(0, n-res_big, extra_data_size, dtype=int)))
    first_data = data[idx, :]
    return sp.vstack((first_data, last_data))

def time_idx_to_time(t, res, start_time):
    posix_time = start_time + t*res
    return datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%d %H:%M:%S')

def drop_sparse_cols(a):
    """
    Hack to remove columns that are always zero from a sparse matrix.
    """
    a = sp.coo_matrix(a)
    nz_cols, new_col = np.unique(a.col, return_inverse=True)                    ## Return the index of each unique value and can be useful if I want to get rid of some values

    a.col[:] = new_col
    a._shape = (a.shape[0], len(nz_cols))
    return sp.csr_matrix(a)


def sparse_pdist(a):
    K = np.empty((a.shape[0], a.shape[0]))

    for row_i in range(a.shape[0]):
        K[row_i, row_i] = 0
        for row_j in range(row_i+1, a.shape[0]):
            K[row_i, row_j] = sparse_norm(a[row_i,:] - a[row_j,:])
            K[row_j, row_i] = K[row_i, row_j]
            
    return K

def sparse_norm(v):
    v2 = v.power(2)
    return np.sqrt(v2.sum())

def update_moving_average(raw_data, old_average, n):
    t = old_average.shape[0]
    raw_data = raw_data[max(t-n+1,0):,:]
    new_av = moving_average(raw_data, n)

    # pad the old average if required
    if new_av.shape[1] > old_average.shape[1]:
        pad = new_av.shape[1] - old_average.shape[1]
        expansion = sp.csr_matrix((old_average.shape[0], pad))

        old_average = sp.hstack((old_average, expansion))

    return sp.vstack((old_average, new_av))

def apply_moving_average_and_update_counts(data, counts, n):                                                                 # For Labels
    """
    Applies moving average to the data and updates attack counts to match the resulting data shape.
    """
    initial_rows = data.shape[0]
    data = moving_average(data, n)
    if data.shape[0] != initial_rows:
        # Update counts to match the resulting data rows
        counts = counts[n-1:]  # Assuming counts is a list or similar iterable
    return data, counts


def moving_average(a, n):
    rows = a.shape[0]
    if n >= rows:
        print("Tried to do moving average with small matrix.")
        return sp.csr_matrix((0,a.shape[1]))
    ones = np.ones(rows)/n
    sparse_ones = sp.diags(n*[ones], list(range(n)))
    return (sparse_ones*a)[n-1:,:]


# consistient xlim and ylim
def const_Xlim_Ylim(transformed_timesteps):
    # CALCULATE THE X SCALE AND Y SCALE
    x_min, x_max = np.min(transformed_timesteps[:, 0]), np.max(transformed_timesteps[:, 0])
    y_min, y_max = np.min(transformed_timesteps[:, 1]), np.max(transformed_timesteps[:, 1])

    # Expand the range a bit for better visualization
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    return x_min,x_max,y_min,y_max