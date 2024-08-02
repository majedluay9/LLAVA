import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import itertools

# Here is the data per timestep
n_timesteps = 20
x_full = [np.random.randn(1000) for t in range(n_timesteps)]
y_full = [np.random.randn(1000) for t in range(n_timesteps)]

# Squish the data so we can work out the bins
bin_count = 20
x_full_squished = np.array(x_full).reshape(-1)
y_full_squished = np.array(y_full).reshape(-1)

x_bins = np.histogram_bin_edges(x_full_squished, bins=bin_count)
y_bins = np.histogram_bin_edges(y_full_squished, bins=bin_count)

# Function to create a histogram using pre-determined bin edges
def create_histogram(x, y, x_bins, y_bins):
    hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return hist

all_timestep_histograms = []
for timestep_i in range(n_timesteps):
    print(f"Processing {timestep_i=}")
    x_timestep = x_full[timestep_i]
    y_timestep = y_full[timestep_i]

    # Create histogram of the full data using fixed bins
    hist_timestep = create_histogram(x_timestep, y_timestep, x_bins, y_bins)
    all_timestep_histograms.append(hist_timestep)


    # Optional: Visualize the original histogram
    plt.figure(figsize=(8, 6))
    plt.imshow(hist_timestep, interpolation='nearest', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
    plt.colorbar()
    plt.title(f'2D Histogram with {bin_count} Bins, in timestamp {timestep_i}')
    plt.xlabel('x')
    plt.ylabel('y')

all_timestep_histograms = np.array(all_timestep_histograms)

# Flatten the histogram to use in KPCA
hist_flattened = all_timestep_histograms.flatten().reshape(n_timesteps, -1)

kpca = KernelPCA(n_components=2, kernel='rbf')
transformed_timesteps = kpca.fit_transform(hist_flattened)

for i in range(1, n_timesteps):
    # Print the transformed point
    plt.scatter(transformed_timesteps[:i, 0], transformed_timesteps[:i, 1], c=np.arange(0, i))
    plt.colorbar()
    plt.show()