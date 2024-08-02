import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import itertools

# Here is the data per timestep
n_timesteps = 10
x_full = np.array([np.random.randn(1000) for t in range(n_timesteps)])
y_full = np.array([np.random.randn(1000) for t in range(n_timesteps)])

# Squish the data so we can work out the bins
bin_count = 20
x_full_squished = np.array(x_full.reshape(-1))
y_full_squished = np.array(y_full.reshape(-1))

x_bins = np.histogram_bin_edges(x_full_squished, bins=bin_count)
y_bins = np.histogram_bin_edges(y_full_squished, bins=bin_count)

print(x_bins)
print(y_bins)

# Function to create a histogram using pre-determined bin edges
def create_histogram(x, y, x_bins, y_bins):
    hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return hist

# Create histogram of the full data using fixed bins
hist_full = create_histogram(x_full, y_full, x_bins, y_bins)

# Flatten the histogram to use in KPCA
hist_flattened = hist_full.flatten().reshape(1, -1)

# Apply Kernel PCA to reduce the histogram to a single point (x, y) in 2D
kpca = KernelPCA(n_components=2, kernel='rbf')
transformed_point = kpca.fit_transform(hist_flattened)

# Print the transformed point
print("Transformed point (x, y):", transformed_point)

# Optional: Visualize the original histogram
plt.figure(figsize=(8, 6))
plt.imshow(hist_full, interpolation='nearest', origin='lower',
           extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
plt.colorbar()
plt.title('2D Histogram with Fixed Bins')
plt.xlabel('x')
plt.ylabel('y')
plt.show()