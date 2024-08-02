import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

Heading = 'UNSW_Niloo'

dfs_dir = f"D:\python datasets\Liam\\{Heading}"
os.makedirs(dfs_dir, exist_ok=True)
file_list = os.listdir(dfs_dir)
csv_files = [file for file in file_list if file.endswith('.csv')]

# __________________________________________________ Here start reading Liam's code ______________________________
for i, csv_file in tqdm(enumerate(csv_files), desc="Reading CSV files"):
    df = pd.read_csv(os.path.join(dfs_dir, csv_file), usecols=['IN_BYTES'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    kpca = KernelPCA(n_components=2, kernel='rbf')
    transformed_data = kpca.fit_transform(df)

    plt.figure(figsize=(8,6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title('Kernel PCA Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    plt.close()
