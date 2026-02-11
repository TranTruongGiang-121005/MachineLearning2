import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_spectrogram_elbow(data_path, n_mels=128, duration=3):
    data=np.load("../spectrogram_vectors.npz")
    X = data['X']
    print(f"Data shape: {X.shape} (Samples, Pixel Features)")

    #standardize the features (Crucial for PCA)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #apply PCA
    print("Fitting PCA...")
    pca = PCA()
    pca.fit(X_scaled)

    #calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Spectrogram PCA Elbow Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    
    #add reference lines
    plt.axhline(y=0.80, color='r', linestyle='-', label='80% Variance')
    plt.axhline(y=0.90, color='g', linestyle='-', label='90% Variance')
    plt.legend(loc='best')
    plt.savefig('../plot.png')

    #find the exact number of components for 90% variance
    n_90 = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Number of components needed for 90% variance: {n_90}")
start_time=time.perf_counter()
plot_spectrogram_elbow('../genres_original')
end_time=time.perf_counter()
print(f"Execution time: {end_time-start_time: .4f} seconds")