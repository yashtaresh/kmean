import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

MD_x = pd.read_csv('mcdonalds.csv')


# Set seed for reproducibility
np.random.seed(1234)

# Function to perform k-means clustering for different cluster numbers (from 2 to 8)
def find_best_kmeans(data, k_range, nrep=10):
    best_kmeans = None
    best_inertia = np.inf  # Inertia is a measure of clustering performance
    
    for k in k_range:
        for i in range(nrep):
            kmeans = KMeans(n_clusters=k, random_state=np.random.randint(0, 10000))
            kmeans.fit(data)
            
            # Track the best clustering based on inertia (lower is better)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
    
    return best_kmeans

# Perform k-means clustering for k = 2 to 8 clusters, repeated 10 times for each
MD_kmeans = find_best_kmeans(MD_x, range(2, 9), nrep=10)

# Optionally, relabel the clusters
encoder = LabelEncoder()
MD_kmeans.labels_ = encoder.fit_transform(MD_kmeans.labels_)

# Print the final cluster labels and centroids
print("Cluster labels:", MD_kmeans.labels_)
print("Cluster centroids:", MD_kmeans.cluster_centers_)
