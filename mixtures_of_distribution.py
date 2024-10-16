import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset from 'mcdonalds.csv'
# Replace 'path/to/mcdonalds.csv' with the actual path to your dataset
MD_x = pd.read_csv('mcdonalds.csv')

# If your dataset has columns with categorical data that needs encoding:
# You may want to ensure the data is binary, and if needed, you can convert it.
# For simplicity, assuming the dataset is already in binary format

# Function to perform Gaussian Mixture Model (GMM) for different cluster numbers (2 to 8)
def find_best_gmm(data, k_range, nrep=10):
    best_gmm = None
    best_bic = np.inf  # BIC (Bayesian Information Criterion) is used for model selection
    
    for k in k_range:
        for i in range(nrep):
            # Initialize GaussianMixture with k components (clusters)
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=np.random.randint(0, 10000))
            gmm.fit(data)
            
            # Compute BIC (lower BIC indicates better model)
            bic = gmm.bic(data)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

    return best_gmm

# Perform Gaussian Mixture modeling for k = 2 to 8 clusters, with 10 repetitions for each
MD_gmm = find_best_gmm(MD_x, range(2, 9), nrep=10)

# Optionally, relabel the clusters consistently (this is similar to relabeling in R's flexmix)
encoder = LabelEncoder()
MD_gmm_labels = encoder.fit_transform(MD_gmm.predict(MD_x))

# Print final cluster labels and cluster means (centroids)
print("Cluster labels:", MD_gmm_labels)
print("Cluster means (centroids):", MD_gmm.means_)
