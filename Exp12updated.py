from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# Generate synthetic data for demonstration
X, y = make_blobs(n_samples=200, centers=3, random_state=42)
# Create a KMeans clustering object and specify the number of clusters
kmeans = KMeans(n_clusters=3) # You can choose the desired number of clusters
# Perform clustering on the data
kmeans.fit(X)
# Get the cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_
# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red')
plt.title("k-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()