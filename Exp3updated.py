import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
np.random.seed(0)
data = np.random.rand(10, 5)  # 10 samples, 5 features

# Step 2: Initialize PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions

# Step 3: Fit and transform the data
reduced_data = pca.fit_transform(data)

# Step 4: Print the reduced data
print("Original Data Shape:", data.shape)
print("Reduced Data Shape:", reduced_data.shape)
print("Reduced Data:\n", reduced_data)

# Step 5: Optional - Visualize the reduced data
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('PCA - Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
