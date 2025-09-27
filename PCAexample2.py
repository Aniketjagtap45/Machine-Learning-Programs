import numpy as np
import matplotlib.pyplot as plt

# Manually created real-life dataset (Housing Data)
# Features: [Square Footage, Number of Rooms, Number of Bathrooms, Price ($1000s), Year Built]
data = np.array([
    [2100, 5, 3, 550, 1990],
    [1600, 4, 2, 400, 2000],
    [2400, 5, 3, 600, 1985],
    [1410, 3, 2, 350, 2005],
    [1800, 4, 3, 450, 1998],
    [3000, 6, 4, 700, 1992],
    [2000, 4, 3, 500, 2000],
    [1500, 3, 2, 375, 2002],
    [2800, 5, 4, 675, 1987],
    [2200, 5, 3, 530, 1995]
])

# Step 1: Standardize the data (mean = 0, variance = 1)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# Step 2: Compute the covariance matrix
cov_matrix = np.cov(standardized_data.T)

# Step 3: Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvalues and eigenvectors in descending order of eigenvalue magnitude
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Select the top 2 eigenvectors (for a 2D reduction)
top_2_eigenvectors = sorted_eigenvectors[:, :2]

# Step 6: Transform the data into the new 2D space
pca_transformed_data = np.dot(standardized_data, top_2_eigenvectors)

# Step 7: Plot the PCA-transformed data in 2D
plt.figure(figsize=(8, 6))
plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], color='blue', label='Houses')
plt.title('PCA on Housing Data (Reduced to 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend()
plt.show()

# Print results
print("Eigenvalues (sorted):")
print(sorted_eigenvalues)

print("\nTop 2 Principal Components (Eigenvectors):")
print(top_2_eigenvectors)

print("\nPCA Transformed Data (2D):")
print(pca_transformed_data)
