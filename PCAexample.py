import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Step 2: Mean center the data
X_meaned = X - np.mean(X, axis=0)

# Step 3: Calculate the covariance matrix
covariance_matrix = np.cov(X_meaned, rowvar=False)

#Columns are variables (features), rows are observations (samples). This is more commonly used when dealing with datasets in machine learning and data science.
#Each column is treated as a variable, and each row is treated as an observation, which is more typical in real-world datasets.


# Step 4: Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)


# Step 5: Sort the eigenvectors by eigenvalues in descending order
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

plt.bar(range(len(eigenvalues)),sorted_eigenvalues)
plt.xlabel("Indexes of eigenvalues")
plt.ylabel("eigenvalues")
plt.show()
# Step 6: Select the top k eigenvectors (k=2)
k = 1
eigenvector_subset = sorted_eigenvectors[:, 0:k]

# Step 7: Transform the data
X_reduced = np.dot(X_meaned, eigenvector_subset)

# Output the reduced data
print("Reduced Data:\n", X_reduced)

#import matplotlib.pyplot as plt
plt.scatter(X_reduced[:,0],X_reduced[:,-1])
plt.title('PCA - Dimension Reduction')
plt.xlabel('principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
