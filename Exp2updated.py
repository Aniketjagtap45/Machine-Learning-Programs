import numpy as np

# Define a simple 2x2 matrix
A = np.array([[4, 2], 
              [3, 1]])

print("Original Matrix A:")
print(A)

# Eigenvalue Decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalue Decomposition:")
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)

print("\nSingular Value Decomposition (SVD):")
print("U Matrix:\n", U)
print("Singular Values:", S)
print("Vt Matrix:\n", Vt)

# Reconstruct the matrix using SVD components
A_reconstructed = U @ np.diag(S) @ Vt
print("\nReconstructed Matrix from SVD:")
print(A_reconstructed)
