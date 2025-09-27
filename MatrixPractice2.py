
import numpy as np
import scipy.linalg as linalg

# Define a matrix
A = np.array([[1, 2], [3, 4]])

# Compute the inverse
A_inv = linalg.inv(A)
print("Inverse of A:\n", A_inv)

# Compute the determinant
det_A = linalg.det(A)
print("Determinant of A:", det_A)

transpose=A.T
print(transpose)