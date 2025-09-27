import numpy as np

# Create a 1D array
arr = np.arange(12)
print("Original array:", arr)

# Reshape to 2D array
reshaped_arr = arr.reshape((3, 4))
print("Reshaped array (3x4):")
print(reshaped_arr)

# Reshape to 3D array
reshaped_arr_3d = arr.reshape((2, 3, 2))
print("Reshaped array (2x3x2):")
print(reshaped_arr_3d)

import numpy as np
A=np.array([1,2,3,4])
B=A.reshape(2,2)
print("Matrix:",B)



import numpy as np
F=np.arange(4)
G=F.reshape(2,2)
H=np.array([5,6,7,8])
K=H.reshape(2,2)
J=G@K
print(J)














