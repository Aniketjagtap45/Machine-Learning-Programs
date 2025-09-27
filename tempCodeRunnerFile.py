import numpy as np
F=np.arange(4)
G=F.reshape(2,2)
H=np.array([5,6,7,8])
K=H.reshape(2,2)
J=G@K
print(J)