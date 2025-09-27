import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

x=np.array([[0,0],[1,1],[1,0],[0,1]])
y=np.array([0,1,1,0])

clt=DecisionTreeClassifier()
clt.fit(x,y)
fig=plt.subplots(figsize=(5,7))
tree.plot_tree(clt,filled=True)
plt.show()