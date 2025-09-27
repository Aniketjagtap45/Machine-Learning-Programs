from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
irisData = load_iris()
X = irisData.data
y = irisData.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the number of neighbors to a specific value, e.g., 5
k = 5

# Initialize the KNN classifier with the specified number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Compute training and testing accuracy
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

# Print the training and testing accuracy
print(f"Training Accuracy (k={k}): {train_accuracy}")
print(f"Testing Accuracy (k={k}): {test_accuracy}")

# Plot the training and testing accuracy for the specified k value
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title(f'KNN Accuracy with k={k}')
plt.show()
