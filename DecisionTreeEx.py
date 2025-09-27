# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load a sample dataset (Iris dataset in this case)
data = datasets.load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
