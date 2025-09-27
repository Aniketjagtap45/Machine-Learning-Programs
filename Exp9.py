import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define your classification model
model = LogisticRegression()

# Sample data (replace with your actual data)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])  # Binary classification labels (0 or 1)

# Define the number of folds (common values: 5 or 10)
k = 5

# Initialize a KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Adjust random_state if needed

# Stores evaluation scores (accuracy) for each fold
scores = []

# K-Fold Cross-Validation Loop
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance (accuracy)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

# Print average score across all folds
average_accuracy = np.mean(scores)
print(f"Average Accuracy Score: {average_accuracy:.4f}")
