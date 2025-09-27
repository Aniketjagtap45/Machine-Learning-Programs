import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Create DataFrame for easier manipulation
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Features and Target
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target (labels)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate and print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
