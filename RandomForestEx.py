import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Assuming you have the heart.csv file in your working directory
data = pd.read_csv('heart.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 2: Preprocess the data
# Features (independent variables)
X = data.drop('target', axis=1)  # Drop the target column

# Target (dependent variable)
y = data['target']  # Keep the target column

# Step 3: Split the data into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model
# Initialize the Random Forest Classifier with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Step 5: Make predictions
# Predict the labels for the test set
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy * 100:.2f}%")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Feature Importance
# Get the importance of each feature in the prediction
feature_importances = rf_model.feature_importances_

# Create a bar plot for feature importance
plt.figure(figsize=(12, 8))
plt.barh(X.columns, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Heart Disease Prediction')
plt.show()
