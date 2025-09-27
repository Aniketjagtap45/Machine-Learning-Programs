# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with your dataset)
data = pd.read_csv('email_spam_dataset.csv')

# Display first few rows to understand the dataset
print(data.head())

# Assume the dataset has a 'fraud' column that indicates fraudulent transactions (1 for fraud, 0 for non-fraud)
# Also assume the rest are feature columns (like 'amount', 'age', etc.)

# Define the feature columns and target variable
X = data.drop('Is_Spam', axis=1)  # Features
y = data['Is_Spam']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  #By default 0.25 means 25% dataset will go for test and remaining 75% dataset will go for training.

# Initialize the Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tree.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(6,7))
feature_names = data.columns.drop('Is_Spam')  # Drop the target column
class_names = ['Not Spam', 'Spam'] 
plot_tree(tree, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()
