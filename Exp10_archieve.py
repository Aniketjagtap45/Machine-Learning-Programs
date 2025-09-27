from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris  # Replace with your dataset
from sklearn.model_selection import train_test_split

# Load data (replace with your data loading)
X, y = load_iris().data, load_iris().target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust n_estimators as needed

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance (replace with your desired metric)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")
