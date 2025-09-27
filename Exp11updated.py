import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data (optional) or use your own data
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)  # Simple binary target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the neural network
input_size, hidden_size, output_size = 10, 5, 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
learning_rate = 0.01

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training the model
for epoch in range(1000):
    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    
    # Backward propagation
    dZ2 = A2 - y_train
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dZ1 = dZ2 @ W2.T * A1 * (1 - A1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Predict class labels for testing data
y_pred = sigmoid(X_test @ W1 + b1) @ W2 + b2
y_pred = (sigmoid(y_pred) > 0.5).astype(int)

# Evaluate the model's accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
