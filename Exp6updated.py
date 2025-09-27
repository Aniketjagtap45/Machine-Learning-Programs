import numpy as np
from sklearn.linear_model import LinearRegression
# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 7, 8])
# Create and fit the model
model = LinearRegression()
model.fit(X, y)
# Predict the output
X_test = np.array([[6], [7], [8]])
y_pred = model.predict(X_test)
# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Print the predictions
print("Predictions:", y_pred)
