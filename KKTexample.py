# Define the objective function
def objective_function(x, y):
    return x**2 + y**2

# Define the constraint function
def constraint_function(x, y):
    return x + y - 1

# Gradient Descent Parameters
learning_rate = 0.1
num_iterations = 100

# Initial guess
x = 0.5
y = 0.5

# Gradient descent loop
for _ in range(num_iterations):
    # Calculate gradients
    grad_x = 2 * x  # Partial derivative of f with respect to x
    grad_y = 2 * y  # Partial derivative of f with respect to y

    # Update x and y according to the constraint
    x -= learning_rate * grad_x
    y -= learning_rate * grad_y
    
    # Ensure the constraint x + y = 1 is satisfied
    if (x + y) != 0:
        scale = (x + y) / 1
        x /= scale
        y /= scale

# Final optimized values
minimum_value = objective_function(x, y)

print(f"Optimized x: {x}")
print(f"Optimized y: {y}")
print(f"Minimum value of the function: {minimum_value}")
