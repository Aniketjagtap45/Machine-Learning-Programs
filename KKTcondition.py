import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2

# Define the constraint function
def constraint_function(x):
    return x[0] + x[1] - 1

# Initial guess for the variables
initial_guess = [0.5, 0.5]

# Define the constraint in the format required by 'minimize'
constraints = {'type': 'eq', 'fun': constraint_function}

# Perform the optimization
result = minimize(objective_function, initial_guess, constraints=constraints)

# Extract the optimized variables and the minimum value
optimized_x = result.x
minimum_value = result.fun

print(f"Optimized x: {optimized_x}")
print(f"Minimum value of the function: {minimum_value}")
