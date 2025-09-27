import pandas as pd
from sklearn.datasets import make_blobs

# Generate high-dimensional data (adjust parameters as needed)
X, y = make_blobs(n_samples=500, n_features=20, random_state=42)

# Create a DataFrame and save to CSV
data = pd.DataFrame(X)
data.to_csv("high_dimensional_data.csv", index=False)

print("High-dimensional data saved to high_dimensional_data.csv")
