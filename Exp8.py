import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define data path (replace with your actual file path)
data_path = "C:\Users\Shree\Machine Learning Programs\high_dimensional_data.csv"

# Function to load data with error handling
def load_data(data_path):
  try:
    data = pd.read_csv(data_path)
    print("Data loaded successfully!")
    return data
  except FileNotFoundError:
    print(f"Error: File '{data_path}' not found.")
    return None

# Load data using the defined function
data = load_data(data_path)

# Check if data was loaded successfully before proceeding
if data is not None:
  # t-SNE parameters
  n_components = 2  # Target dimensionality (e.g., 2 for 2D plot)
  perplexity = 50  # Experiment with different values

  # Separate features (optional for coloring by target)
  target_column = "target_label"  # Replace with your target column name (if applicable)
  features = data.drop(target_column, axis=1)
  target = data[target_column]  # Optional for coloring data points

  # Apply t-SNE
  tsne = TSNE(n_components=n_components, perplexity=perplexity)
  tsne_features = tsne.fit_transform(features)

  # Visualizationimport pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample high-dimensional data (replace with your actual data)
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [6, 7, 8, 9, 10],
    'feature3': [11, 12, 13, 14, 15],
    # Add a target column if you have one for coloring (optional)
    # 'target_label': ['A', 'B', 'A', 'B', 'A']
})

# Function to load data (assuming your data is a CSV file)
def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        print("Data loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found.")
        return None

# Load data using the defined function (uncomment if using a CSV)
# data = load_data(data_path)  # Replace data_path with your actual file path

# Check if data was loaded successfully before proceeding
if data is not None:
    # t-SNE parameters
    n_components = 2  # Target dimensionality (e.g., 2 for 2D plot)
    perplexity = 50  # Experiment with different values

    # Separate features (optional for coloring by target)
    target_column = "target_label"  # Replace with your target column name (if applicable)

    try:
        features = data.drop(target_column, axis=1)
        target = data[target_column]  # Optional for coloring data points
    except KeyError:
        print("Warning: Target column not found. Visualization will proceed without coloring.")
        features = data

    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_features = tsne.fit_transform(features)

    # Visualization
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=target if 'target_label' in data.columns else 'blue')  # Adjust color based on target (if applicable)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization")
    plt.show()

    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=target)  # Adjust color based on target (if applicable)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization")
    plt.show()
