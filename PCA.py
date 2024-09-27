from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA

# Exercise 1
# 1.Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target

print(f"Dataset shape: {X.shape}")  

# 2. Display one instance of each digit
def plot_unique_digits(images, labels):
    unique_digits = np.unique(labels)  # Find the unique digit labels (0-9)
    plt.figure(figsize=(10, 4))
    
    for i, digit in enumerate(unique_digits):
        # Get the index of the first occurrence of each unique digit
        index = np.where(labels == digit)[0][0]
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[index].reshape(28, 28), cmap="binary")
        plt.axis("off")
        plt.title(f"Digit: {digit}")
    
    plt.show()

plot_unique_digits(X, y)

# 3. Apply PCA to retrieve the 1st and 2nd principal component 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Explained variance ratio of 1st compnent: {round(pca.explained_variance_ratio_[0], 4)}")
print(f"Explained variance ratio of 2nd compnent: {round(pca.explained_variance_ratio_[1], 4)}")

# 4. Plot the projections onto the 2D hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap="tab10", s=1) # Scatter plot with digit labels as colors
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.title('Projection of MNIST onto the first two principal components')
plt.show()

# 5. Use IncrementalPCA for reducing dimensionality to 154 components
n_components = 154
inc_pca = IncrementalPCA(n_components=n_components)
X_reduced = inc_pca.fit_transform(X)

print(f"Explained variance ratio for IncrementalPCA: {round(inc_pca.explained_variance_ratio_.sum()), 2}")

# 6. Inverse transform the compressed data
X_compressed_back = inc_pca.inverse_transform(X_reduced)

# Plot original and compressed digits
def plot_comparison(original, compressed, n=10):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Original digits
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="binary")
        plt.axis("off")
        # Compressed digits
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(compressed[i].reshape(28, 28), cmap="binary")
        plt.axis("off")
    plt.show()

# Display first 10 original vs compressed digits
plot_comparison(X[:10], X_compressed_back[:10])


# Exercise 2
# 1. Generate Swiss roll dataset
X, color = make_swiss_roll(n_samples=1000, noise=0.2)

# Plot to display the Swiss roll dataset
plt.figure(figsize=(8, 6))

# Plot the first two features of the Swiss roll dataset
plt.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.get_cmap('Spectral'), edgecolor='k')

plt.title("Swiss Roll Dataset (2D Projection)")
plt.xlabel("Feature 1 (X[:, 0])")
plt.ylabel("Feature 2 (X[:, 2])")
plt.colorbar(label='Color Dimension')
plt.grid(True)
plt.show()

# 3. Apply kPCA with different kernels
kpca_linear = KernelPCA(n_components=2, kernel='linear')
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
kpca_sigmoid = KernelPCA(n_components=2, kernel='sigmoid')

# Transform the dataset using the different kernels
X_kpca_linear = kpca_linear.fit_transform(X)
X_kpca_rbf = kpca_rbf.fit_transform(X)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X)

# 4. Plot the results of Kernel PCA using different kernels
def plot_kpca_results(X_kpca, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap=plt.get_cmap('Spectral'))
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

plot_kpca_results(X_kpca_linear, "kPCA with Linear Kernel")
plot_kpca_results(X_kpca_rbf, "kPCA with RBF Kernel")
plot_kpca_results(X_kpca_sigmoid, "kPCA with Sigmoid Kernel")

# 5. Generate labels for classification
# Create two classes based on the color variable
y = color > np.median(color)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the pipeline with kPCA and Logistic Regression
pipe = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

# Set up the grid search parameters for kPCA
param_grid = [{
    "kpca__kernel": ["rbf", "sigmoid", "linear"],
    "kpca__gamma": np.logspace(-3, 3, 7) # Search over a range of gamma values for non-linear kernels
}]

# Perform GridSearchCV to find the best kernel and gamma value
grid_search = GridSearchCV(pipe, param_grid, cv=3) # 3-fold cross-validation
grid_search.fit(X_train, y_train)

# Print the best parameters and accuracy
print("Best parameters found: ", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("Test accuracy: ", accuracy_score(y_test, y_pred))

# 6. Plot the results with the best found parameters from GridSearchCV
best_kpca = grid_search.best_estimator_.named_steps["kpca"]
X_train_kpca = best_kpca.transform(X_train)
X_test_kpca = best_kpca.transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_kpca[:, 0], X_train_kpca[:, 1], c=y_train, cmap=plt.get_cmap('Spectral'), alpha=0.7, label="Training set")
plt.scatter(X_test_kpca[:, 0], X_test_kpca[:, 1], c=y_test, cmap=plt.get_cmap('Spectral'), edgecolor='k', label="Test set")
plt.title("Best kPCA Projection (based on GridSearchCV)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

