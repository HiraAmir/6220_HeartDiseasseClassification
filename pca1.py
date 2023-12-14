import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# Load the processed dataset
file_path = 'data/processed_heart.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Applying PCA
pca = PCA(n_components=None)  # None: all components are kept
X_pca = pca.fit_transform(X)

# Creating a DataFrame for the first few principal components and the target
pca_df = pd.DataFrame(X_pca[:, :4], columns=['PC1', 'PC2', 'PC3', 'PC4'])
pca_df['Target'] = y

# 1. Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# 2. 2D Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of PCA')
plt.legend(title='Target')
plt.show()

# 3. 3D Scatter Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', s=50)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D Scatter Plot of PCA')
plt.show()

# 4. Pair Plot of the first few principal components with target
sns.pairplot(pca_df, hue='Target', diag_kind='kde')
plt.suptitle('Pair Plot of First Four Principal Components with Target', y=1.02)
plt.show()

# 5. Heatmap of Component Loadings
plt.figure(figsize=(12, 6))
sns.heatmap(pca.components_.T, cmap='hot', annot=True)
plt.yticks(range(len(X.columns)), X.columns, rotation=0)
plt.xticks(range(X_pca.shape[1]), [f'PC{i+1}' for i in range(X_pca.shape[1])], rotation=45)
plt.title('Heatmap of PCA Component Loadings')
plt.xlabel('Principal Component')
plt.ylabel('Original Features')
plt.show()
