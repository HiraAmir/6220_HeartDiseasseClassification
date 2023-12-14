import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the processed dataset
file_path = 'data/processed_heart.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# 1. Histograms for each numerical feature
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
num_features = len(numerical_features)

# Determine the layout size
n_rows = int(np.ceil(num_features / 5))  # Adjust the divisor based on how many columns you want per row
data[numerical_features].hist(bins=15, figsize=(15, n_rows * 3), layout=(n_rows, 5))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 3. Boxplots for each numerical feature against the target variable
for feature in numerical_features.drop('target'):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='target', y=feature, data=data)
    plt.title(f'Boxplot of {feature} vs Target')
    plt.show()

# 4. Countplot of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=data)
plt.title('Countplot of Target Variable')
plt.show()
