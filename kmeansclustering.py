# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv("D:/Study notes/Semester 2/ECE 710 IoT and wearble devices/Project 3/projectdata.csv")

# Split the data into features and target variables
columns_to_include = ['Initial Age at initial X-ray time versus the birthdate', 'Sex', 'Brace Treatment', 'Height (cm)', 'Max Scoliometer Standing for major curve', 'Inclinometer (Kyphosis)(T1/T12)', 'Inclinometer (lordosis)(T12/S2)', 'Risser sign', 'Curve direction', 'Curve Number', 'Curve Length', 'Curve Location', 'Curve Classfication from TSC', 'AVR Measurement', 'No. of exercise sessions']

# Select only the columns to include
X = data[columns_to_include]

# Preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps for all features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
    ])

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Find optimal number of clusters using silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_processed)
    silhouette_avg = silhouette_score(X_processed, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 because range starts from 2

# Perform K-Means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_processed)

# Calculate silhouette score and inertia for the optimal number of clusters
silhouette_avg = silhouette_score(X_processed, cluster_labels)
inertia = kmeans.inertia_

print(f"Optimal number of clusters: {optimal_n_clusters}")
print(f"Silhouette score: {silhouette_avg}")
print(f"Inertia: {inertia}")

# Verify the shape of X_processed
print("Shape of X_processed:", X_processed.shape)


# Reduce dimensionality of data using PCA
pca = PCA(n_components=2, svd_solver='arpack')
X_pca = pca.fit_transform(X_processed)

# Plot scatter plot of the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


