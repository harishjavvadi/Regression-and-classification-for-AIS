# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("D:/Study notes/Semester 2/ECE 710 IoT and wearble devices/Project 3/projectdata.csv")

# Split the data into features and target variables
columns_to_include = ['Initial Age at initial X-ray time versus the birthdate', 'Sex', 'Brace Treatment', 'Height (cm)', 'Max Scoliometer Standing for major curve', 'Inclinometer (Kyphosis)(T1/T12)', 'Inclinometer (lordosis)(T12/S2)', 'Risser sign', 'Curve direction', 'Curve Number', 'Curve Length', 'Curve Location', 'Curve Classfication from TSC', 'AVR Measurement', 'No. of exercise sessions']

# Select only the columns to include
X = data[columns_to_include]
y_cobb = data['Curve Length']
y_progression = data['Curve Classfication from TSC']

# Split the data into train and test sets
X_train, X_test, y_cobb_train, y_cobb_test = train_test_split(X, y_cobb, test_size=0.1, random_state=42)
X_train, X_test, y_progression_train, y_progression_test = train_test_split(X, y_progression, test_size=0.1, random_state=42)

# Define preprocessing steps for numerical and categorical features
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
        ('num', numeric_transformer, X_train.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X_train.select_dtypes(include=['object']).columns)
    ])

# Create pipelines for regression and classification
regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor())
])

classification_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Train the regression model
regression_pipeline.fit(X_train, y_cobb_train)

# Train the classification model
classification_pipeline.fit(X_train, y_progression_train)

# Make predictions on the test set
y_cobb_pred = regression_pipeline.predict(X_test)
y_progression_pred = classification_pipeline.predict(X_test)

# Evaluate the regression model's performance
mse = mean_squared_error(y_cobb_test, y_cobb_pred)
r2 = r2_score(y_cobb_test, y_cobb_pred)
print(f"Regression Metrics (Final Cobb Angle Prediction):")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Evaluate the classification model's performance
accuracy = accuracy_score(y_progression_test, y_progression_pred)
precision = precision_score(y_progression_test, y_progression_pred, average='weighted')
recall = recall_score(y_progression_test, y_progression_pred, average='micro')
f1 = f1_score(y_progression_test, y_progression_pred, average='micro')
print("\nClassification Metrics (Curve Progression Prediction):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Generate confusion matrix for the classification model
cm = confusion_matrix(y_progression_test, y_progression_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate the correlation matrix for numeric columns only
numeric_columns = X.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()
print("Correlation Matrix:")
print(correlation_matrix)