# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("D:/Study notes/Semester 2/ECE 710 IoT and wearble devices/Project 3/projectdata.csv")

# Split the data into features and target variables
columns_to_include = ['Initial Age at initial X-ray time versus the birthdate', 'Sex', 'Brace Treatment', 'Height (cm)', 'Max Scoliometer Standing for major curve', 'Inclinometer (Kyphosis)(T1/T12)', 'Inclinometer (lordosis)(T12/S2)', 'Risser sign', 'Curve direction', 'Curve Number', 'Curve Length', 'Curve Location', 'Curve Classfication from TSC', 'AVR Measurement', 'No. of exercise sessions']
X = data[columns_to_include]
y_cobb = data['Curve Length']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cobb, test_size=0.1, random_state=42)

# Define preprocessing steps for numerical and categorical features
class OutlierHandler:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply your outlier handling strategy here
        # For example, you can use winsorization to cap outliers
        X_clipped = X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95), axis=1)
        return X_clipped

numeric_transformer = Pipeline(steps=[
    ('outlier_handling', OutlierHandler()),  # You need to define OutlierHandler class
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
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

# Create a pipeline for linear regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the preprocessing pipeline
X_train_preprocessed = preprocessor.fit_transform(X_train)
print("Shape of X_train after preprocessing and outlier handling:", X_train_preprocessed.shape)
pipeline.fit(X_train, y_train)

# Get the feature names after one-hot encoding
feature_names = np.concatenate([X_train.select_dtypes(include=['int64', 'float64']).columns,
                                pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out()])

# Get the coefficients and corresponding feature names
coefficients = pipeline.named_steps['regressor'].coef_
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Select important features (features with non-zero coefficients)
important_features = coef_df[coef_df['Coefficient'] != 0]

# Making predictions
X_test_preprocessed = preprocessor.transform(X_test)
print("Shape of X_test after preprocessing and outlier handling:", X_test_preprocessed.shape)
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plot the linear regression line
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Major Curve Angle')
plt.ylabel('Predicted Major Curve Angle')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

# Print the important features
print("Important Features:")
print(important_features)

# Calculate the correlation matrix for numeric columns only
numeric_columns = X.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

numeric_columns_before_preprocessing = data.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(20, 20))
sns.violinplot(data=numeric_columns_before_preprocessing)
plt.title('Before Preprocessing')
plt.xticks(rotation=45)


# Select only the numeric columns after outlier handling
X_train_numeric_outlier_handling = X_train.select_dtypes(include=['int64', 'float64'])
# Visualize data after outlier handling but not after one-hot encoding using a violin plot
plt.figure(figsize=(20, 20))
sns.violinplot(data=X_train_numeric_outlier_handling)
plt.title('After Outlier Handling (Excluding One-Hot Encoding)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

