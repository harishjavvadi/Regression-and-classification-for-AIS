import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("D:/Study notes/Semester 2/ECE 710 IoT and wearble devices/Project 3/projectdata.csv")

# Specify the columns to include
columns_to_include = ['Initial Age at initial X-ray time versus the birthdate', 'Sex', 'Brace Treatment',
                     'Height (cm)', 'Max Scoliometer Standing for major curve',
                     'Inclinometer (Kyphosis)(T1/T12)', 'Inclinometer (lordosis)(T12/S2)',
                     'Risser sign', 'Curve direction', 'Curve Number', 'Curve Length',
                     'Curve Location', 'Curve Classfication from TSC', 'AVR Measurement',
                     'No. of exercise sessions']

# Select only the columns to include
X = data[columns_to_include]
y_regression = data['Curve Length']
y_classification = data['Curve Length'].apply(lambda x: 1 if x >= 6 else 0)

# Split the data into train and test sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.5, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.5, random_state=42)

# Define preprocessing steps for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant")),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Outlier detection using IsolationForest
outlier_detector = IsolationForest(contamination=0.1)

# Combine preprocessing steps for all features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
    ])

# Preprocess the data for regression
X_train_preprocessed_reg = preprocessor.fit_transform(X_train_reg)
X_test_preprocessed_reg = preprocessor.transform(X_test_reg)

# Outlier detection and removal for regression
outliers_reg = outlier_detector.fit_predict(X_train_preprocessed_reg)
X_train_preprocessed_reg = X_train_preprocessed_reg[outliers_reg == 1]
y_train_reg = y_train_reg[outliers_reg == 1]

# Preprocess the data for classification
X_train_preprocessed_cls = preprocessor.fit_transform(X_train_cls)
X_test_preprocessed_cls = preprocessor.transform(X_test_cls)

# Outlier detection and removal for classification
outliers_cls = outlier_detector.fit_predict(X_train_preprocessed_cls)
X_train_preprocessed_cls = X_train_preprocessed_cls[outliers_cls == 1]
y_train_cls = y_train_cls[outliers_cls == 1]

# Define the models
model_reg = RandomForestRegressor(random_state=42)
model_cls = RandomForestClassifier(random_state=42)

# Train the models
model_reg.fit(X_train_preprocessed_reg, y_train_reg)
model_cls.fit(X_train_preprocessed_cls, y_train_cls)

# Make predictions on the test set for regression
y_pred_reg = model_reg.predict(X_test_preprocessed_reg)

# Make predictions on the test set for classification
y_pred_cls = model_cls.predict(X_test_preprocessed_cls)
y_pred_proba_cls = model_cls.predict_proba(X_test_preprocessed_cls)[:, 1]  # Probability of positive class

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print("\nRegression Metrics (Final Curve Length Prediction):")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Evaluate the classification model
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls)
recall = recall_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)

print("\nClassification Metrics (Curve Progression Prediction):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Number of columns in preprocessed data for regression
num_columns_reg = X_train_preprocessed_reg.shape[1]
print(f"Number of columns in preprocessed data for regression: {num_columns_reg}")

# Number of columns in preprocessed data for classification
num_columns_cls = X_train_preprocessed_cls.shape[1]
print(f"Number of columns in preprocessed data for classification: {num_columns_cls}")

# Display preprocessed data for regression
print("\nPreprocessed Data for Regression:")
print(pd.DataFrame(X_train_preprocessed_reg))

# Display preprocessed data for classification
print("\nPreprocessed Data for Classification:")
print(pd.DataFrame(X_train_preprocessed_cls))

# Number of columns in preprocessed data for regression
num_columns_reg = X_train_preprocessed_reg.shape[1]
print(f"Number of columns in preprocessed data for regression: {num_columns_reg}")

# Number of columns in preprocessed data for classification
num_columns_cls = X_train_preprocessed_cls.shape[1]
print(f"Number of columns in preprocessed data for classification: {num_columns_cls}")

# Display preprocessed data for regression
print("\nPreprocessed Data for Regression:")
print(pd.DataFrame(X_train_preprocessed_reg))

# Display preprocessed data for classification
print("\nPreprocessed Data for Classification:")
print(pd.DataFrame(X_train_preprocessed_cls))


# Feature Importance Plots for Regression
plt.figure(figsize=(10, 6))
num_features = min(5,len(X.columns), len(model_reg.feature_importances_))
plt.bar(range(num_features), model_cls.feature_importances_[:num_features])
plt.xticks(range(num_features), X.columns[:num_features], rotation=90)
plt.title('Feature Importance for Regression')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Feature Importance Plots for Classification
plt.figure(figsize=(10, 6))
num_features = min(len(X.columns), len(model_cls.feature_importances_))
plt.bar(range(num_features), model_cls.feature_importances_[:num_features])
plt.xticks(range(num_features), X.columns[:num_features], rotation=90)
plt.title('Feature Importance for Classification')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
