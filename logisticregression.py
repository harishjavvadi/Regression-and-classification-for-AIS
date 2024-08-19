import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("D:/Study notes/Semester 2/ECE 710 IoT and wearble devices/Project 3/projectdata.csv")

# Specify the columns to include
columns_to_include = ['Sex', 'Brace Treatment',
                     'Height (cm)', 'Max Scoliometer Standing for major curve',
                     'Inclinometer (Kyphosis)(T1/T12)', 'Inclinometer (lordosis)(T12/S2)',
                     'Risser sign', 'Curve direction', 'Curve Number', 'Curve Length',
                     'Curve Location', 'Curve Classfication from TSC', 'AVR Measurement',
                     'No. of exercise sessions']

# Select only the columns to include
X = data[columns_to_include]

# Define the target variables
y_regression = data['Curve Length']
y_classification = data['Curve Length'].apply(lambda x: 1 if x >= 6 else 0)

# Generate violin plots before preprocessing
plt.figure(figsize=(12, 6))
sns.violinplot(data=X, palette="viridis", linewidth=2)
plt.title('Data Distribution Before Preprocessing')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split the data into train and test sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.1, random_state=42)

# Split the data into train and test sets for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.1, random_state=42)

# Handle missing values before preprocessing
imputer = SimpleImputer(strategy='mean')
X_train_reg_numeric = X_train_reg.select_dtypes(include=['int64', 'float64'])
X_train_reg_numeric_imputed = imputer.fit_transform(X_train_reg_numeric)

# Define preprocessing steps for numerical and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data for regression
X_train_preprocessed_reg = preprocessor.fit_transform(X_train_reg)
X_test_preprocessed_reg = preprocessor.transform(X_test_reg)

# Preprocess the data for classification
X_train_preprocessed_cls = preprocessor.fit_transform(X_train_cls)
X_test_preprocessed_cls = preprocessor.transform(X_test_cls)


# Outlier detection for regression
outlier_detector_reg = IsolationForest(contamination=0.1)  # Adjust contamination as needed
outliers_reg = outlier_detector_reg.fit_predict(X_train_preprocessed_reg)
X_train_preprocessed_reg = X_train_preprocessed_reg[outliers_reg == 1]
y_train_reg = y_train_reg[outliers_reg == 1]


# Outlier detection for classification
outlier_detector_cls = IsolationForest(contamination=0.1)  # Adjust contamination as needed
outliers_cls = outlier_detector_cls.fit_predict(X_train_preprocessed_cls)
X_train_preprocessed_cls = X_train_preprocessed_cls[outliers_cls == 1]
y_train_cls = y_train_cls[outliers_cls == 1]


# Define the logistic regression model for regression
model_reg = LogisticRegression()

# Define the logistic regression model for classification
model_cls = LogisticRegression()

# Train the regression model
model_reg.fit(X_train_preprocessed_reg, y_train_reg)

# Train the classification model
model_cls.fit(X_train_preprocessed_cls, y_train_cls)

# Make predictions on the test set for regression
y_pred_reg = model_reg.predict(X_test_preprocessed_reg)

# Make predictions on the test set for classification
y_pred_cls = model_cls.predict(X_test_preprocessed_cls)
y_pred_proba_cls = model_cls.predict_proba(X_test_preprocessed_cls)[:, 1]  # Probability of positive class

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print("\nRegression Metrics (Final Curve Length Prediction):")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

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

# ROC Curve and AUC
roc_auc = roc_auc_score(y_test_cls, y_pred_proba_cls)
fpr, tpr, thresholds = roc_curve(y_test_cls, y_pred_proba_cls)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='b')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
