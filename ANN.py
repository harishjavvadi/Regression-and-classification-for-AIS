import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, r2_score

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
y_progression_cls = data['Curve Length']  # Classification target
y_progression_reg = data['Curve Length']  # Regression target

# Convert to binary classification
y_progression_binary = y_progression_cls.apply(lambda x: 1 if x >= 6 else 0)  # Example threshold (adjust as needed)

# Split the data into train and test sets for both tasks
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_progression_binary, test_size=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_progression_reg, test_size=0.1, random_state=42)

# Generate violin plots before preprocessing
plt.figure(figsize=(12, 6))
sns.violinplot(data=X, palette="viridis", linewidth=2)
plt.title('Data Distribution Before Preprocessing')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Define preprocessing steps for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Combine preprocessing steps for all features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
    ])

# Outlier detection using IsolationForest
outlier_detector = IsolationForest(contamination=0.1)

# Preprocess the data for classification
X_train_preprocessed_cls = preprocessor.fit_transform(X_train_cls)
X_test_preprocessed_cls = preprocessor.transform(X_test_cls)

# Remove outliers from training data for classification
outliers_cls = outlier_detector.fit_predict(X_train_preprocessed_cls)
X_train_preprocessed_cls = X_train_preprocessed_cls[outliers_cls == 1]
y_train_cls = y_train_cls[outliers_cls == 1]

# Preprocess the data for regression
X_train_preprocessed_reg = preprocessor.fit_transform(X_train_reg)
X_test_preprocessed_reg = preprocessor.transform(X_test_reg)

# Remove outliers from training data for regression
outliers_reg = outlier_detector.fit_predict(X_train_preprocessed_reg)
X_train_preprocessed_reg = X_train_preprocessed_reg[outliers_reg == 1]
y_train_reg = y_train_reg[outliers_reg == 1]

# Generate violin plots after preprocessing
plt.figure(figsize=(12, 6))
sns.violinplot(data=pd.DataFrame(X_train_preprocessed_reg, columns=X.select_dtypes(include=['int64', 'float64']).columns), palette="viridis", linewidth=2)
plt.title('Data Distribution After Preprocessing (Regression)')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=pd.DataFrame(X_train_preprocessed_cls, columns=X.select_dtypes(include=['int64', 'float64']).columns), palette="viridis", linewidth=2)
plt.title('Data Distribution After Preprocessing (Classification)')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Define the ANN model for classification
model_cls = Sequential()
model_cls.add(Dense(X_train_preprocessed_cls.shape[1] * 2, activation='relu', input_shape=(X_train_preprocessed_cls.shape[1],)))
model_cls.add(Dense(X_train_preprocessed_cls.shape[1], activation='relu'))
model_cls.add(Dense(1, activation='sigmoid'))  # Binary classification requires a sigmoid activation

# Compile the model with custom learning rate for classification
opt_cls = Adam(learning_rate=0.001)  # Specify learning_rate instead of lr
model_cls.compile(optimizer=opt_cls, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for classification
model_cls.fit(X_train_preprocessed_cls, y_train_cls, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test set for classification
y_pred_proba_cls = model_cls.predict(X_test_preprocessed_cls)
y_pred_cls = (y_pred_proba_cls > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model for classification
accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
precision_cls = precision_score(y_test_cls, y_pred_cls)
recall_cls = recall_score(y_test_cls, y_pred_cls)
f1_cls = f1_score(y_test_cls, y_pred_cls)

print("\nClassification Metrics (Curve Progression Prediction):")
print(f"Accuracy: {accuracy_cls:.2f}")
print(f"Precision: {precision_cls:.2f}")
print(f"Recall: {recall_cls:.2f}")
print(f"F1-score: {f1_cls:.2f}")

# Confusion Matrix for classification
cm_cls = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cls, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Classification)")
plt.show()

# Define the ANN model for regression
model_reg = Sequential()
model_reg.add(Dense(X_train_preprocessed_reg.shape[1] * 2, activation='relu', input_shape=(X_train_preprocessed_reg.shape[1],)))
model_reg.add(Dense(X_train_preprocessed_reg.shape[1], activation='relu'))
model_reg.add(Dense(1))  # No activation for regression

# Compile the model with custom learning rate for regression
opt_reg = Adam(learning_rate=0.001)  # Specify learning_rate instead of lr
model_reg.compile(optimizer=opt_reg, loss='mean_squared_error')  # Using mean squared error for regression

# Train the model for regression
model_reg.fit(X_train_preprocessed_reg, y_train_reg, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test set for regression
y_pred_reg = model_reg.predict(X_test_preprocessed_reg)

# Evaluate the model for regression
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
r2_reg = r2_score(y_test_reg, y_pred_reg)

print("\nRegression Metrics (Curve Length Prediction):")
print(f"Mean Squared Error: {mse_reg:.2f}")
print(f"R-squared: {r2_reg:.2f}")

# Print entire dataset
print("\nEntire Dataset:")
print(data)

# Check for overfitting or underfitting for classification
# Evaluate the model on training data for classification
train_loss_cls, train_accuracy_cls = model_cls.evaluate(X_train_preprocessed_cls, y_train_cls, verbose=0)
print(f"\nTraining Loss (Classification): {train_loss_cls:.4f}")
print(f"Training Accuracy (Classification): {train_accuracy_cls:.4f}")

# Evaluate the model on test data for classification
test_loss_cls, test_accuracy_cls = model_cls.evaluate(X_test_preprocessed_cls, y_test_cls, verbose=0)
print(f"\nTest Loss (Classification): {test_loss_cls:.4f}")
print(f"Test Accuracy (Classification): {test_accuracy_cls:.4f}")

# Check for overfitting or underfitting for regression
# Evaluate the model on training data for regression
train_loss_reg = mean_squared_error(y_train_reg, model_reg.predict(X_train_preprocessed_reg))
print(f"\nTraining Loss (Regression): {train_loss_reg:.4f}")

# Evaluate the model on test data for regression
test_loss_reg = mean_squared_error(y_test_reg, model_reg.predict(X_test_preprocessed_reg))
print(f"\nTest Loss (Regression): {test_loss_reg:.4f}")

# Print final preprocessed data for classification
print("\nFinal Preprocessed Data for Classification:")
print(pd.DataFrame(X_train_preprocessed_cls))

# Print final preprocessed data for regression
print("\nFinal Preprocessed Data for Regression:")
print(pd.DataFrame(X_train_preprocessed_reg))

# Print classification target
print("\nClassification Target:")
print(y_train_cls)

# Print regression target
print("\nRegression Target:")
print(y_train_reg)
