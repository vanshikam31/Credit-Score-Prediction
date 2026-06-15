import pandas as pd

df = pd.read_csv("Credit Score Classification Dataset.csv")

print(df.head())
print(df.info())

# Features and Target

X = df.drop("Credit Score", axis=1)
y = df["Credit Score"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Classes:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.preprocessing import LabelEncoder

# Copy data
X_train = X_train.copy()
X_test = X_test.copy()

# Store encoders
encoders = {}

# Categorical columns
categorical_cols = [
    "Gender",
    "Education",
    "Marital Status",
    "Home Ownership"
]

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()

    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

    encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()

y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)

print("Encoded Training Data:")
print(X_train.head())

print("\nEncoded Target Classes:")
print(target_encoder.classes_)

from sklearn.ensemble import RandomForestClassifier

# Create model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Predictions
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_
))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print(classification_report(
    y_test,
    lr_pred,
    target_names=target_encoder.classes_
))

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

print(classification_report(
    y_test,
    dt_pred,
    target_names=target_encoder.classes_
))

print("\nModel Comparison")

print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred))

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

import joblib

joblib.dump(rf_model, "credit_model.pkl")

print("Model saved successfully!")

import joblib

# Save feature encoders
joblib.dump(encoders, "encoders.pkl")

# Save target encoder
joblib.dump(target_encoder, "target_encoder.pkl")