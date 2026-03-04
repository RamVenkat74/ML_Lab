# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt  # Added for visualization
from typing import Any           # Added to fix Pylance warnings
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# =================================================
# PART 1: RANDOM FOREST CLASSIFICATION
# =================================================

print("===== RANDOM FOREST CLASSIFICATION =====")

# Load Classification Dataset - Added ': Any' to fix the squiggly lines
data_cls: Any = load_breast_cancer()

X_cls = data_cls.data
y_cls = data_cls.target

# Split Dataset
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls,
    test_size=0.2,
    random_state=42
)

# Train Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    max_depth=6,
    random_state=42
)

rf_classifier.fit(X_train_c, y_train_c)

# Prediction
y_pred_c = rf_classifier.predict(X_test_c)

# Evaluation
acc = accuracy_score(y_test_c, y_pred_c)
print(f"\nClassification Accuracy: {acc:.4f}")

# =================================================
# PART 2: RANDOM FOREST REGRESSION
# =================================================

print("\n===== RANDOM FOREST REGRESSION =====")

# Load Regression Dataset - Added ': Any' to fix the squiggly lines
data_reg: Any = fetch_california_housing()

X_reg = data_reg.data
y_reg = data_reg.target

# Split Dataset
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)

# Train Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

rf_regressor.fit(X_train_r, y_train_r)

# Prediction
y_pred_r = rf_regressor.predict(X_test_r)

# Evaluation
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# =================================================
# NEW: FEATURE IMPORTANCE VISUALIZATION
# =================================================
# This shows which features "contributed" most to the Random Forest's decisions
importances = rf_classifier.feature_importances_
feat_importances = pd.Series(importances, index=data_cls.feature_names)


plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 Most Important Features (Classification)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# =================================================
# END OF PROGRAM
# =================================================
