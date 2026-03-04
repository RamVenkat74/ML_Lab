# Import Libraries
import pandas as pd
from typing import Any  # Added for type hinting
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# --------------------------------
# Step 1: Load Dataset
# --------------------------------
# Adding ': Any' tells VS Code / Pylance to stop worrying about the return type
data: Any = load_breast_cancer()

X = data.data      # Features (30 columns)
y = data.target    # Target (0 = Malignant, 1 = Benign)

print("Dataset Shape:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# --------------------------------
# Step 2: Convert to DataFrame
# --------------------------------
df = pd.DataFrame(X, columns=data.feature_names)
df["Target"] = y

print("\nSample Data:")
print(df.head())

# --------------------------------
# Step 3: Split Dataset
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Step 4: Train Decision Tree
# --------------------------------
model = DecisionTreeClassifier(
    criterion="entropy",     # Use Information Gain
    max_depth=4,             # Prevent overfitting
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# --------------------------------
# Step 5: Prediction & Evaluation
# --------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------
# Step 6: Visualize Decision Tree
# --------------------------------


plt.figure(figsize=(20, 10), dpi=300)  # Increased DPI for clarity

tree.plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=["Malignant", "Benign"],
    filled=True,
    rounded=True,
    fontsize=10
)

# Save the figure so you can zoom in on the details later
plt.savefig('breast_cancer_decision_tree.png')
print("\nTree visualization saved as 'breast_cancer_decision_tree.png'")
plt.show()
