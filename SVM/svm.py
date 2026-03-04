import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'Offer': [1, 0, 1, 0, 0, 0],
    'Free': [1, 0, 0, 0, 1, 0],
    'Length': [120, 60, 95, 70, 110, 80],
    'Class': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Offer', 'Free', 'Length']]
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create SVM model
model = svm.SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Predicted:", y_pred)
print("Actual:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
