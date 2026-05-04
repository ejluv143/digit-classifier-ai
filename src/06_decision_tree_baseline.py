import os
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Data
# -----------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images: 28x28 -> 784
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# -----------------------------
# Train Decision Tree Baseline
# -----------------------------

tree_model = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)

tree_model.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------

y_pred = tree_model.predict(X_test)

# -----------------------------
# Evaluate
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Baseline")
print("----------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Compare with Neural Network
# -----------------------------

print("\nModel Comparison:")
print("Decision Tree Accuracy:   ", round(accuracy, 4))
print("Neural Network Accuracy:  ~0.9669 to 0.9749")
print("\nConclusion:")
print("The neural network performs better because it learns complex patterns from pixel data.")