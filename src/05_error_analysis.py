import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "regularized_digit_classifier.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load Data
# -----------------------------

(_, _), (X_test, y_test) = mnist.load_data()

X_test = X_test / 255.0
X_test_flat = X_test.reshape(-1, 784)

# -----------------------------
# Predictions
# -----------------------------

y_pred_probs = model.predict(X_test_flat)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

print(f"\nConfusion matrix saved to: {cm_path}")

# -----------------------------
# Find Wrong Predictions
# -----------------------------

wrong_indices = np.where(y_pred != y_test)[0]

print(f"\nTotal wrong predictions: {len(wrong_indices)}")

# Show some wrong predictions
plt.figure(figsize=(10,5))

for i, idx in enumerate(wrong_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx], cmap="gray")
    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.axis("off")

wrong_path = os.path.join(OUTPUT_DIR, "wrong_predictions.png")
plt.savefig(wrong_path)
plt.show()

print(f"Wrong predictions saved to: {wrong_path}")

# -----------------------------
# Most Confused Digits
# -----------------------------

cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

# find top confusion pair
max_idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
actual_digit, predicted_digit = max_idx

print("\nMost Confused Digits:")
print(f"Actual: {actual_digit} → Predicted: {predicted_digit}")
print(f"Confusion count: {cm[actual_digit, predicted_digit]}")