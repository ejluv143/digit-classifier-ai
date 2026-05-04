import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "regularized_digit_classifier.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Model and Data
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

(_, _), (X_test, y_test) = mnist.load_data()

X_test_normalized = X_test / 255.0
X_test_flat = X_test_normalized.reshape(-1, 784)

# -----------------------------
# Choose One Image
# -----------------------------

index = 0

image = X_test[index]
image_input = X_test_flat[index].reshape(1, 784)
true_label = y_test[index]

# -----------------------------
# Predict
# -----------------------------

prediction_probs = model.predict(image_input)
predicted_label = np.argmax(prediction_probs)
confidence = np.max(prediction_probs)

# -----------------------------
# Show Result
# -----------------------------

print("\nSingle Digit Prediction")
print("-----------------------")
print("True Label:", true_label)
print("Predicted Label:", predicted_label)
print("Confidence:", round(confidence * 100, 2), "%")

print("\nClass Probabilities:")
for digit, prob in enumerate(prediction_probs[0]):
    print(f"Digit {digit}: {prob:.4f}")

# -----------------------------
# Save Image Result
# -----------------------------

plt.figure()
plt.imshow(image, cmap="gray")
plt.title(f"True: {true_label} | Predicted: {predicted_label}")
plt.axis("off")

output_path = os.path.join(OUTPUT_DIR, "single_digit_prediction.png")
plt.savefig(output_path)
plt.show()

print(f"\nPrediction image saved to: {output_path}")