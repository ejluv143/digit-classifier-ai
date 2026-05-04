import os
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Load MNIST Dataset
# -----------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -----------------------------
# Build Neural Network
# -----------------------------

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

# -----------------------------
# Compile Model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# -----------------------------
# Evaluate Model
# -----------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# -----------------------------
# Save Model (robust path)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "digit_classifier_model.keras")
model.save(model_path)

print(f"\nModel saved to {model_path}")