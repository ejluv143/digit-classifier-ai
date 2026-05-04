import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load + Prepare Data
# -----------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -----------------------------
# Build Regularized Model
# -----------------------------

model = Sequential([
    Input(shape=(784,)),

    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.3),

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
# Early Stopping
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    X_train,
    y_train_cat,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate Model
# -----------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)

print("\nRegularized Model Test Loss:", test_loss)
print("Regularized Model Test Accuracy:", test_accuracy)

# -----------------------------
# Save Model
# -----------------------------

model_path = os.path.join(MODEL_DIR, "regularized_digit_classifier.keras")
model.save(model_path)

print(f"\nModel saved to: {model_path}")

# -----------------------------
# Plot Accuracy
# -----------------------------

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Regularized Model - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(OUTPUT_DIR, "regularized_accuracy_curve.png")
plt.savefig(accuracy_path)
plt.show()

# -----------------------------
# Plot Loss
# -----------------------------

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Regularized Model - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(OUTPUT_DIR, "regularized_loss_curve.png")
plt.savefig(loss_path)
plt.show()

print(f"\nAccuracy curve saved to: {accuracy_path}")
print(f"Loss curve saved to: {loss_path}")

# -----------------------------
# Diagnosis
# -----------------------------

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print("\nBias-Variance Diagnosis:")

if train_acc < 0.90 and val_acc < 0.90:
    print("High bias: model may be underfitting.")
elif train_acc > 0.97 and (train_acc - val_acc) > 0.03:
    print("High variance: model may be overfitting.")
else:
    print("Good fit: regularization reduced overfitting.")