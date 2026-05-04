import os
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN expects: samples, height, width, channels
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -----------------------------
# Build CNN + Augmentation Model
# -----------------------------

model = Sequential([
    Input(shape=(28, 28, 1)),

    # Data augmentation
    RandomRotation(0.08),
    RandomTranslation(0.10, 0.10),
    RandomZoom(0.10),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------

history = model.fit(
    X_train,
    y_train_cat,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate
# -----------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)

print("\nCNN Augmented Model Test Loss:", test_loss)
print("CNN Augmented Model Test Accuracy:", test_accuracy)

# -----------------------------
# Save
# -----------------------------

model_path = os.path.join(MODEL_DIR, "cnn_augmented_digit_classifier.keras")
model.save(model_path)

print(f"\nModel saved to: {model_path}")