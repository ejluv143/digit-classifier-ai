import numpy as np

np.random.seed(42)

# -----------------------------
# Activation Functions
# -----------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# -----------------------------
# Loss and Accuracy
# -----------------------------

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    correct_probs = y_pred[np.arange(m), y_true]
    loss = -np.mean(np.log(correct_probs))

    return loss


def accuracy(y_true, y_pred):
    predicted_classes = np.argmax(y_pred, axis=1)
    return np.mean(predicted_classes == y_true)


# -----------------------------
# Forward Propagation
# -----------------------------

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    return A2


# -----------------------------
# Example Data
# -----------------------------

X = np.array([
    [0.2, 0.4, 0.6, 0.8],
    [0.9, 0.1, 0.3, 0.7],
    [0.5, 0.5, 0.5, 0.5],
    [0.1, 0.8, 0.2, 0.6],
    [0.7, 0.3, 0.9, 0.4]
])

# Correct labels
y = np.array([0, 1, 2, 1, 0])

W1 = np.random.randn(4, 6) * 0.01
b1 = np.zeros((1, 6))

W2 = np.random.randn(6, 3) * 0.01
b2 = np.zeros((1, 3))

predictions = forward_propagation(X, W1, b1, W2, b2)

loss = cross_entropy_loss(y, predictions)
acc = accuracy(y, predictions)

print("Softmax Predictions:")
print(predictions)

print("\nPredicted Classes:")
print(np.argmax(predictions, axis=1))

print("\nTrue Labels:")
print(y)

print("\nCross-Entropy Loss:")
print(loss)

print("\nAccuracy:")
print(acc)