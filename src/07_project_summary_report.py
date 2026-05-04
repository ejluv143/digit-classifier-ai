import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

report = """
# Digit Classifier AI - Project Summary

## Project Goal
Build a simple multi-class handwritten digit classifier using neural networks and compare it with a decision tree baseline.

## Dataset
MNIST handwritten digit dataset.

- Classes: 10 digits, from 0 to 9
- Image size: 28x28 pixels
- Input features after flattening: 784

## Models Built

### 1. NumPy Forward Propagation
Implemented a neural network forward pass manually using:

- Dense layer
- ReLU activation
- Softmax activation
- Cross-entropy loss
- Accuracy calculation

### 2. TensorFlow Neural Network
Built and trained a neural network using TensorFlow/Keras.

Architecture:

- Input: 784 features
- Dense layer: 128 neurons, ReLU
- Dense layer: 64 neurons, ReLU
- Output layer: 10 neurons, Softmax

Result:

- Test Accuracy: around 97%

### 3. Regularized Neural Network
Improved the model using:

- Dropout
- L2 regularization
- Early stopping

Result:

- Test Accuracy: around 96.69%
- Better generalization and reduced overfitting

### 4. Decision Tree Baseline
Trained a Decision Tree classifier for comparison.

Result:

- Test Accuracy: 88.18%

## Error Analysis

The model made 331 wrong predictions out of 10,000 test images.

Most confused digits:

- Actual 5 predicted as 3
- Confusion count: 25

## Key Learnings

This project demonstrates:

- Forward propagation
- Activation functions
- Softmax for multi-class classification
- Cross-entropy loss
- TensorFlow model training
- Model selection
- Bias and variance diagnosis
- Regularization
- Learning curves
- Error analysis
- Decision tree baseline comparison

## Conclusion

The Neural Network significantly outperformed the Decision Tree baseline.

The Decision Tree achieved 88.18% accuracy, while the Neural Network achieved around 97% accuracy.

This shows that neural networks are better suited for image classification tasks because they can learn complex nonlinear patterns from pixel data.
"""

report_path = os.path.join(OUTPUT_DIR, "project_summary_report.md")

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print(f"Project summary report saved to: {report_path}")