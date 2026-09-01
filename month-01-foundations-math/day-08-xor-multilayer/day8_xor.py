"""
Day 8: The XOR Failure of Single Perceptrons & 2-Layer Network Solution
Author: Suresh Kondrati
Domain: Mathematical Foundations & Multi-Layer Perceptrons (MLPs)
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def single_neuron_predict(X, W, b):
    return (sigmoid(np.dot(X, W) + b) >= 0.5).astype(int)

def two_layer_mlp(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    h = sigmoid(z1)
    z2 = np.dot(h, W2) + b2
    out = sigmoid(z2)
    return h, out

def main():
    print("=== 1. THE XOR TRUTH TABLE ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    print("=== 2. SINGLE NEURON ATTEMPT ===")
    W_single = np.array([[0.5], [0.5]])
    b_single = -0.5
    single_preds = single_neuron_predict(X, W_single, b_single)
    print("Single Neuron Accuracy:", f"{np.mean(single_preds == y_xor) * 100:.1f}%")

    print("\n=== 3. 2-LAYER NETWORK (SOLVING XOR) ===")
    W1 = np.array([[20.0, -20.0], [20.0, -20.0]])
    b1 = np.array([-10.0, 30.0])
    W2 = np.array([[20.0], [20.0]])
    b2 = -30.0

    hidden_reps, final_probs = two_layer_mlp(X, W1, b1, W2, b2)
    final_classes = (final_probs >= 0.5).astype(int)

    print("Hidden Layer Transformations:\n", hidden_reps.round(2))
    print("Final Predictions:\n", final_classes.ravel())
    print("2-Layer MLP Accuracy:", f"{np.mean(final_classes == y_xor) * 100:.1f}%")

if __name__ == "__main__":
    main()
