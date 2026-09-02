"""
Day 10: 2-Layer Neural Network (MLP) from Scratch in NumPy
Author: Suresh Kondrati
Domain: Multi-Layer Forward Pass, ReLU/Sigmoid, Vectorized Backpropagation & Training
"""

import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    print("=== TRAINING A 2-LAYER NEURAL NETWORK ON XOR ===")

    # 1. Dataset: XOR logic gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])
    N = X.shape[0]

    # 2. Architecture Hyperparameters
    input_dim = 2
    hidden_dim = 4      # 4 hidden neurons
    output_dim = 1
    lr = 0.5
    epochs = 1000

    # 3. Weight Initialization (He/Xavier style)
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
    b2 = np.zeros((1, output_dim))

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        # --- FORWARD PASS ---
        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = binary_cross_entropy(y, a2)

        # --- BACKWARD PASS (CHAIN RULE) ---
        # Layer 2 gradients
        dz2 = a2 - y
        dW2 = (1 / N) * np.dot(a1.T, dz2)
        db2 = (1 / N) * np.sum(dz2, axis=0, keepdims=True)

        # Layer 1 gradients (through ReLU)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = (1 / N) * np.dot(X.T, dz1)
        db1 = (1 / N) * np.sum(dz1, axis=0, keepdims=True)

        # --- GRADIENT DESCENT PARAMETER UPDATE ---
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.5f}")

    print("\n=== FINAL MODEL PREDICTIONS ===")
    z1_final = np.dot(X, W1) + b1
    a1_final = relu(z1_final)
    predictions = sigmoid(np.dot(a1_final, W2) + b2)

    for i in range(N):
        pred_class = int(predictions[i, 0] >= 0.5)
        print(f"Input: {X[i]} | True Target: {y[i,0]} | Pred Probability: {predictions[i,0]:.4f} -> Class {pred_class}")

if __name__ == "__main__":
    main()
