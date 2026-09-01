"""
Day 7: Week 1 Synthesis — From-Scratch Artificial Neuron (Logistic Perceptron)
Author: Suresh Kondrati
Domain: End-to-End Forward Pass, Binary Cross-Entropy Loss & Gradient Descent
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def compute_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    print("=== WEEK 1 SYNTHESIS: TRAINING A SINGLE NEURON ===")
    X = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
        [0.7, 0.8],
        [0.9, 0.9]
    ])
    y = np.array([[0], [0], [1], [1]])
    m = X.shape[0]

    np.random.seed(42)
    W = np.random.randn(2, 1) * 0.1
    b = 0.0
    lr = 1.0
    epochs = 200

    for epoch in range(1, epochs + 1):
        z = np.dot(X, W) + b
        a = sigmoid(z)
        loss = compute_loss(y, a)

        dz = a - y
        dW = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)

        W -= lr * dW
        b -= lr * db

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | W: [{W[0,0]:.3f}, {W[1,0]:.3f}] | b: {b:.3f}")

    print("\n=== FINAL PREDICTIONS ===")
    predictions = sigmoid(np.dot(X, W) + b)
    for i in range(m):
        verdict = "PASS" if predictions[i, 0] >= 0.5 else "FAIL"
        print(f"Sample {i+1}: Features={X[i]} | Target={y[i,0]} | Prob={predictions[i,0]:.4f} -> {verdict}")

if __name__ == "__main__":
    main()
