"""
Day 6: Single-Neuron Perceptron with Sigmoid Activation
Author: Suresh Kondrati
Domain: Mathematical Foundations & Scratch Neural Networks
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def perceptron_forward(X, W, b):
    z = np.dot(X, W) + b
    a = sigmoid(z)
    return z, a

def main():
    print("=== 1. TOY PROBLEM: PASS/FAIL PREDICTION ===")
    X = np.array([
        [0.1, 0.2],
        [0.8, 0.9],
        [0.5, 0.4]
    ])

    W = np.array([
        [4.0],
        [2.0]
    ])
    b = -2.5

    z, predictions = perceptron_forward(X, W, b)

    print(f"Inputs X:\n{X}\n")
    print(f"Linear Comb. z:\n{z.round(3)}\n")
    print(f"Sigmoid Activations (Probabilities):\n{predictions.round(3)}\n")

    classes = (predictions >= 0.5).astype(int)
    print(f"Predicted Class (1=Pass, 0=Fail):\n{classes}")

if __name__ == "__main__":
    main()
