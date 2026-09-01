"""
Day 9: Modern Activation Functions — ReLU, LeakyReLU, Sigmoid & Tanh
Author: Suresh Kondrati
Domain: Non-Linearity, Vanishing Gradients & Forward/Backward Derivatives
"""

import numpy as np

def relu(z):
    return np.maximum(0, z), (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    val = np.where(z > 0, z, alpha * z)
    grad = np.where(z > 0, 1.0, alpha)
    return val, grad

def sigmoid(z):
    a = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    grad = a * (1.0 - a)
    return a, grad

def tanh(z):
    a = np.tanh(z)
    grad = 1.0 - a ** 2
    return a, grad

def main():
    print("=== ACTIVATION FUNCTIONS & DERIVATIVES COMPARISON ===")
    test_inputs = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    print(f"Input z: {test_inputs}\n")

    activations = [
        ("Sigmoid", sigmoid),
        ("Tanh", tanh),
        ("ReLU", relu),
        ("LeakyReLU", leaky_relu)
    ]

    for name, func in activations:
        val, grad = func(test_inputs)
        print(f"[{name}]")
        print(f"  Forward  a(z): {np.round(val, 4)}")
        print(f"  Gradient a'(z): {np.round(grad, 4)}\n")

if __name__ == "__main__":
    main()
