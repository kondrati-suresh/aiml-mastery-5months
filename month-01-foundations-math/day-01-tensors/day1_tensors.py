import numpy as np

# Day 1: Tensor Hierarchy & Linear Layer Transformation
scalar = np.array(0.045)
vector = np.array([1.2, -0.8, 2.5, 0.4])
matrix_X = np.array([
    [1.2, -0.8,  2.5,  0.4],
    [0.5,  1.1, -1.0,  0.2],
    [2.0,  0.0,  0.7, -1.5]
])

print("Vector Shape:", vector.shape)
print("Batch Matrix Shape:", matrix_X.shape)

# Linear Layer Forward Pass: Z = X · W + b
W = np.random.randn(4, 2)
b = np.zeros((1, 2))
Z = np.dot(matrix_X, W) + b
print("Output Z Shape:", Z.shape)