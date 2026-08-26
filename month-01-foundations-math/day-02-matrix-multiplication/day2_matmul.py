import time
import numpy as np

def manual_matrix_multiply(A, B):
    """
    Manual matrix multiplication using explicit 3-nested loops.
    Demonstrates the raw scalar calculations before vectorization.
    """
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    
    assert cols_A == rows_B, f"Shape mismatch: {cols_A} != {rows_B}"
    
    # Initialize output matrix of shape (rows_A, cols_B) with zeros
    result = np.zeros((rows_A, cols_B))
    
    for i in range(rows_A):
        for j in range(cols_B):
            dot_sum = 0.0
            for k in range(cols_A):
                dot_sum += A[i, k] * B[k, j]
            result[i, j] = dot_sum
            
    return result

def main():
    print("=== 1. VECTOR DOT PRODUCT & COSINE SIMILARITY ===")
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    
    dot_val = np.dot(v1, v2)
    print(f"Dot Product (v1 · v2): {dot_val}")  # 1*4 + 2*5 + 3*6 = 32.0

    print("\n=== 2. MANUAL MULTIPLICATION VS NUMPY (@ OPERATOR) ===")
    # Batch of 3 samples, 4 features
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [2.0, 0.0, 1.0, -1.0]
    ])
    # Weight matrix connecting 4 features to 2 output neurons
    W = np.array([
        [0.1, 0.2],
        [-0.5, 0.3],
        [0.4, -0.1],
        [1.0, 0.5]
    ])

    manual_out = manual_matrix_multiply(X, W)
    numpy_out = X @ W  # Using Python's matrix multiplication operator (@)

    print("Manual Loop Output:\n", manual_out)
    print("Vectorized Output (X @ W):\n", numpy_out)
    print("Results Match Exactly:", np.allclose(manual_out, numpy_out))

    print("\n=== 3. SPEED BENCHMARK (WHY VECTORIZATION MATTERS) ===")
    # Simulate a realistic neural layer: Batch size 256, 512 inputs, 512 outputs
    large_X = np.random.randn(256, 512)
    large_W = np.random.randn(512, 512)

    # Time NumPy Vectorized Matrix Multiplication
    start_time = time.time()
    _ = np.dot(large_X, large_W)
    numpy_duration = (time.time() - start_time) * 1000  # in ms

    print(f"NumPy BLAS Vectorized Dot Product: {numpy_duration:.4f} ms")
    print("Note: Running manual 3-nested loops on 256x512 would take ~3-5 seconds!")

if __name__ == "__main__":
    main()