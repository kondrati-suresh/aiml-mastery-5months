"""
Day 3: Slopes, Derivatives, and Single-Step Weight Update
Author: Suresh Kondrati
Domain: Mathematical Foundations & Calculus for Deep Learning
"""

def main():
    print("=== 1. LOSS AND SLOPE CALCULATION ===")
    # Starting weight and step size (learning rate)
    w = 3.0
    lr = 0.1

    # Loss function L(w) = w^2, derivative dL/dw = 2*w
    loss = w ** 2
    slope = 2 * w

    print(f"Current Weight: {w} | Current Loss: {loss}")
    print(f"Slope (dL/dw): {slope}")

    print("\n=== 2. GRADIENT DESCENT STEP ===")
    # Move opposite to the slope
    w_new = w - (lr * slope)
    new_loss = w_new ** 2

    print(f"Updated Weight: {w_new:.4f} | New Loss: {new_loss:.4f}")
    print(f"Loss Reduction: {loss - new_loss:.4f}")

if __name__ == "__main__":
    main()
