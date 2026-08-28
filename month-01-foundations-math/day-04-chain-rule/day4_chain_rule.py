"""
Day 4: The Chain Rule & 5-Step Gradient Descent Loop
Author: Suresh Kondrati
Domain: Mathematical Foundations & Backpropagation Mechanics
"""

def main():
    print("=== 1. CHAIN RULE COMPUTATION GRAPH ===")
    # Forward: z = 3*w + 1 -> L = z^2
    w = 2.0
    z = 3 * w + 1
    loss = z ** 2

    # Local derivatives
    dL_dz = 2 * z
    dz_dw = 3.0
    dL_dw = dL_dz * dz_dw

    print(f"Forward: w = {w} -> z = {z} -> Loss = {loss}")
    print(f"Local Gradients: dL/dz = {dL_dz}, dz/dw = {dz_dw}")
    print(f"Total Gradient (dL/dw): {dL_dw}")

    print("\n=== 2. 5-STEP OPTIMIZATION LOOP ===")
    w_train = 2.0
    lr = 0.02

    for step in range(1, 6):
        # Forward pass
        z_curr = 3 * w_train + 1
        loss_curr = z_curr ** 2

        # Backward pass (Chain rule)
        dL_dz_curr = 2 * z_curr
        dL_dw_curr = dL_dz_curr * 3.0

        # Optimization step
        w_train = w_train - (lr * dL_dw_curr)

        print(f"Step {step}: Loss = {loss_curr:8.4f} | dL/dw = {dL_dw_curr:7.2f} | New w = {w_train:.4f}")

    print(f"\nTarget weight for zero loss is -0.3333. Final weight: {w_train:.4f}")

if __name__ == "__main__":
    main()
