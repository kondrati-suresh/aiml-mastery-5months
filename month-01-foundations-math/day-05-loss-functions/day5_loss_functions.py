"""
Day 5: Loss Functions — Mean Squared Error (MSE) vs. Binary Cross-Entropy (BCE)
Author: Suresh Kondrati
Domain: Mathematical Foundations & Objective Functions for Deep Learning
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Computes MSE: (1/N) * sum((y_true - y_pred)^2)
    Used primarily for regression tasks.
    """
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Computes BCE: -mean(y_true * ln(y_pred) + (1 - y_true) * ln(1 - y_pred))
    Used for binary classification probabilities in range (0, 1).
    epsilon prevents log(0) numerical undefined errors.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    print("=== 1. REGRESSION SCENARIO (HOUSING PRICES) ===")
    actual_prices = np.array([300.0, 450.0, 200.0])
    predicted_prices = np.array([310.0, 420.0, 210.0])

    mse_loss = mean_squared_error(actual_prices, predicted_prices)
    print(f"Target Prices:     {actual_prices}")
    print(f"Predicted Prices:  {predicted_prices}")
    print(f"Mean Squared Error (MSE): {mse_loss:.2f}\n")

    print("=== 2. CLASSIFICATION SCENARIO (FRAUD DETECTION) ===")
    true_labels = np.array([1, 0, 1, 0])
    
    good_preds = np.array([0.95, 0.05, 0.88, 0.02])
    bad_preds  = np.array([0.10, 0.85, 0.15, 0.90])

    good_bce = binary_cross_entropy(true_labels, good_preds)
    bad_bce  = binary_cross_entropy(true_labels, bad_preds)

    print(f"True Labels:               {true_labels}")
    print(f"Good Predictions:          {good_preds} -> BCE Loss: {good_bce:.4f}")
    print(f"Confidently Wrong Preds:   {bad_preds}  -> BCE Loss: {bad_bce:.4f}")
    print(f"Penalty Multiplier:        {bad_bce / good_bce:.1f}x higher error!")

if __name__ == "__main__":
    main()
