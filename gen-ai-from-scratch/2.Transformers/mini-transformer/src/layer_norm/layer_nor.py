"""
Layer Normalization normalizes each token's embedding independently by making
its embedding values have a mean of 0 and a standard deviation of
1. This helps stabilize training and keeps activations on a consistent scale.

Token 1: [10, 20, 30, 40]

Mean = 25
Std  ≈ 11.18

Token 1:
[-1.34, -0.45, 0.45, 1.34]

Mean = 0
Std  = 1

        (X - μ)
 LN =  -----------
        √(σ² + ε)
Mean → Center the values around zero.
Variance σ² → Measure how spread out the values are.
Standard Deviation √(σ²) → Normalize the spread.
Epsilon → Prevent division by zero and improve numerical stability.
"""
import numpy as np


class LayerNormalization:
    @staticmethod
    def layer_norm(residual_output):
        # ln = x - mean/std + E
        mean = np.mean(residual_output,axis=1,keepdims=True) # Take each row and find mean
        variance = np.var(residual_output,axis=1,keepdims=True)
        epsilon = 1e-5
        normalized = (residual_output - mean) / (np.sqrt(variance+epsilon))
        return normalized