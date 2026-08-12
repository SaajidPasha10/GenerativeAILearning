"""
Top-P means:

Keep adding the most probable tokens until we've reached at least P.
Logits
   ↓
Temperature
   ↓
Softmax
   ↓
Probabilities
   ↓
Sort probabilities
   ↓
Cumulative probabilities
   ↓
Keep until cumulative >= P
   ↓
Sample
"""

import  numpy as np

from ..attention.scaled_attention import ScaledAttention

class TopPSampling:
    @staticmethod
    def sample(logits,p,temperature):

        # Scale logits
        scaled_logits = logits/temperature
        # Probabilities probabilities:
        # [0.50, 0.25, 0.10, 0.05, 0.04]
        probabilities = ScaledAttention.softmax(scaled_logits)
        # Sort in descending order [0.50, 0.25, 0.10, 0.05, 0.04]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_prob = probabilities[sorted_indices]
        # Cumulative sum of P=0.9 .[0.50, 0.75, 0.85, 0.90, 0.94]
        cum_prob = np.cumsum(sorted_prob)
        # Keep prob that are >= P
        cutoff_index = np.searchsorted(cum_prob,p)
        # Include the token that crosses threshold
        cutoff_index += 1
        top_p_indices = sorted_indices[:cutoff_index]
        top_p_prob = probabilities[top_p_indices]
        # Re normalize prob
        top_p_prob = (top_p_prob/np.sum(top_p_prob))
        # Sample
        selected_index = np.random.choice(len(top_p_indices),p=top_p_prob)
        # Map back to tokens
        next_token = top_p_indices[selected_index]
        return int(next_token)
