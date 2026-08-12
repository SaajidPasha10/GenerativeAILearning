import numpy as np
from ..attention.scaled_attention import ScaledAttention


class TopKSampler:

    @staticmethod
    def sample(logits, k, temperature=0.1):

        # 1. Get indices of top K logits
        top_k_indices = np.argsort(logits)[-k:]

        # 2. Get the actual logits for those tokens
        top_k_logits = logits[top_k_indices]

        # 3. Apply temperature
        scaled_logits = top_k_logits / temperature

        # 4. Convert to probabilities
        probabilities = ScaledAttention.softmax(
            scaled_logits
        )

        # 5. Randomly select one of the K candidates
        selected_index = np.random.choice(
            len(top_k_indices),
            p=probabilities
        )

        # 6. Convert local index back to vocabulary ID
        next_token = top_k_indices[selected_index]

        return next_token