"""
Temperature changes the "sharpness" of probabilities.
Before softmax:

scaled_logits= logits/temperature
if temp is lower, then the model is deterministic.[AI-99%, Cat-1%,Dog-0%]
else model result is more randomized[AI-55%, Cat-30%,Dog-15%]
"""
from ..attention.scaled_attention import ScaledAttention
import numpy as np
class TemperatureSampler:
    """
            logits:
                Raw model scores (vocab_size,)

            temperature:
                Controls randomness

            returns:
                Selected token id
            """

    @staticmethod
    def sample(logits,temperature=0.2):
        # 1. Scaled logits
        scaled_logits = logits/temperature
        # 2. Convert to prob
        prob = ScaledAttention.softmax(scaled_logits)
        # 3. Randomly select token
        # Select random elements of size len(prob)
        next_token = np.random.choice(len(prob),p=prob)

        return next_token