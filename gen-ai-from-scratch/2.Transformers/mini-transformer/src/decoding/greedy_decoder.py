from ..attention.scaled_attention import ScaledAttention
import numpy as np

class GreedyDecoder:

    @staticmethod
    def decode(logits):
        """
                logits: shape (vocab_size,)
                returns: token id with highest probability
                """
        probabilities = ScaledAttention.softmax(logits)
        next_token = np.argmax(probabilities)
        return next_token