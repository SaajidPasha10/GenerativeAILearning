def readme_txt():
    """
    After layer normalization, we can forward the normalized data
    to FFN,
    Attention : Which tokens should i pay attention to?
    FFN : What should i do with the information i collected?
    FFN is applied to every token independently
    Token 1 → FFN → transformed token 1
    ...

    "Apple released a new iPhone"
    Attention learns relationships:
    Apple ↔ released
    iPhone ↔ new
    Apple ↔ iPhone

    FFN learns transformations:
    It may activate features like:
    Apple → company
    iPhone → product
    released → launch event

    Analogy:
    Attention is like a team discussion (tokens talk to each other)
    FFN is like each person thinking and updating their own knowledge after the discussion.
              Tokens
                 |
                 ↓
        +----------------+
        |    Attention   |
        |                |
        | Mix information|
        +----------------+
                 |
                 ↓
        +----------------+
        |      FFN       |
        |                |
        | Transform and  |
        | add knowledge  |
        +----------------+
                 |
                 ↓
          Better token
          representation
    """
    pass


# Input --> Linear (Multiply by W input) -->GELU --> Linear
# In ReLU max(x,0) negative values are ignored
# In GeLU negative values are not ignored Gelu - xΦ(x) Φ is the Gaussian CDF
import  numpy as np
class FeedForward:

    def __init__(self,embedding_dim,hidden_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.w1 = np.random.randn(embedding_dim,hidden_dim)
        self.w2 = np.random.randn(hidden_dim,embedding_dim)

    # Activation function
    @staticmethod
    def gelu(x):
        return 0.5 * x * (
                1 + np.tanh(
            np.sqrt(2 / np.pi) *
            (
                    x +
                    0.044715 * np.power(x, 3)
            )
        )
        )

    def forward(self,normalized_ctx_aware_embeddings):
        # Ex  input (5,12) w1 (12,768) = (5,768)
        input_layer = np.dot(normalized_ctx_aware_embeddings,self.w1)
        hidden_layer = self.gelu(input_layer)
        # Hidden layer (5,768) w2 (768,12) = (5,12)
        output = np.dot(hidden_layer,self.w2)
        return output

"""
Input (5,4)

↓

W1

↓

(5,16)

↓

GELU

↓

(5,16)

↓

W2

↓

(5,4)
"""