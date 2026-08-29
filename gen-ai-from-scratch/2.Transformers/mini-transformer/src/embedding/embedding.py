
"""
Embeddings are basically the set of numbers that define a token
The embeddings can be mathematically anything. But usually
Vocabsize % embedding dim = 0 is considered.
Because transformers essentially initialize a token with some
embedding numbers
Token -> Token id -> Embedding vector [0.1,...] -> Transformer block
-> Linear Head -> Transform embedding to Vocab size
Ultimately, we need to know which word should come from the vocab next
So we linearly convert the embedding dim to vocab size dim
"""
import numpy as np
class Embeddings:
    def __init__(self,vocab_size,embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        self.weights = np.random.randn(self.vocab_size, self.embedding_dim,)

    def forward(self,token_ids):
        return self.weights[token_ids]

