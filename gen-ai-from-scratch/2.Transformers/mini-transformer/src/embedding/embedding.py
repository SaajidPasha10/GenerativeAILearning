
import numpy as np
class Embeddings:
    def __init__(self,vocab_size,embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        self.weights = np.random.randn(self.vocab_size, self.embedding_dim,)

    def forward(self,token_ids):
        return self.weights[token_ids]

