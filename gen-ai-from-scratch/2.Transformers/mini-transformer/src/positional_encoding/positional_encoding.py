import numpy as np

class PositionalEncoding:
    def __init__(self,max_length,embedding_dim):
        self.max_length = max_length # max num of vector rows for the model
        # GPT has 8192 pos vector rows
        self.embedding_dim = embedding_dim
        self.positional_embeddings = np.random.randn(self.max_length,self.embedding_dim)

    def forward(self,embeddings):
        sequence_length = embeddings.shape[0] # Num of tokens (5,128) -> 5
        return self.positional_embeddings[:sequence_length] + embeddings