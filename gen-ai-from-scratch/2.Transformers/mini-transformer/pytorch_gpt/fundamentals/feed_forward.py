import torch.nn as nn

"""
FFN comes after MHA.
FFN is used to project the embedding dim of a word to a higher
dimension to transform it so that it can capture more rich features.
A word : Good when projected to higher dim can include [nice, polite,verb..]  

embedding_dim -> 8
hidden_dim -> 32
output_dim -> 8
8 -> 32 -> 8
"""
class FeedForward(nn.Module):
    def __init__(self,embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embedding_dim)
        )

    def forward(self,x):
        return self.network(x)
