"""
Each head will get the slice of each token
Each head will find the patterns like relationship, grammar, skills etc
head dim(Each head will get a slice of embedding)
 will be embedding dim / num of heads
Ex : Embeddings size - (5,12)
num heads = 4
Head dim = 12/4 = 3
Each head will get 5,3 slice from the embeddings
Once heads output their learned patterns Ex:Verb, pronoun etc
They are concatenated back, but these patterns should be combined
to get the combined meaning hence we use linear transformation
Wo = (embedding_dim,embedding_dim)
Wo * concatenated context aware embeddings
                     Input
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
     Head1           Head2           Head3
       │               │               │
       ▼               ▼               ▼
   Self-Attn      Self-Attn      Self-Attn
       │               │               │
       └───────────────┼───────────────┘
                       ▼
                 Concatenate
                       ▼
               Output Projection
                  (Wₒ)
                       ▼
                 Final Output
"""
from .scaled_attention import ScaledAttention
import numpy as np
class MultiHeadAttention:
    def __init__(self,embedding_dim,num_heads):
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("Embedding dim should be divisible by num of heads!")
        self.heads = []
        self.head_dim = self.embedding_dim // self.num_heads
        # Create Multiple attention head objects
        for _ in range(self.num_heads):
            self.heads.append(ScaledAttention(embedding_dim=self.head_dim))

    # Split the embeddings into num heads slices
    def split_heads(self,embeddings):
        return np.split(embeddings,self.num_heads,axis=1)

    # Forward
    def forward(self,embeddings):
        head_inputs = self.split_heads(embeddings)
        outputs = []
        for head, head_input in zip(self.heads,head_inputs):
            outputs.append(head.forward(head_input))
        concatenated_context_aware_embeddings= np.concatenate(outputs,axis=1)
        WO = np.random.randn(self.embedding_dim,self.embedding_dim)
        # WO will be updated after backpropagation based on which embedding is imp
        return concatenated_context_aware_embeddings @ WO