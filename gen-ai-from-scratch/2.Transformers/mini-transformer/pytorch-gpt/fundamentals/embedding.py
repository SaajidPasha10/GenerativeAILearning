import torch
import torch.nn as nn

vocab_size = 5 # There can be any num of vocab in vocab dict
embedding_dim = 4 # How many dim each token can have

embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
print(embedding)
print(f"Shape {embedding.weight.shape}") # 5,4

token_ids = torch.tensor([[1,2,3],[2,3,4]])
vectors = embedding(token_ids)

print(f"Token ids \n {token_ids}")
print(f"Embeddings \n {vectors}")
"""
Output
Embeddings 
 tensor([[[-0.2266,  0.2361,  0.0837, -0.9565],
         [ 0.9484,  1.6586, -1.6733, -1.6095],
         [-0.6417, -0.3315,  0.2728, -1.3054]],

        [[ 0.9484,  1.6586, -1.6733, -1.6095],
         [-0.6417, -0.3315,  0.2728, -1.3054],
         [-0.1858,  1.6158, -1.1920,  0.3782]]], grad_fn=<EmbeddingBackward0>)
"""

print(f"Vector shape {vectors.shape}")
# [batch size, seq len, embedding dim] -> [2,3,4]
# [2,3] -> nn.Embedding -> [2,3,4]

print(embedding.weight.requires_grad) # True
print(type(embedding.weight)) #torch.nn.parameter.Parameter
