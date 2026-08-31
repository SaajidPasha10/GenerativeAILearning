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

"""
Embedding(5, 4)
Shape torch.Size([5, 4])
Token ids 
 tensor([[1, 2, 3],
        [2, 3, 4]])
Embeddings 
 tensor([[[-0.7560,  2.1394, -0.7317,  0.1840],
         [-0.1588,  1.3942, -1.1920, -2.4950],
         [-1.1470, -0.7126, -0.1837, -0.9069]],

        [[-0.1588,  1.3942, -1.1920, -2.4950],
         [-1.1470, -0.7126, -0.1837, -0.9069],
         [-0.7363,  0.7585,  2.2258,  0.8307]]], grad_fn=<EmbeddingBackward0>)
Vector shape torch.Size([2, 3, 4])
True
<class 'torch.nn.parameter.Parameter'>

"""