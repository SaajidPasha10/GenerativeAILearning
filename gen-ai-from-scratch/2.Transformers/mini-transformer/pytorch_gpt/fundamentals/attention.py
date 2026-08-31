
"""
                  Input X
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
       Linear      Linear     Linear
         Wq          Wk         Wv
          ↓          ↓          ↓
          Q          K          V
          │          │
          └────┬─────┘
               ↓
          Q × Kᵀ
               ↓
          / √d_k
               ↓
            Softmax
               ↓
       Attention Weights
               │
               ↓
       Weights × V
               ↓
            Output
"""
import torch
import torch.nn as nn
import math
class Attention(nn.Module):
    """
    nn.Linear(3,4) produces W -> [4,3] and b -> [3]
    given x = [1,2,3]
    y = x.W^T + b -> Matrix mul -> [1,3] * [3,4] = [1,4] + [4] = [1,4]
    """
    def __init__(self,embedding_dim):
        super().__init__()
        # Below are the linear weights
        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,x):
        """
        Q = X Wq + bq
        K = X Wk + bk
        V = X Wv + bv
        :param x:
        :return:
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        print("Q:", Q.shape) #  shapes [[2,3,4]]
        print("K:", K.shape) # shapes [[2,3,4]]
        print("V:", V.shape) # shapes [[2,3,4]]
        # Attention scores : Softmax (Q.K^T /sqrt dk ). V
        # K shape : (2,3,8) -> (Batch, seq len, embedding dim)
        # K.transpose(-2,-1) -> (2,8,3)
        # Q @ K.transpose -> (2,3,8) @ (2,8,3) -> (2,3,3)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(Q.size(-1))
        # Masking
        """
                         I   love   AI
                I         ✓    ✗     ✗
                love      ✓    ✓     ✗
                AI        ✓    ✓     ✓
        """
        seq_len = x.size(1) # x -> [2,3,4] x.size(1) = 3
        # Create a mask of ones of 3,3 shape
        """
       [ [1.0 0.0 0.0],
        [1.0 1.0 0.0],
        [1.0 1.0 1.0]]
        """
        mask = torch.tril(torch.ones(seq_len,seq_len,device=x.device))
        """
        Wherever there are zeroes in the mask, replace with -inf in scores
        """
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # Attention probabilities
        weights = torch.softmax(scores, dim=-1)
        print(weights)
        # Weighted values
        output = weights @ V
        return output

# 2 batches (Sentences), 3 words per sentence seq len
# 4 -> Num of embedding for each word
x = torch.randn(2,3,4)
attention = Attention(embedding_dim=4)
output = attention(x)
print(output)

"""
Q: torch.Size([2, 3, 4])
K: torch.Size([2, 3, 4])
V: torch.Size([2, 3, 4])
Query tensor([[[ 0.3099,  0.6800, -0.2279,  0.0433],
         [-0.9132,  0.1302, -0.2572,  0.0579],
         [ 0.5565, -0.5292, -1.2956, -0.4443]],

        [[ 0.1905,  0.6738, -0.1241,  0.6871],
         [-0.9083, -0.0963, -0.1049, -0.4102],
         [ 0.5941, -0.1651, -0.7039, -0.6000]]], grad_fn=<ViewBackward0>) 
 Key tensor([[[ 0.3099,  0.6800, -0.2279,  0.0433],
         [-0.9132,  0.1302, -0.2572,  0.0579],
         [ 0.5565, -0.5292, -1.2956, -0.4443]],

        [[ 0.1905,  0.6738, -0.1241,  0.6871],
         [-0.9083, -0.0963, -0.1049, -0.4102],
         [ 0.5941, -0.1651, -0.7039, -0.6000]]], grad_fn=<ViewBackward0>) 
 Value tensor([[[ 0.3099,  0.6800, -0.2279,  0.0433],
         [-0.9132,  0.1302, -0.2572,  0.0579],
         [ 0.5565, -0.5292, -1.2956, -0.4443]],

        [[ 0.1905,  0.6738, -0.1241,  0.6871],
         [-0.9083, -0.0963, -0.1049, -0.4102],
         [ 0.5941, -0.1651, -0.7039, -0.6000]]], grad_fn=<ViewBackward0>)
Output:
tensor([[[-0.2742,  0.0996,  0.0752,  0.1377],
         [-1.0378, -0.0193,  0.5992,  0.7544],
         [-0.7965,  0.7343, -0.0687,  0.8932]],

        [[-0.0674,  0.1735, -0.0449,  0.1194],
         [-0.2810,  0.4133, -0.1122,  0.3678],
         [-2.3193,  1.6197, -0.4088,  2.0755]]], grad_fn=<UnsafeViewBackward0>)

"""