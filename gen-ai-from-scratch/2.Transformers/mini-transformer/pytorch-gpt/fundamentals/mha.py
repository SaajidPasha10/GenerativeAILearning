
import torch
import torch.nn as nn
import math
"""
Goal of MHA is to create Context Aware tokens.
Each token after mha will have a representation that is 
aware of other tokens. 
Ex: 
After attention:

"it"
   ↓
[representation of "it"
 + information about animal
 + information about tired
 + information from surrounding context]
 
 It produces one contextualized vector per token.
 Input:
Token1 → Token2 → Token3 → ... → Token8
  ↓        ↓        ↓             ↓
 MHA      MHA      MHA           MHA
  ↓        ↓        ↓             ↓
Vec1     Vec2     Vec3    ...    Vec8

        MHA OUTPUT
        (8 × embedding_dim)
        
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
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,embedding_dim):
        """
        :param num_heads:
        :param embedding_dim: Ex : (8)
        Query,Key,Value Embedding = W(8*8) matrix + B(8) matrix
        """
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        assert embedding_dim % self.num_heads == 0
        self.head_dim = embedding_dim // self.num_heads
        # Create 3 matrices:
        # query = Wq + bq
        # key = Wk + bk
        # value = Wv + bv
        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,x):
        batch_size,seq_len, _ = x.shape
        # --------------------------------
        # Q K V
        # --------------------------------
        # query = X. Wq + bq
        # key =X. Wk + bk
        # value = X. Wv + bv
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # --------------------------------
        # Split into heads
        # --------------------------------
        # x.shape (2,3,8)
        # Q = (2,3,2,4)
        Q = Q.view(batch_size,seq_len,self.num_heads,self.head_dim)
        K = K.view(batch_size,seq_len,self.num_heads,self.head_dim)
        V = V.view(batch_size,seq_len,self.num_heads,self.head_dim)
        # [batch, seq, heads, head_dim] ->[2,3,2,4]
        #          ↓
        # [batch, heads, seq, head_dim] -> [2,2,3,4]
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        # --------------------------------
        # Attention
        # --------------------------------
        # K -> [2,2,3,4] K.T [2,2,4,3]
        # Q.KT -> [2,2,3,4] @ [2,2,4,3] -> [2,2,3,3]
        scores = Q @ K.transpose(-2,-1)
        # Self.head_dim -> 8 // 2 = 4
        scores = scores/math.sqrt(self.head_dim)
        # --------------------------------
        # Causal mask - (seq_len,seq_len)
        # --------------------------------
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # Softmax : scores shape : (2,2,3,3)
        # dim = -1 -> Column wise
        weights = torch.softmax(scores,dim=-1)
        # --------------------------------
        # Weighted V -> (2,2,3,3) @ (2,2,3,4) - (2,2,3,4)
        # --------------------------------
        output = weights @ V
        # [batch, heads, seq, head_dim]
        #          ↓
        # [batch, seq, heads, head_dim]
        output = output.transpose(1,2)
        # Combine heads
        # [batch, seq, heads, head_dim] + [batch, seq, heads, head_dim]...
        # Becomes [batch, seq, head_dim]
        output  = output.contiguous().view(batch_size,seq_len,self.embedding_dim)
        return output


x = torch.randn(2,3,8)
mha = MultiHeadAttention(num_heads=2,embedding_dim=8)
output = mha(x)
print(output.shape)
print(output)
"""
torch.Size([2, 3, 8])
tensor([[[ 0.1258, -0.2657, -0.4855,  0.9779,  0.1154, -0.9951, -0.5009,
          -0.6541],
         [ 0.1683, -0.2462, -0.0463,  1.0059, -0.0249, -0.3425, -0.2279,
          -0.6679],
         [ 0.0493, -0.5050,  0.0647,  0.7313, -0.1043, -0.2525, -0.1335,
          -0.4963]],

        [[-0.8283, -0.2442,  0.3938, -0.0311,  0.4522,  0.7248, -0.2998,
           0.1902],
         [-0.4209, -0.4541,  0.3844,  0.0736,  0.3019,  0.3985, -0.1499,
           0.0178],
         [-0.3129, -0.3755,  0.2008,  0.1601,  0.1787,  0.1646, -0.2093,
          -0.0639]]], grad_fn=<ViewBackward0>)
"""




