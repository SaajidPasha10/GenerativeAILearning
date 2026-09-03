"""
After
 Token ids -> Embeddings -> MHA -> Residual + Layer Norm
 -> FFN -> Residual + LN
 Residual -> Adding original input to the transformed input
 helps in preserving the original data.
 LN -> Layer Normalization normalizes each token's embedding independently by making
its embedding values have a mean of 0 and a standard deviation of
1. This helps stabilize training and keeps activations on a consistent scale.
"""
import torch
import torch.nn as nn
from .feed_forward import FeedForward
from .mha import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim,hidden_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim=hidden_dim
        self.num_heads = num_heads
        # Initialize the LN object
        self.norm1 = nn.LayerNorm(embedding_dim,embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim,embedding_dim)
        self.feed_forward = FeedForward(self.embedding_dim,self.hidden_dim)
        self.mha = MultiHeadAttention(self.num_heads,self.embedding_dim)
    def forward(self,x):

        attention_output = self.mha(x)
        # Pre LN Attention
        x = x + self.norm1(x + attention_output)
        # Pre LN Feed Forward
        x = x + self.feed_forward(self.norm2(x))
        return x

block = TransformerBlock(embedding_dim=8,
    num_heads=2,
    hidden_dim=32)
x = torch.randn(2,3,8)
output = block(x)
print(f"{'*' * 20} Transformer Block {'*' * 20}")
print(output.shape)
print(output)

"""
torch.Size([2, 3, 8])
tensor([[[ 0.3730, -0.0024, -0.0821, -0.3305,  0.0059, -0.1097,  0.2795,
          -0.1337],
         [ 0.2429, -0.1363,  0.0285, -0.2903,  0.1210, -0.0156,  0.2567,
          -0.2068],
         [-0.0317, -0.2096, -0.1072, -0.1509,  0.2259,  0.0371,  0.3313,
          -0.0949]],

        [[ 0.1674, -0.0747, -0.2044,  0.0948,  0.0901, -0.0918,  0.0027,
           0.0160],
         [ 0.0605,  0.0444, -0.1207, -0.1806,  0.1024, -0.2039,  0.3461,
          -0.0483],
         [ 0.0537, -0.1913,  0.1426,  0.1235, -0.1238, -0.0005,  0.0742,
          -0.0784]]], grad_fn=<NativeLayerNormBackward0>)

Process finished with exit code 0
"""
