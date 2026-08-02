
from ..attention.multi_head_attention import MultiHeadAttention
from ..residual.residual import Residual
from ..layer_norm.layer_nor import LayerNormalization
from ..feed_forward.feed_forward import FeedForward

class TransformerBlock:
    def __init__(self,embedding_dim, num_heads, hidden_dim):
        self.mh = MultiHeadAttention(embedding_dim=embedding_dim,num_heads=num_heads)
        self.ffn = FeedForward(embedding_dim,hidden_dim)

    def forward(self, embeddings):
        # Create multi head attention and get ctx_aware_mh_embeddings
        x = self.mh.forward(embeddings=embeddings)

        # Add input + ctx_aware_mh_embeddings
        residual_outputs = Residual.forward(embeddings,x)
        # Layer Norm : Normalize the residual outputs

        x = LayerNormalization.layer_norm(residual_outputs)
        # Feed forward

        ffn_output = self.ffn.forward(x)
        # Add residual again
        x = Residual.forward(x, ffn_output)
        # Layer Norm : Normalize the residual outputs
        x = LayerNormalization.layer_norm(x)
        return x



        