import torch
import torch.nn as nn
from ..fundamentals.transformer_block import TransformerBlock

class TinyGPT(nn.Module):
    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 max_seq_len,
                 num_layers,
                 hidden_dim,
                 num_heads
                 ):
        super().__init__()

        # 1. Token Embeddings
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        # 2. Positional Embeddings
        self.pos_embeddings = nn.Embedding(max_seq_len,embedding_dim)
        # 3. Transformer Block output shape:
        self.blocks = nn.ModuleList(TransformerBlock(embedding_dim,hidden_dim,num_heads) for _ in range(num_layers))
        # [Batch size, Seq len, Embedding dim]
        # 4. Layer Normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # 5. Language Model [B,S,ED] -> [B,S,VOCAB SIZE]
        self.lm = nn.Linear(in_features=embedding_dim,out_features=vocab_size)

    def forward(self,token_ids):
        batch_size,seq_len = token_ids.shape
        # -------------------------
        # Token embeddings
        # -------------------------
        token_vectors = self.token_embeddings(token_ids)
        # -------------------------
        # Positional embeddings
        # -------------------------
        positions = torch.arange(seq_len,device=token_ids.device)
        position_vectors = self.pos_embeddings(positions)
        x = token_vectors + position_vectors
        # -------------------------
        # Transformer Block
        # -------------------------
        for block in self.blocks:
            x = block(x)
        # -------------------------
        # Final LM
        # -------------------------
        x = self.layer_norm(x)
        # -------------------------
        # Linear Model Head - Logits
        # -------------------------
        logits = self.lm(x)
        return logits

model = TinyGPT(
    vocab_size=10,
    embedding_dim=8,
    max_seq_len=20,
    num_heads=2,
    hidden_dim=32,
    num_layers=2)

token_ids = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
logits = model(token_ids)
print(logits.shape)
print(logits)