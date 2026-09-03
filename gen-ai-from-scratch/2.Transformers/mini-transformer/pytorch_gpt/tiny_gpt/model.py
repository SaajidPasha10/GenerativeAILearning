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

    @torch.no_grad()
    def generate(
            self,
            token_ids,
            max_new_tokens,
            temperature=1.0,
            top_k=None,
            top_p=None
    ):
        for _ in range(max_new_tokens):

            # -------------------------------------------------
            # 1. Keep only the context the model can handle
            # -------------------------------------------------
            context = token_ids[
                      :, -self.pos_embeddings.num_embeddings:
                      ]

            # -------------------------------------------------
            # 2. Get logits
            # Shape: [B, context_length, vocab_size]
            # -------------------------------------------------
            logits = self(context)

            # Only care about the prediction for the
            # final token position
            next_token_logits = logits[:, -1, :]

            # -------------------------------------------------
            # 3. Temperature
            # -------------------------------------------------
            if temperature <= 0:
                raise ValueError("temperature must be > 0")

            next_token_logits = (
                    next_token_logits / temperature
            )

            # -------------------------------------------------
            # 4. Top-K filtering
            # -------------------------------------------------
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))

                values, indices = torch.topk(
                    next_token_logits,
                    k=top_k,
                    dim=-1
                )

                filtered_logits = torch.full_like(
                    next_token_logits,
                    float("-inf")
                )

                filtered_logits.scatter_(
                    dim=-1,
                    index=indices,
                    src=values
                )

                next_token_logits = filtered_logits

            # -------------------------------------------------
            # 5. Convert logits → probabilities
            # -------------------------------------------------
            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            # -------------------------------------------------
            # 6. Top-P / Nucleus filtering
            # -------------------------------------------------
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probabilities,
                    descending=True,
                    dim=-1
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1
                )

                # Remove tokens after cumulative probability
                # exceeds top_p
                sorted_indices_to_remove = (
                        cumulative_probs > top_p
                )

                # Keep the first token that crosses the
                # threshold
                sorted_indices_to_remove[:, 1:] = (
                    sorted_indices_to_remove[:, :-1].clone()
                )

                sorted_indices_to_remove[:, 0] = False

                # Convert sorted indices back to original
                # vocabulary positions
                indices_to_remove = torch.zeros_like(
                    probabilities,
                    dtype=torch.bool
                )

                indices_to_remove.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )

                probabilities = probabilities.masked_fill(
                    indices_to_remove,
                    0.0
                )

                # Renormalize
                probabilities = (
                        probabilities
                        / probabilities.sum(
                    dim=-1,
                    keepdim=True
                )
                )

            # -------------------------------------------------
            # 7. Sample next token
            # -------------------------------------------------
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # -------------------------------------------------
            # 8. Append to the FULL generated sequence
            # -------------------------------------------------
            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1
            )

        return token_ids

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
