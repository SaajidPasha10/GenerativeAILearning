
import torch
import torch.nn as nn

"""
Take a token embedding and add positional embedding vector
x.shape -> [2,3,4] -> [batch,seq len, embed dim]
Seq len -> 3 Words per sequence, 2 Sequences
we should add the pos encoding to each seq
[
[T1 + P1], 
[T2 + P2],
[T3 + P3],
...[T6 + P3]]
Batch 1:
token0 + position0
token1 + position1
token2 + position2

Batch 2:
token0 + position0
token1 + position1
token2 + position2"""
class PositionalEncoding(nn.Module):

    def __init__(self,max_seq_len,embedding_dim):
        super().__init__()
        self.positionEmbedding = nn.Embedding(max_seq_len,embedding_dim)
        # Creates a pos emb vector of [P1,P2,...] of size EX: (20,4)

    def forward(self,x):
        # x.shape [2,3,4]
        seq_len = x.shape[1]
        # Create position vector of shape (3,4)
        positions = torch.arange(seq_len,device=x.device)
        position_vectors = self.positionEmbedding(positions)
        return x + position_vectors

token_ids = torch.tensor([[1,2,3],[4,5,6]])
token_embeddings = nn.Embedding(num_embeddings=10,embedding_dim=4)
pos_encoding = PositionalEncoding(max_seq_len=20,embedding_dim=4)

embeddings = token_embeddings(token_ids) # Convert tokens to embeddings of (10,4)
print(f"Embeddings \n {embeddings}")
pos_embeddings = pos_encoding.forward(embeddings)
print(f"Pos Embeddings \n {pos_embeddings}")

"""
Embeddings 
 tensor([[[-0.9015, -0.3290,  1.6858, -0.6083],
         [-0.1238,  0.8619,  0.1162,  1.1569],
         [ 1.1264,  0.6370, -0.3030,  0.4837]],

        [[ 0.3331,  0.8313, -0.2086,  0.6615],
         [ 2.0768, -1.4020, -0.4761, -1.0001],
         [ 1.7867, -0.5380, -2.0883, -0.1817]]], grad_fn=<EmbeddingBackward0>)
Pos Embeddings 
 tensor([[[ 0.1209, -1.3607,  1.9416, -1.5259],
         [ 0.3772,  0.9022, -1.2181, -0.2032],
         [ 0.9449,  0.3425, -0.2797, -0.5504]],

        [[ 1.3555, -0.2005,  0.0472, -0.2561],
         [ 2.5778, -1.3616, -1.8104, -2.3602],
         [ 1.6052, -0.8325, -2.0650, -1.2157]]], grad_fn=<AddBackward0>)

"""