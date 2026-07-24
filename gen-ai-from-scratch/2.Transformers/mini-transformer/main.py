
from src.tokenizer.tokenizer import Tokenizer
from src.embedding.embedding import Embeddings
from src.positional_encoding.positional_encoding import PositionalEncoding
from src.attention.scaled_attention import ScaledAttention

import numpy as np
vocab = {
     "i" : 1,
     "love" : 2,
    "ai" : 3,
    "<UNK>" : 0
}

tokenizer = Tokenizer(vocab)

token_ids = tokenizer.encode("i love robotics")
print(f"Token IDS {token_ids}")

decoded_text = tokenizer.decode(token_ids)
print(f"Decoded text {decoded_text}")

embeddings = Embeddings(vocab_size=len(vocab),embedding_dim=4)
embeddings_vector = embeddings.forward(token_ids)
print(f"Embedding Vectors {embeddings_vector}")
print(f"Embedding Vectors Shape {embeddings_vector.shape}") # 3,4


pos_embedding = PositionalEncoding(max_length=30,embedding_dim=embeddings_vector.shape[1])
pos_embedding_vectors = pos_embedding.forward(embeddings_vector)
print(f"After adding Positional Vectors {pos_embedding_vectors}")

scaled_attention = ScaledAttention(embedding_dim=pos_embedding_vectors.shape[1])
(q,k,v) = scaled_attention.project(pos_embedding_vectors)
print(f"Q Vector {q}")
print(f"Q vector for the first token {q[0]}")
# print(f"K Vector {k}")
# print(f"V Vector {v}")
attention_scores = scaled_attention.attention_scores(q,k)
print(f"Attention Scores vector {attention_scores}")

scaled_attention_scores = scaled_attention.scaled_attention_scores(attention_scores,embedding_dim=pos_embedding_vectors.shape[1])
print(f"Scaled Attention scores {scaled_attention_scores}")

attention_scores_probabilities = scaled_attention.softmax(scaled_attention_scores,)
print(f"Probabilities {attention_scores_probabilities}")

final_attention_scores = scaled_attention.forward(pos_embedding_vectors)
print(f"Final Attention Scores {final_attention_scores}")