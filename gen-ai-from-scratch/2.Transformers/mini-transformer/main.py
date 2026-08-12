import numpy as np
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from src.tokenizer.tokenizer import Tokenizer
from src.embedding.embedding import Embeddings
from src.positional_encoding.positional_encoding import PositionalEncoding
from src.attention.scaled_attention import ScaledAttention
from src.attention.multi_head_attention import MultiHeadAttention
from src.residual.residual import Residual
from src.layer_norm.layer_nor import LayerNormalization
from src.feed_forward.feed_forward import FeedForward
from src.gpt.gpt import GPT
from src.generation.generator import Generator
vocab = {
     "i" : 1,
     "love" : 2,
    "ai" : 3,
    "<UNK>" : 0
}

def e2e_flow():
    tokenizer = Tokenizer(vocab)

    token_ids = tokenizer.encode("i love robotics")
    print(f"Token IDS {token_ids}")

    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text {decoded_text}")

    embeddings = Embeddings(vocab_size=len(vocab), embedding_dim=4)
    embeddings_vector = embeddings.forward(token_ids)
    print(f"Embedding Vectors {embeddings_vector}")
    print(f"Embedding Vectors Shape {embeddings_vector.shape}")  # 3,4

    pos_embedding = PositionalEncoding(max_length=30, embedding_dim=embeddings_vector.shape[1])
    pos_embedding_vectors = pos_embedding.forward(embeddings_vector)
    print(f"After adding Positional Vectors {pos_embedding_vectors}")

    scaled_attention = ScaledAttention(embedding_dim=pos_embedding_vectors.shape[1])
    (q, k, v) = scaled_attention.project(pos_embedding_vectors)
    print(f"Q Vector {q}")
    print(f"Q vector for the first token {q[0]}")
    # print(f"K Vector {k}")
    # print(f"V Vector {v}")
    attention_scores = scaled_attention.attention_scores(q, k)
    print(f"Attention Scores vector {attention_scores}")

    scaled_attention_scores = scaled_attention.scaled_attention_scores(attention_scores,
                                                                       embedding_dim=pos_embedding_vectors.shape[1])
    print(f"Scaled Attention scores {scaled_attention_scores}")

    attention_scores_probabilities = scaled_attention.softmax(scaled_attention_scores, )
    print(f"Probabilities {attention_scores_probabilities}")

    print("******** Single head Attention *****")
    context_aware_embeddings = scaled_attention.forward(pos_embedding_vectors)
    print(f"Context Aware Embeddings {context_aware_embeddings}")

    # Multi head attention
    print("******** Multi head Attention *****")
    multi_head = MultiHeadAttention(embedding_dim=pos_embedding_vectors.shape[1], num_heads=2)
    context_aware_embeddings = multi_head.forward(pos_embedding_vectors)
    print(f'Context Aware Embeddings {context_aware_embeddings}')
    print(f'Context Aware Embeddings Shape {context_aware_embeddings.shape}')

    # Residual Connection
    print("******** Residual Connection *****")
    residual = Residual.forward(pos_embedding_vectors, context_aware_embeddings)
    print(residual)

    # Layer Normalization
    print("******** Layer Normalization *****")
    normalized_ctx_aware_vector = LayerNormalization.layer_norm(residual_output=residual)
    print(normalized_ctx_aware_vector)
    # print(np.allclose(np.average(normalized,axis=1),0))

    # Feed Forward
    print("******** Feed Forward Network *****")
    ffn = FeedForward(embedding_dim=normalized_ctx_aware_vector.shape[1], hidden_dim=16)
    print(ffn.forward(normalized_ctx_aware_vector))

def gpt_flow():
    tokenizer = Tokenizer(vocab)
    token_ids = np.array(tokenizer.encode("i love ai"))
    model = GPT(vocab_size=len(vocab),num_layers=2,embedding_dim=8,max_length=28,num_heads=2,hidden_dim=16)
    logits  = model.forward(token_ids)
    print(f"logits {logits}")
    print(logits.shape)
    prompt = "i love"
    token_ids = np.array(
        tokenizer.encode(prompt)
    )

    generator = Generator(model)
    generated_tokens = generator.generate(token_ids=token_ids,max_new_tokens=5,strategy="temperature",temperature=0.3)
    generated_text = tokenizer.decode(generated_tokens)
    print("Sampling using Temperature")
    print(generated_text)
    print("Sampling using Top K")
    generated_tokens = generator.generate(token_ids=token_ids, max_new_tokens=5, strategy="top_k", k =2,
                                          temperature=0.3)
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)
    print("Sampling using Top P")
    generated_tokens = generator.generate(token_ids=token_ids, max_new_tokens=5, strategy="top_p", p=0.9,
                                          temperature=0.3)
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)

gpt_flow()