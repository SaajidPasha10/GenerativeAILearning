import  numpy as np
from src.attention.multi_head_attention import MultiHeadAttention
def test_multi_head_output_shape():
    embeddings = np.random.randn(5,12)
    mha = MultiHeadAttention(embedding_dim=embeddings.shape[1],num_heads=3)
    assert mha.forward(embeddings).shape == embeddings.shape