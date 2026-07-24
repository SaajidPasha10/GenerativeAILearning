import numpy as np
from src.positional_encoding.positional_encoding import PositionalEncoding

# Output shape should be equal
def test_output_shape():
    embeddings = np.random.randn(5,128)
    pe = PositionalEncoding(max_length=30,embedding_dim=128)
    assert pe.forward(embeddings).shape == embeddings.shape

# Output should not be equal to input

def test_output_not_equal_input():
    embeddings = np.random.randn(5, 128)
    pe = PositionalEncoding(max_length=30, embedding_dim=128)
    assert not np.array_equal(pe.forward(embeddings),embeddings)

# Output is correct, if we add zeroes to pos encodings it
# should be equal to pos embeddings

def test_output_correct():
    embeddings = np.zeros(shape=(5,128))
    pe = PositionalEncoding(max_length=30, embedding_dim=128)
    assert np.array_equal(pe.forward(embeddings),pe.positional_embeddings[:5])