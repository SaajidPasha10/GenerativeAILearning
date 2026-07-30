import  numpy as np
from src.feed_forward.feed_forward import FeedForward
# Output and input shape equal
def test_ffn_output_shape():
    embeddings = np.random.randn(5,12)
    ffn = FeedForward(embedding_dim=12,hidden_dim=16)
    output = ffn.forward(embeddings)
    assert output.shape == embeddings.shape

# Output should not be equal to Input
def test_ffn_changes_value():
    embeddings = np.random.randn(5,12)
    ffn = FeedForward(embedding_dim=12,hidden_dim=16)
    output = ffn.forward(embeddings)
    assert not np.array_equal(output, embeddings)