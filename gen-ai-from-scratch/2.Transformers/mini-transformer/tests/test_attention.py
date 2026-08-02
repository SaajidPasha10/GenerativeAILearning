from src.attention.scaled_attention import ScaledAttention
import numpy as np

def test_q_shape():
    embeddings = np.random.randn(5,128)
    sa = ScaledAttention(embedding_dim = embeddings.shape[1])
    assert sa.project(embeddings)[0].shape == embeddings.shape

def test_k_shape():
    embeddings = np.random.randn(5,128)
    sa = ScaledAttention(embedding_dim = embeddings.shape[1])
    assert sa.project(embeddings)[1].shape == embeddings.shape

def test_v_shape():
    embeddings = np.random.randn(5,128)
    sa = ScaledAttention(embedding_dim = embeddings.shape[1])
    assert sa.project(embeddings)[2].shape== embeddings.shape

# q,k,v should not be identical to input
def test_q_k_v_identity():
    embeddings = np.random.randn(5, 128)
    sa = ScaledAttention(embedding_dim=embeddings.shape[1])
    q,k,v = sa.project(embeddings)
    assert not np.array_equal(q,embeddings)
    assert not np.array_equal(k,embeddings)
    assert not np.array_equal(v,embeddings)

def test_attention_score_shape():
    # Attention score shape for Q*K(transpose)
    # Ex: 3,4 4,3 should be 3,3
    embeddings = np.random.randn(3, 4)
    sa = ScaledAttention(embedding_dim=embeddings.shape[1])
    q,k,v = sa.project(embeddings=embeddings)
    assert sa.attention_scores(q,k).shape == (3,3)

def test_square_matrix():
    # Attention score matrix will be a square matrix rows=columns
    embeddings = np.random.randn(3, 4)
    sa = ScaledAttention(embedding_dim=embeddings.shape[1])
    q, k, v = sa.project(embeddings=embeddings)
    attention_scores = sa.attention_scores(q, k)
    assert attention_scores.shape[0] == attention_scores.shape[1]


def test_scaled_attention_score_shape():
    # Scaling attention scores should not change the shape of the scores
    embeddings = np.random.randn(3, 4)
    sa = ScaledAttention(embedding_dim=embeddings.shape[1])
    q, k, v = sa.project(embeddings=embeddings)
    attention_scores = sa.attention_scores(q, k)
    scaled_scores = sa.scaled_attention_scores(attention_scores,embedding_dim=embeddings.shape[1])
    assert scaled_scores.shape == attention_scores.shape

def test_scaled_attention_score_reduce_values():
    # Scaling attention scores should be reduced
    embeddings = np.random.randn(3, 4)
    sa = ScaledAttention(embedding_dim=embeddings.shape[1])
    q, k, v = sa.project(embeddings=embeddings)
    attention_scores = sa.attention_scores(q, k)
    scaled_scores = sa.scaled_attention_scores(attention_scores,embedding_dim=embeddings.shape[1])
    assert np.max(attention_scores) > np.max(scaled_scores)

def test_prob_equal_to_one():
    scores = np.random.randn(5,5)
    sa = ScaledAttention(embedding_dim=scores.shape[1])
    scores = sa.softmax(scores)
    rows_sum = np.sum(scores,axis=1)
    assert np.allclose(rows_sum,1)


def test_casual_mask():
    scores = np.zeros((4,4))
    masked_scores = ScaledAttention.casual_mask(scores)
    min_inf = -np.inf
    expected = [
        [0,min_inf,min_inf,min_inf],
        [0, 0, min_inf, min_inf],
        [0, 0, 0, min_inf],
        [0, 0, 0, 0],
    ]
    assert np.array_equal(expected,masked_scores)
