from src.attention.softmax import softmax
import numpy as np
def test_prob_equal_to_one():
    scores = np.random.randn(5,5)
    scores = softmax(scores)
    rows_sum = np.sum(scores,axis=1)
    assert np.allclose(rows_sum,1)