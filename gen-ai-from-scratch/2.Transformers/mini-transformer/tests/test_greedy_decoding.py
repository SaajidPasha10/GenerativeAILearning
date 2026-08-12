import  numpy as np
from src.decoding.greedy_decoder import GreedyDecoder


def test_greedy_decoder():
    # Should pick arg with highest number
    logits = np.array([0.1,0.2,5.8,1.0])
    token = GreedyDecoder.decode(logits)
    assert  token == 2
