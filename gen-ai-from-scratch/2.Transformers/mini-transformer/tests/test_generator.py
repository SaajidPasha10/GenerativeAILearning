
import numpy as np
from src.generation.generator import Generator


class DummyModel:
    @staticmethod
    def forward(token_ids):
        # GPT forward will generate vocab proj token
        # Shape will be num_tokens, vocab size
        vocab_size = 5
        return np.array([[1,2,3,4,5] for _ in token_ids])

def test_generator():
    model = DummyModel()
    output = Generator(model).generate([1,2],max_new_tokens=3)
    # Generator will add 3 more tokens to the original tokens
    assert len(output) == 5