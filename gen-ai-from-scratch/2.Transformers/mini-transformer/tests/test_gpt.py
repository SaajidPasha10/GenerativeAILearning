import  numpy as np
from src.gpt.gpt import GPT

def test_gpt_shape():
    token_ids = np.array([1,2,3])
    gpt = GPT(num_layers=2,embedding_dim=4,hidden_dim=8,num_heads=2,vocab_size=10,max_length=20)
    assert gpt.forward(token_ids).shape == (3,10)