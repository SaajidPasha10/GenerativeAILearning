
import  numpy as np
from src.layer_norm.layer_nor import LayerNormalization

def test_layer_norm_shape_unchanged():
    embeddings = np.random.randn(5,12)
    output = LayerNormalization.layer_norm(embeddings)
    assert output.shape == embeddings.shape

def test_layer_norm_mean_zero():
    embeddings = np.random.randn(5,12)
    output = LayerNormalization.layer_norm(embeddings)
    means = np.mean(output)
    assert np.allclose(means,0)

def test_layer_norm_std_1():
    embeddings = np.random.randn(5,12)
    output = LayerNormalization.layer_norm(embeddings)
    std = np.std(output,axis=1)
    assert np.allclose(std,1,atol=1e-5)