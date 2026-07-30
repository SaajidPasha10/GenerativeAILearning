from src.residual.residual import Residual
import numpy as np
def test_residual():
    x = np.random.randn(5,12)
    y = np.random.randn(5,12)
    assert Residual.forward(x,y).shape == x.shape
