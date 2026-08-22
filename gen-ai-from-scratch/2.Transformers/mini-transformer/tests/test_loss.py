from src.training.loss import CrossEntropy
import math
import numpy as np
# def test_loss():
#     logits = [0.1,0.5,0.3,0.9]
#     target = np.array([3]) #index of 0.9
#     loss = CrossEntropy.calculate(logits,target)
#     assert round(float(loss),3) == 0.105

def test_loss_2():
    logits = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]) # Logits
    # Targets :
    # 2nd index of first row 3.0, 1st index of second row 5.0
    targets = np.array([2,1])
    loss = CrossEntropy.calculate(logits,targets)
    assert round(float(loss),3) == -1.354
