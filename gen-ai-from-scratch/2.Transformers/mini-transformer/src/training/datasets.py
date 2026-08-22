import numpy as np

"""
Given a sequence I love AI. Token ids [1,2,3]
GPT should have
 input   target prediction
 [1]  --> [2]
 [1,2] --> [3]
 
 GPT gives us logits
 
 logits
   ↓
softmax
   ↓
probabilities
   ↓
look at probability of TARGET
   ↓
"""
class NextTokenPrediction:
    @staticmethod
    def create(token_ids:list):
        token_ids = np.array(token_ids)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        return input_ids, target_ids