"""
logits
   ↓
softmax
   ↓
probabilities
   ↓
look at probability of TARGET
   ↓
-log(probability)
   ↓
loss (More the probability less the loss -log(p))

"""
import numpy as np
class CrossEntropy:
    @staticmethod
    def calculate(logits:np.array,targets) -> float:
        """
        :param logits: list of logits [..]
        :param targets: index of the token with highest probability
        :return: loss for the target probability
        """
        """
        Gpt produces predictions for each token [p1,p2]
        For all tokens [[p1 p2], [p3 p4] [p5 p6]...]
        For each of the predicted values we will have one target prediction
        We take the target prediction value(one with highest prob)
        Then we calculate the loss -> -log(p)
        We take avg of all losses
    Prediction 1 → target 2 → loss₁
    Prediction 2 → target 3 → loss₂
    Prediction 3 → target 4 → loss₃
    Overall loss = average of all prediction losses. 
    This loss tells us how accurately a seq is predicted    
        """
        """
        logits[
            [0, 1, 2],
            [2, 3, 1]
        ]
        logits[0,2]
        logits[1,3]
        logits[2,1]
        """
        print(np.arange(len(targets)))
        target_probabilities = logits[np.arange(len(targets)),targets]
        losses = -np.log(target_probabilities)
        return np.mean(losses)
