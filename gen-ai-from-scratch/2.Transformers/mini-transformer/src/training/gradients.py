
"""
After calculating loss we need to update weights using
Gradient Descent: Calculate how much each weight is responsible for loss
Either the weights should be increased or decreased by how much

x → weight → prediction → loss

Chain rule : dL/dW = dL/dpred  * dpred/dW

Loss = (prediction - expected)^2
dL / dpred = 2 * (prediction - expected)

pred = w * x
dpred / dW = x

"""
"""
                 FORWARD
                   ↓
Input tokens → GPT → Logits
                   ↓
              Cross Entropy
                   ↓
                  Loss
                   ↓
                BACKWARD
                   ↓
               Gradients
                   ↓
               Optimizer
                   ↓
            Updated weights
                   ↓
                 REPEAT
"""
import math as Math
class SimpleGradientDescent:

    @staticmethod
    def calculate(x, w, target):
        # Forward
        prediction = w*x
        # Change in Loss wrt to prediction
        dl_wrt_dpred = 2 * (prediction - target)
        # Change in prediction wrt to change in weights
        dpred_wrt_dw = x
        # Change in loss wrt to change in weights
        dl_wrt_dw = (dl_wrt_dpred * dpred_wrt_dw)
        return dl_wrt_dw


class ChainRuleExample:

    @staticmethod
    def calculate(x):

        # Forward
        a = 2 * x
        y = a ** 2
        loss = y

        # Backward
        d_loss_d_y = 1

        d_y_d_a = 2 * a

        d_a_d_x = 2

        d_loss_d_x = (
            d_loss_d_y
            * d_y_d_a
            * d_a_d_x
        )

        return loss, d_loss_d_x