
"""
After we calculate gradients, we need to update the weights
Wnew = Wold - learning_rate  * gradients

"""

class GradientDescent:

    @staticmethod
    def update(w,learning_rate,gradients):
        return w - (learning_rate * gradients)