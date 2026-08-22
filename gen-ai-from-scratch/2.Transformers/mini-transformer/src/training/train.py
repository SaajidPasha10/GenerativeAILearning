from gradients import SimpleGradientDescent
from optimizer import GradientDescent
from loss import CrossEntropy


def train(epochs, targets, model, input_ids):
    for epoch in range(epochs):
        # 1. Forward
        logits = model.forward(input_ids)

        # 2. Calculate loss
        loss = CrossEntropy.calculate(
            logits,
            targets
        )

        # 3. Calculate gradients
        gradients = SimpleGradientDescent.calculate(loss)

        # 4. Update parameters
        GradientDescent.update(gradients)