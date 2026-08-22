from src.training.gradients import SimpleGradientDescent

def test_gradients():
    x = 2
    weights = 3
    target = 10
    gradient = SimpleGradientDescent.calculate(x,weights,target)
    assert gradient == -16
