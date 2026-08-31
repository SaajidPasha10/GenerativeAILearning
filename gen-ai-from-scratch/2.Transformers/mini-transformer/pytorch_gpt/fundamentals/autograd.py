"""
Torch autograd basically can do a backpropagation calculation

a = x*2
b = a ** 2
b.backward() --> db/da * da/dx
Where a and b are tensors
"""
import  torch
from torchviz import make_dot

def autograd_ex():
    x = torch.tensor(6.0, requires_grad=True)
    a = x * 2
    b = a ** 2

    b.backward()
    print(f"Gradient Descent of X {x.grad}")

def autograd_with_graph():

    # Data
    x = torch.tensor(2.0,requires_grad=True)
    # Target
    target = 10
    weight = torch.tensor(1.0,requires_grad=True)
    learning_rate = 0.01

    step = 0
    while step < 100:
        pred = x * weight

        # Calculate loss
        loss = (pred-target) ** 2
        #when loss is near to 0
        # item() gives the value from torch.float to float
        if loss.item() < 0.01:
            print("\n🎯 Target reached!")
            print(f"Weight {weight.item():.4f}")
            print(f"Loss {loss.item():.4f}")
            print(f"Prediction {pred.item():.4f}")
            break
        # Visualize computational graph
        if step == 0:
            graph = make_dot(loss,params={"weight" : weight})
            graph.render("computation_graph",format="png")
        # Backward pass
        loss.backward()
        print(
            f"Step {step:2d} | "
            f"Weight {weight.item():.4f} | "
            f"Prediction {pred.item():.4f} | "
            f"Loss {loss.item():.6f} | "
            f"Gradient {weight.grad.item():.4f}"
        )
        # Disable autograd tracking while manually updating the parameter.
        # The weight update should NOT become part of the computational graph.
        with torch.no_grad():
            # Update weight
            weight -= (learning_rate * weight.grad)
        #Clear gradient
        # After each iteration, if we don't clear then torch will
        # add previous step gradients as well
        # old gradient + new gradient
        weight.grad.zero_()
        step += 1
autograd_with_graph()