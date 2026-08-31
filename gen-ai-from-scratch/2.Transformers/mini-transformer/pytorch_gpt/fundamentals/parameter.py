
"""
nn.param() : this module is used to create tensors of autograd:True
by default. This module also helps in automatically discovering
the parameters of nn.Module.
Ex : weights = [embedding wts, ffn wts, output wts, attention wts]
manually we don't have to maintain weights.
With nn.param() pytorch will understand weights belonging to nn module
"""
import torch
import torch.nn as nn
import typing
class SimpleModel(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(1.0))

    def forward(self,x):
        return self.weights * x

model = SimpleModel()
print(model.weights.requires_grad) # True: Autograd default enabled
for param in model.parameters():
    print(param)

x = torch.tensor(2.0)
target = torch.tensor(10.0)
model = SimpleModel()
pred = model(x) # internally calls model.forward(x)
print(pred)
loss = (pred-target) ** 2
loss.backward()
print("Prediction:", pred)
print("Loss:", loss)
print("Weight gradient:", model.weights.grad)



