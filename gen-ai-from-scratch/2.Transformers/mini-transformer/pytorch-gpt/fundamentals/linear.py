

import torch
import torch.nn as nn

"""
Create a linear model with x. y=Wx + b
nn.Linear(3,4) produces W -> [4,3] and b -> [3]
given x = [1,2,3] 
y = x.W^T + b -> Matrix mul -> [1,3] * [3,4] = [1,4] + [4] = [1,4] 
"""
class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1,out_features=1)

    def forward(self,x):
        return self.linear(x) # internally does Wx + B. W,B are auto created

model = LinearModel()
x=torch.tensor([[1.0]])
pred = model(x)
target = torch.tensor([[10.0]])
loss = (target-pred) ** 2
loss.backward()
print(f'loss {loss}')
for name,param in model.named_parameters():
    print()
    print(f"Name : {name}")
    print(f"Value : {param}")
    print(f"Grad : {param.grad}")

    """
    Output: 
    ('linear.weight', Parameter containing:
tensor([[-0.0486]], requires_grad=True))
('linear.bias', Parameter containing:
tensor([-0.9150], requires_grad=True))
    """
