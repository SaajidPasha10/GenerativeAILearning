import torch
import torch.nn as nn

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1,out_features=1)

    def forward(self,x):
        return self.linear(x)

model = SimpleModel()
x = torch.tensor([[2.0]])
target = torch.tensor([[10.0]])
step = 0
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

while step < 100:
    # -------------------------
    # Model
    # -------------------------
    pred = model(x)
    # -------------------------
    # Calculate Loss
    # -------------------------v
    loss = (target-pred) ** 2
    # -------------------------
    # Clear gradients
    # -------------------------
    optimizer.zero_grad()
    # -------------------------
    # Calculate gradients
    # -------------------------
    loss.backward()
    # -------------------------
    # Update weights
    # -------------------------
    optimizer.step()
    # Print for every 10 steps
    if step % 10 == 0:
        print(f"Step {step:3d} | "
            f"Prediction: {pred.item():.4f} | "
            f"Loss: {loss.item():.6f}")

    step += 1

