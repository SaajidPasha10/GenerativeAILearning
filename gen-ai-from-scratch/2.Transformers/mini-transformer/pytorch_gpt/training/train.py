import torch

from ..tiny_gpt.model import TinyGPT
"""
                input_ids
                    ↓
                  GPT
                    ↓
                 logits
                    ↓
              CrossEntropy
                    ↓
                   loss
                    ↓
              loss.backward()
                    ↓
              gradients
                    ↓
            optimizer.step()
                    ↓
             updated weights
                    │
                    └──────→ repeat
"""

# -----------------------------
# Tiny dataset
# -----------------------------

sequences = [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
]


input_ids = torch.tensor([
    sequence[:-1]
    for sequence in sequences
])

targets = torch.tensor([
    sequence[1:]
    for sequence in sequences
])


print("Inputs:")
print(input_ids)

print("Targets:")
print(targets)

model = TinyGPT(
    vocab_size=4,
    embedding_dim=8,
    max_seq_len=10,
    num_heads=2,
    hidden_dim=32,
    num_layers=2
)
logits = model(input_ids)

print(logits.shape)
loss_fn = torch.nn.CrossEntropyLoss()

loss = loss_fn(
    logits.view(-1, logits.size(-1)),
    targets.view(-1)
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

for step in range(1000):

    # Forward
    logits = model(input_ids)

    # Loss
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if step % 100 == 0:
        print(
            f"Step {step} | Loss {loss.item():.4f}"
        )

print("Testing")
print(model(torch.tensor([[1]])))
print(model(torch.tensor([[1,2]])))

model.eval()

prompt = torch.tensor([[1]])

generated = model.generate(
    prompt,
    max_new_tokens=5
)

print(generated)