@torch.no_grad()
def evaluate(model, data_loader, loss_fn):

    model.eval()

    total_loss = 0

    for x, y in data_loader:

        logits = model(x)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()

    return total_loss / len(data_loader)