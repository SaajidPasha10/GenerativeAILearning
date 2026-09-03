import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from pytorch_gpt.tiny_gpt.training.dataset import GPTDataset
from pytorch_gpt.tiny_gpt.model import TinyGPT
from pytorch_gpt.tiny_gpt.tokenizer import CharacterTokenizer


# ============================================================
# DATA
# ============================================================

def load_text(file_path):
    """
    Load the training text from disk.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find training file: {path.resolve()}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def create_tokenizer(text):
    """
    Build a character-level tokenizer.
    """

    return CharacterTokenizer(text)


def tokenize_text(text, tokenizer):
    """
    Convert text into integer token IDs.
    """

    return tokenizer.encode(text)


def create_data_loaders(
    token_ids,
    block_size,
    batch_size
):
    """
    Split tokens into training and validation data.

    90% -> training
    10% -> validation
    """

    split_index = int(
        len(token_ids) * 0.9
    )

    train_tokens = token_ids[
        :split_index
    ]

    val_tokens = token_ids[
        split_index:
    ]

    # Safety check
    if len(train_tokens) <= block_size:
        raise ValueError(
            "Training data is too small for the selected block_size."
        )

    if len(val_tokens) <= block_size:
        raise ValueError(
            "Validation data is too small for the selected block_size."
        )

    train_dataset = GPTDataset(
        train_tokens,
        block_size
    )

    val_dataset = GPTDataset(
        val_tokens,
        block_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


# ============================================================
# MODEL
# ============================================================

def create_model(
    vocab_size,
    block_size,
    embedding_dim=128,
    num_layers=4,
    hidden_dim=512,
    num_heads=4
):
    """
    Create TinyGPT.
    """

    model = TinyGPT(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        max_seq_len=block_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
    learning_rate=3e-4
):
    """
    Create AdamW optimizer.
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Prefer Apple Silicon GPU (MPS),
    then CUDA,
    then CPU.
    """

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    loss_fn,
    device,
    max_batches=10
):
    """
    Evaluate model without calculating gradients.
    """

    model.eval()

    total_loss = 0.0
    batches = 0

    for step, (x, y) in enumerate(data_loader):

        if step >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(x)

        # logits:
        #
        # [B, S, V]
        #
        # B = batch
        # S = sequence length
        # V = vocabulary size
        #
        # CrossEntropyLoss wants:
        #
        # predictions:
        # [B*S, V]
        #
        # targets:
        # [B*S]

        loss = loss_fn(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1)
        )

        total_loss += loss.item()

        batches += 1

    if batches == 0:
        return float("nan")

    return total_loss / batches


# ============================================================
# TRAINING
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    max_steps=2000,
    checkpoint_path="checkpoint.pt"
):
    """
    Train TinyGPT for a fixed number of steps.

    Each step:

        Input tokens
              ↓
            GPT
              ↓
        logits [B,S,V]
              ↓
        CrossEntropyLoss
              ↓
         loss.backward()
              ↓
        optimizer.step()
    """

    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    print("\nStarting training...")
    print("Device:", device)
    print("Max steps:", max_steps)

    step = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    while step < max_steps:

        model.train()

        for x, y in train_loader:

            if step >= max_steps:
                break

            # ------------------------------------------------
            # Move batch to device
            # ------------------------------------------------

            x = x.to(device)
            y = y.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(x)

            # Expected:
            #
            # x:
            # [B, S]
            #
            # logits:
            # [B, S, V]
            #
            # y:
            # [B, S]

            loss = loss_fn(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                y.reshape(-1)
            )

            # ------------------------------------------------
            # Clear old gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update parameters
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if step % 100 == 0:

                print(
                    f"Step {step:4d}/{max_steps} "
                    f"| Train Loss: {loss.item():.4f}"
                )

            step += 1

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nStarting validation...")

    val_loss = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device
    )

    print(
        f"\nTraining completed"
        f"\nSteps: {step}"
        f"\nFinal Train Loss: {loss.item():.4f}"
        f"\nValidation Loss: {val_loss:.4f}"
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        loss=loss,
        checkpoint_path=checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    checkpoint_path
):
    """
    Save model + optimizer state.
    """

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "step":
                step,

            "loss":
                loss.item()
        },
        checkpoint_path
    )


# ============================================================
# DEBUG / INSPECTION
# ============================================================

def inspect_tokenization(
    text,
    tokenizer,
    token_ids
):
    """
    Inspect tokenizer output.
    """

    print("\n==============================")
    print("TOKENIZATION INSPECTION")
    print("==============================")

    print(
        "Characters:",
        len(text)
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    print(
        "Number of token IDs:",
        len(token_ids)
    )

    print(
        "\nFirst 100 token IDs:"
    )

    print(
        token_ids[:100]
    )

    print(
        "\nDecoded text:"
    )

    print(
        tokenizer.decode(
            token_ids[:100]
        )
    )


def inspect_batch(
    train_loader
):
    """
    Inspect one batch.
    """

    x, y = next(
        iter(train_loader)
    )

    print("\n==============================")
    print("BATCH INSPECTION")
    print("==============================")

    print(
        "x shape:",
        x.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "\nx[0]:"
    )

    print(
        x[0]
    )

    print(
        "\ny[0]:"
    )

    print(
        y[0]
    )

    print(
        "\nExpected relationship:"
    )

    print(
        "y[0][:-1] should equal x[0][1:]"
    )

    return x, y


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CONFIGURATION
    # ========================================================

    BASE_DIR = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    text_file = (
        BASE_DIR
        / "data"
        / "shakespeare.txt"
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    block_size = 128

    batch_size = 8

    learning_rate = 3e-4

    # Fixed number of optimization steps.
    #
    # This is easier to control than epochs
    # when we want approximately 5-10 minutes.
    max_steps = 2000

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    embedding_dim = 128

    num_layers = 4

    hidden_dim = 512

    num_heads = 4

    # --------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------

    max_text_chars = 200_000

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print("==============================")
    print("TinyGPT Training")
    print("==============================")

    print(
        "Device:",
        device
    )

    # ========================================================
    # LOAD TEXT
    # ========================================================

    print(
        "\nLoading:",
        text_file
    )

    text = load_text(
        text_file
    )

    # Limit corpus size so our experiment
    # finishes quickly.

    text = text[
        :max_text_chars
    ]

    print(
        "Training characters:",
        len(text)
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = create_tokenizer(
        text
    )

    # ========================================================
    # TOKENIZE
    # ========================================================

    token_ids = tokenize_text(
        text,
        tokenizer
    )

    # ========================================================
    # INSPECT TOKENIZATION
    # ========================================================

    inspect_tokenization(
        text,
        tokenizer,
        token_ids
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader, val_loader = (
        create_data_loaders(
            token_ids=token_ids,
            block_size=block_size,
            batch_size=batch_size
        )
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ========================================================
    # INSPECT BATCH
    # ========================================================

    inspect_batch(
        train_loader
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n==============================")
    print("Creating TinyGPT")
    print("==============================")

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
    model = model.to(device)
    # Count parameters

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    print(
        "Embedding dimension:",
        embedding_dim
    )

    print(
        "Number of layers:",
        num_layers
    )

    print(
        "Hidden dimension:",
        hidden_dim
    )

    print(
        "Number of heads:",
        num_heads
    )

    print(
        "Block size:",
        block_size
    )

    print(
        "Trainable parameters:",
        f"{parameter_count:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        checkpoint_path="checkpoint.pt"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()