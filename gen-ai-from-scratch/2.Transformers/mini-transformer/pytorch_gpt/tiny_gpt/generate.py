import torch
from pathlib import Path

from pytorch_gpt.tiny_gpt.model import TinyGPT
from pytorch_gpt.tiny_gpt.tokenizer import CharacterTokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_text(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path.resolve()}"
        )

    return path.read_text(encoding="utf-8")


def load_model(checkpoint_path, vocab_size, block_size, device):
    model = TinyGPT(
        embedding_dim=128,
        vocab_size=vocab_size,
        max_seq_len=block_size,
        num_layers=4,
        hidden_dim=512,
        num_heads=4
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def main():
    BASE_DIR = Path(__file__).resolve().parent

    text_file = BASE_DIR / "data" / "shakespeare.txt"
    checkpoint_path = BASE_DIR / "training" / "checkpoint.pt"
    block_size = 128
    device = get_device()
    print("Device:", device)

    # -------------------------
    # Load training corpus
    # -------------------------

    text = load_text(text_file)
    text = text[:200_000]

    tokenizer = CharacterTokenizer(text)

    print("Vocabulary size:", tokenizer.vocab_size)

    # -------------------------
    # Load trained model
    # -------------------------

    model = load_model(
        checkpoint_path=checkpoint_path,
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        device=device
    )

    # -------------------------
    # Prompt
    # -------------------------

    prompt = "ROMEO:"

    prompt_ids = tokenizer.encode(prompt)

    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    print("\nPrompt:")
    print(prompt)

    # -------------------------
    # Generate
    # -------------------------

    with torch.no_grad():
        generated_ids = model.generate(
            token_ids,
            max_new_tokens=300,
            temperature=1.0,
            top_p=0.9,
            top_k=20

        )

    # -------------------------
    # Decode
    # -------------------------

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    print("\n==============================")
    print("GENERATED TEXT")
    print("==============================")
    print(generated_text)


if __name__ == "__main__":
    main()
