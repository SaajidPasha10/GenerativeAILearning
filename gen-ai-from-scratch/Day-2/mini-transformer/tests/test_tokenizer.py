from src.tokenizer.tokenizer import Tokenizer

def test_encode_known_words():

    vocab = {
        "i": 1,
        "love": 2,
        "ai": 3,
        "<UNK>": 0
    }
    tokenizer = Tokenizer(vocab)
    ids = tokenizer.encode("i love ai")
    assert ids == [1,2,3]

def test_encode_unknown_words():

    vocab = {
        "i": 1,
        "love": 2,
        "ai": 3,
        "<UNK>": 0
    }
    tokenizer = Tokenizer(vocab)
    ids = tokenizer.encode("i love robots")
    assert ids == [1,2,0]

def test_decode_known_idx():
    vocab = {
        "i": 1,
        "love": 2,
        "ai": 3,
        "<UNK>": 0
    }
    tokenizer = Tokenizer(vocab)

    tokens = tokenizer.decode([1,2,3])
    assert tokens == "i love ai"