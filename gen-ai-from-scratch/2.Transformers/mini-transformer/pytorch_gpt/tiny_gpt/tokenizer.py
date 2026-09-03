
class CharacterTokenizer:

    def __init__(self, text):

        # Get every unique character
        chars = sorted(set(text))

        # Special token for unknown characters
        self.stoi = {
            "<UNK>": 0
        }

        # Character -> ID
        for i, char in enumerate(chars, start=1):
            self.stoi[char] = i

        # ID -> Character
        self.itos = {
            i: char
            for char, i in self.stoi.items()
        }

    def encode(self, text):

        return [
            self.stoi.get(char, self.stoi["<UNK>"])
            for char in text
        ]

    def decode(self, token_ids):

        return "".join(
            self.itos.get(token_id, "")
            for token_id in token_ids
        )

    @property
    def vocab_size(self):

        return len(self.stoi)

