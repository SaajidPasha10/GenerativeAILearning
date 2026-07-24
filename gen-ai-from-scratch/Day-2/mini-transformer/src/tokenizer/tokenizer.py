
class Tokenizer:
    def __init__(self,vocab : dict):
        self.vocab = vocab

    def encode(self,text:str) -> list:
        text = text.lower().split()
        return [self.vocab.get(word,self.vocab["<UNK>"]) for word in text]

    def decode(self,token_ids:list):
        id_to_vocab = {v : k for k,v in self.vocab.items()}
        return " ".join([id_to_vocab.get(token_id,"<UNK>") for token_id in token_ids])