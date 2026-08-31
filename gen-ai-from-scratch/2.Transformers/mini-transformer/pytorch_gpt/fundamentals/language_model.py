import torch
import torch.nn as nn

"""
Embeddings are basically the set of numbers that define a token
The embeddings can be mathematically anything. But usually
Vocabsize % embedding dim = 0 is considered.
Because transformers essentially initialize a token with some
embedding numbers
Token -> Token id -> Embedding vector [0.1,...] -> Transformer block
-> Linear Head -> Transform embedding to Vocab size
Ultimately, we need to know which word should come from the vocab next
So we linearly convert the embedding dim to vocab size dim
"I love"
   ↓
Token Embedding
   ↓
Transformer
   ↓
LM Head
   ↓
[score for "I",
 score for "love",
 score for "AI",
 score for "<UNK>",
 ...]
"""

"""
Batch Size - Num of sentences transformer process parallely(Input, target)
Input -> I love Target : AI
Seq len - Num of words in a sentence
Embedding dim - Vector that defines a token

Transformer output
      [Batch size, Seq len, Embedding dim]
          ↓
       LM Head
      Linear(Embedding dim,Vocab size)
          ↓
       logits
      [Batch, Seq len, Vocab size]
"""

class LanguageModel(nn.Module):
    def __init__(self,embedding_dim,vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.linear = nn.Linear(in_features=self.embedding_dim,out_features=self.vocab_size)

    def forward(self,x):
        return self.linear(x)

x = torch.randn(2,5,8)
lm = LanguageModel(embedding_dim=8,vocab_size=10)
logits = lm(x)
print(logits.shape)
print(logits)

"""
Output 
torch.Size([2, 5, 10])
tensor([[[ 0.3945,  0.6027, -0.4724, -0.8976,  0.6163, -0.0312,  0.2425,
          -0.4818,  0.8681,  0.5283],
         [ 0.7451, -0.6749, -1.3233, -0.2080, -0.2996,  0.2647, -0.5123,
          -0.5529,  1.0417,  0.1627],
         [-0.1891,  0.3758, -0.4076,  0.1687, -1.1337,  1.1792, -0.1239,
           0.2262,  0.4174,  0.4524],
         [ 0.5543,  0.6577,  0.1718,  0.5204, -1.0779,  1.6429, -0.3756,
           0.0778,  0.2107, -0.1015],
         [ 0.9261, -0.8864,  0.4232, -0.0437,  1.2321, -1.0593, -0.6712,
          -0.5361,  0.4622, -0.1001]],

        [[ 1.0871,  0.5179,  0.5203,  1.4057, -0.7463,  1.8174, -0.1708,
           0.1239, -0.5709, -0.9159],
         [ 0.0145, -1.4950, -0.1270,  0.5201,  0.3097, -1.5297, -1.0287,
          -0.3500,  0.2446, -0.2440],
         [-0.3169,  1.0708,  1.2867,  0.4139, -0.9686,  0.6947, -0.0236,
           0.4075, -0.4950, -0.0778],
         [ 0.0036, -0.0666, -0.0807, -0.1649, -0.3587, -0.2809, -0.7955,
          -0.3383,  0.6443,  0.2542],
         [ 0.1146,  0.7148, -0.3111,  0.0828, -0.8640,  0.7972, -0.5450,
          -0.3245,  0.3544,  0.2004]]], grad_fn=<ViewBackward0>)

Process finished with exit code 0

"""
