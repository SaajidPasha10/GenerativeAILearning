"""
Suppose our vocabulary contains 50,000 words.

Each input token is first converted into a 768-dimensional embedding vector.

Example:

"The cat sat"

↓

[
 embedding(The),
 embedding(cat),
 embedding(sat)
]

↓

Shape = (sequence_length, 768)

These embeddings pass through multiple Transformer blocks
(Embedding → Positional Encoding → Multi-Head Attention →
Residual Connection → LayerNorm → FFN → ...)

After the final Transformer block, each input token still has
one 768-dimensional hidden representation.

Example:

"The" → [768 values]
"cat" → [768 values]
"sat" → [768 values]

These vectors are not words yet.

To predict the next word, the hidden vector of the current token
is passed through the Vocabulary Projection layer.

Vocabulary Projection converts:

768-dimensional hidden vector
        ↓
50,000 logits (one score for every word in the vocabulary)

Softmax converts these logits into probabilities.

The word with the highest probability is selected as the next token.
"""
import numpy as np
class VocabularyProjection:
    # Embedding shape (5,768) 5 words of 768 nums
    # Vocab size 50,000
    # Vocab projection will be (5,768) @ (768,50000) = (5,50000)
    # Each token will be having 50000 scores of other tokens
    def __init__(self,embedding_dim,vocab_size):
        self.weights = np.random.randn(embedding_dim,vocab_size)

    def forward(self,embeddings):
        return embeddings@self.weights