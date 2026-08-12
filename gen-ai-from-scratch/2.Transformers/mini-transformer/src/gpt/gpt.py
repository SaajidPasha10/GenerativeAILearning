import numpy as np

from ..transformer.transformer_block import TransformerBlock
from ..layer_norm.layer_nor import LayerNormalization
from ..embedding.embedding import Embeddings
from ..positional_encoding.positional_encoding import PositionalEncoding
from ..output.vocabulary_projection import VocabularyProjection
class GPT:
    def __init__(self,num_layers,embedding_dim,hidden_dim,num_heads,vocab_size,max_length):
        self.max_length = max_length
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = Embeddings(self.vocab_size,self.embedding_dim)
        # Positional Encoding
        self.pos_embedding = PositionalEncoding(max_length=self.max_length,embedding_dim=self.embedding_dim)
        # Transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(self.embedding_dim,self.num_heads,self.hidden_dim))
        # Final Normalization
        self.final_ln = LayerNormalization()
        # Vocab Projection : Vector to words
        self.vocab_proj = VocabularyProjection(self.embedding_dim,self.vocab_size)

    def forward(self,token_ids):
        # Embeddings
        x = self.embedding.forward(token_ids)
        x = self.pos_embedding.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_ln.layer_norm(x)
        logits = self.vocab_proj.forward(x)
        return logits

    """
    After the transformer:
    Token         Context it has seen
    I             I 
    Love          I Love
    AI            I Love AI
    So the last token logits will have full context
    """
    @staticmethod
    def next_token_logits(logits):
        return logits[::-1]



