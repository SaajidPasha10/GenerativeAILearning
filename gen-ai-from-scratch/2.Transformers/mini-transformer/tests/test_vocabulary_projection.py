
from src.output.vocabulary_projection import VocabularyProjection
import  numpy as np
def test_vocab_proj_shape():
    embeddings = np.random.randn(5,768)
    vocabProj = VocabularyProjection(embedding_dim=768,vocab_size=50000)
    assert vocabProj.forward(embeddings).shape == (5,50000)
