# Faiss internal implementation
"""
Tasks:

Generate embeddings for all documents.
Generate an embedding for the query.
Compute cosine similarity between the query and each document.
Return the document with the highest similarity.
"""
import numpy as np

documents = [
    "FastAPI builds APIs",

    "Python is a programming language",
"AWS Lambda runs serverless functions"
]

query = "How do I build APIs?"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text:str):
    return model.encode(text)

#Generate embeddings for all documents.
embeddings_list = [create_embedding(doc) for doc in documents]

# Generate an embedding for the query.
query_embedding = create_embedding(query)

# Compute cosine similarity between the query and each document.
import numpy as np
from numpy.linalg import norm
def compute_cosine_similarity(embedding_list  , query_embeddings ) -> list:
    res = np.dot(embedding_list,query_embeddings)/(norm(embedding_list,axis=1) * norm(query_embeddings))
    return res

# Return the document with the highest similarity.

def doc_with_high_similarity(cosine_similarities):
    return documents[np.argmax(cosine_similarities)]

def doc_with_top_k_similarities(cosine_similarities,k):
    arr = np.argsort(cosine_similarities)[::-1]
    return [documents[i] for i in arr]

print(embeddings_list)
print(query_embedding)
cosine_similarities = compute_cosine_similarity(embeddings_list,query_embedding)
print(doc_with_high_similarity(cosine_similarities))
print(doc_with_top_k_similarities(cosine_similarities,2))