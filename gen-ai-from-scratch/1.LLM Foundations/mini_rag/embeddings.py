from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")



def create_embedding(text:str):
    return model.encode(text)

text = "Fast API is a python library to develop web application"
print(create_embedding(text))