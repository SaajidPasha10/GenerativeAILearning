

"""
Expected output
[
"FastAPI is a modern Python web framework.",
"It is used for building APIs quickly.",
"FastAPI supports async programming.",
"AWS Lambda is a serverless compute service.",
"Lambda executes code without managing servers."
]
"""

def split(document):
    document = document.strip().split("\n")
    return document

def chunk_text(document, chunks_size):
    doc_list = split(document)
    doc_list = [d for d in doc_list if len(d) > 1]
    return [doc_list[i : i+chunks_size] for i in range(0,len(doc_list),chunks_size)]

doc = """
            FastAPI is a modern Python web framework.

            It is used for building APIs quickly.

            FastAPI supports async programming.

            AWS Lambda is a serverless compute service.

            Lambda executes code without managing servers.
        """
split(doc)
print(chunk_text(doc,2))
