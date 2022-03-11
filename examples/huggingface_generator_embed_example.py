from embedding import BertHuggingface
import numpy as np

# Variables
NUM_CLASSES = 8 # irrelevant if you dont want to retrain

def sententence_generator():

    sentences = [
        "Hello, this is a test for the Huggingface Bert.",
        "Did you know the Huggingface library was named after the smiley?"
    ]
    while True:
        yield sentences

# embedding
bert = BertHuggingface(NUM_CLASSES)
embeddings = []
for y in bert.embed_generator(sententence_generator()):
    embeddings.append(y)
    if len(embeddings) >= 10:
        break
embeddings = np.vstack(embeddings)

print("Shape of the embeddings:", embeddings.shape)
print("This means there are 10 times 2 embeddings each a vector of size 768!")