from embedding import BertHuggingface

# Variables
NUM_CLASSES = 8 # irrelevant if you dont want to retrain
sentences = [
    "Hello, this is a test for the Huggingface Bert.",
    "Did you know the Huggingface library was named after the smiley?"
]

# embedding
bert = BertHuggingface(NUM_CLASSES)
embeddings = bert.embed(sentences)

print("Shape of the embeddings:", embeddings.shape)
print("This means there are 2 embeddings each a vector of size 768!")