from embedding import BertHuggingface

# Variables
NUM_CLASSES = 8 # irrelevant if you dont want to retrain
sentences = [
    "Hello, this is a test for the Huggingface Bert.",
    "Did you know the Huggingface library was named after the smiley?",
    "a", 
    "b", 
    "c"
]

# embedding
bert = BertHuggingface(NUM_CLASSES, model_name='bert-base-uncased', batch_size=8)
embeddings = bert.embed(sentences)

print("Shape of the embeddings:", embeddings.shape)
print(f"This means there are {embeddings.shape[0]} embeddings each a vector of size {embeddings.shape[1]}!")

tmp_file = "/tmp/testmodel"
bert.save(tmp_file)
bert.load(tmp_file)

embeddings = bert.embed(sentences)

print("Shape of the embeddings:", embeddings.shape)