from embedding import BertHuggingface

# Variables
NUM_CLASSES = 2
sentences = [
    "I absolutely love Huggingface Bert.",
    "Did you know the Huggingface library was named after the smiley?",
    "UKP Lab also has a sentence transformer.",
    "They can also create meaningful sentence embeddings."
]
labels = [
    0,
    0,
    1,
    1
]

# training & embedding
bert = BertHuggingface(NUM_CLASSES, model_name='bert-base-uncased')

bert.retrain(sentences, labels)

embeddings = bert.embed(sentences)

print("Shape of the embeddings:", embeddings.shape)
print("This means there are 4 embeddings each a vector of size 768!")
