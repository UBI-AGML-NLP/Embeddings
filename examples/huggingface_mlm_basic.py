from embedding import BertHuggingfaceMLM

# Variables
sentences = [
    "I absolutely love Huggingface Bert.",
    "Did you know the Huggingface library was named after the smiley?",
    "UKP Lab also has a sentence transformer.",
    "They can also create meaningful sentence embeddings."
]

masked_sentences = [
    "I absolutely [MASK] Huggingface Bert.",
    "Did you [MASK] the Huggingface library was named after the smiley?",
    "UKP Lab also has a [MASK] transformer.",
    "They can also [MASK] meaningful sentence embeddings."
]

# training & embedding
bert = BertHuggingfaceMLM()

bert.retrain(masked_sentences, sentences, epochs=1)
 
# Even more basic: 
bert.lazy_retrain(sentences, epochs=1)


embeddings = bert.embed(sentences)

print("Shape of the embeddings:", embeddings.shape)
print("This means there are 4 embeddings each a vector of size 768!")
