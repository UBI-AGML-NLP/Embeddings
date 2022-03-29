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

predictions = bert.predict(masked_sentences)

print("Original sentences:", sentences)
print("Predictions:", [x[0]['sequence'] for x in predictions])
