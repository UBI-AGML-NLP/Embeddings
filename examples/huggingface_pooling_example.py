from embedding import BertHuggingface, BertHuggingfaceMLM

# Variables
NUM_CLASSES = 8  # irrelevant if you dont want to retrain
sentences = [
    "Hello, this is a test for the Huggingface Bert.",
    "Did you know the Huggingface library was named after the smiley?"
]

# embedding
print("embedding with BERT for sequence classification")
for pooling in ['mean', 'cls', 'pooling_layer']:
    print("using pooling strategy %s" % pooling)
    bert = BertHuggingface(NUM_CLASSES, model_name='bert-base-uncased', batch_size=8, pooling=pooling)
    embeddings = bert.embed(sentences)

print("embedding with BERT for MLM")
for pooling in ['mean', 'cls', 'pooling_layer']:  # pooling layer not used in mlm model
    print("using pooling strategy %s" % pooling)
    bert = BertHuggingfaceMLM(model_name='bert-base-uncased', batch_size=8, pooling=pooling)
    embeddings = bert.embed(sentences)

print("embedding with GPT2")
for pooling in ['mean', 'cls', 'pooling_layer']:
    print("using pooling strategy %s" % pooling)
    bert = BertHuggingface(NUM_CLASSES, model_name='gpt2', batch_size=1, pooling=pooling)
    embeddings = bert.embed(sentences)

print("done")

