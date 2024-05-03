from embedding import BertHuggingface
from datasets import load_dataset
import numpy as np

dataset = load_dataset('sst2')
print("loaded glue sst2")
text_train = dataset['train']['sentence'][:800]
y_train = dataset['train']['label'][:800]
text_test = dataset['test']['sentence'][:100]
y_test = dataset['test']['label'][:100]

# Variables
NUM_CLASSES = np.max(y_train)+1
print(NUM_CLASSES)

# training & embedding
bert = BertHuggingface(NUM_CLASSES, model_name='bert-base-uncased')

bert.retrain(text_train, y_train)

pred = bert.predict(text_test)
y_pred = np.argmax(pred, axis=1)
print(y_pred.shape)
print(pred.shape)

embeddings = bert.embed(text_train[:10])

print("Shape of the embeddings:", embeddings.shape)
print("This means there are 4 embeddings each a vector of size 768!")
