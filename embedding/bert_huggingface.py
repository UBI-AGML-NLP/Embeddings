from .embedder import Embedder
import numpy as np
import math

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

import torch
from torch.nn import functional as F
import sklearn

def first_zero(arr):
    mask = arr==0
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)

def compute_cosine_similarities(X, x):
    return sklearn.metrics.pairwise.cosine_similarity(X, np.array([x]))

def find_maxes(X, num):
    l = list(enumerate(X))
    maxes = []
    for i in range(num):
        maxes.append(max(l, key=(lambda x: x[1]))[0])
        l[maxes[-1]] = (maxes[-1], 0)
    return maxes


class BertHuggingface(Embedder):

    def __init__(self, num_labels, model_name=None, batch_size=16):
        self.model = None
        self.tokenizer = None
        self.num_labels = num_labels
        super().__init__(model_name=model_name, batch_size=batch_size)

    def prepare(self, **kwargs):
        model_name = kwargs.pop('model_name') or 'bert-base-uncased'

        self.model = BertForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=self.num_labels, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print('using bert with cuda')
        self.model.eval()

    def embed(self, text_list):
        outputs = []
        num_steps = int(math.ceil(len(text_list) / self.batch_size))
        for i in range(num_steps):
            ul = min((i + 1) * self.batch_size, len(text_list))
            partial_input = text_list[i * self.batch_size:ul]
            encoding = self.tokenizer(partial_input, return_tensors='pt', padding=True, truncation=True)
            if torch.cuda.is_available():
                encoding = encoding.to('cuda')
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            out = self.model(input_ids, attention_mask=attention_mask)
            encoding = encoding.to('cpu')

            hidden_states = out.hidden_states

            arr = hidden_states[-1].to('cpu')
            arr = arr.detach().numpy()

            attention_mask = attention_mask.to('cpu')
            att_mask = attention_mask.detach().numpy()

            zeros = first_zero(att_mask)
            array = []
            for entry in range(len(partial_input)):
                attention_masked_non_zero_entries = arr[entry]
                if zeros[entry] > 0:
                    attention_masked_non_zero_entries = attention_masked_non_zero_entries[:zeros[entry]]
                array.append(np.mean(attention_masked_non_zero_entries, axis=0))

            embedding_output = np.asarray(array)

            outputs.append(embedding_output)
            out = out.logits
            out = out.to('cpu')

            del encoding
            del partial_input
            del input_ids
            del attention_mask
            del out
            torch.cuda.empty_cache()
            if self.verbose and i % 100 == 0:
                print("at step", i, "of", num_steps)

        return np.vstack(outputs)


    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model = self.model.to('cpu')
        self.model = BertForSequenceClassification.from_pretrained(path)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print('using bert with cuda')
        self.model.eval()

    def predict(self, text_list):
        outputs = []
        num_steps = int(math.ceil(len(text_list) / self.batch_size))
        if self.verbose:
            print('num_steps', num_steps)
        for i in range(num_steps):
            ul = min((i + 1) * self.batch_size, len(text_list))
            partial_input = text_list[i * self.batch_size:ul]
            encoding = self.tokenizer(partial_input, return_tensors='pt', padding=True, truncation=True)
            if torch.cuda.is_available():
                encoding = encoding.to('cuda')
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            out = self.model(input_ids, attention_mask=attention_mask)
            encoding = encoding.to('cpu')
            out = out.logits
            out = out.to('cpu')

            out = out.detach().numpy()
            outputs.append(out)

            del encoding
            del partial_input
            del input_ids
            del attention_mask
            del out
            torch.cuda.empty_cache()
        return np.vstack(outputs)

    def retrain_one_epoch(self, text_list, labels):
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.model.zero_grad()

        num_steps = int(math.ceil(len(text_list) / self.batch_size))
        if self.verbose:
            print('num_steps', num_steps)
        for i in range(num_steps):
            ul = min((i + 1) * self.batch_size, len(text_list))

            partial_input = text_list[i * self.batch_size:ul]
            partial_labels = torch.tensor(labels[i * self.batch_size:ul])
            if torch.cuda.is_available():
                partial_labels = partial_labels.to('cuda')

            encoding = self.tokenizer(partial_input, return_tensors='pt', padding=True, truncation=True)
            if torch.cuda.is_available():
                encoding = encoding.to('cuda')
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, partial_labels)
            outputs.logits.to('cpu')
            # loss = F.mse_loss(outputs.logits, partial_labels)
            loss_divider = num_steps * float(len(
                partial_input)) / self.batch_size  # num_steps alone not completely accurate, as last batch can be smaller than batch_size
            loss /= loss_divider
            loss.backward()
            loss = loss.detach().item()

            optimizer.step()
            self.model.zero_grad()

            if i and not i % 100 and self.verbose:
                print(i, '/', num_steps)
            encoding = encoding.to('cpu')
            partial_labels = partial_labels.to('cpu')
            del encoding
            del partial_labels
        self.model.eval()
        torch.cuda.empty_cache()

    def attention(self, text_list, method='cosine-similarity', token_per_embedding=3):
        if method not in ['cosine-similarity', 'projection']:
            raise Exception('Method not applicable!')
        outputs = []
        num_steps = int(math.ceil(len(text_list) / self.batch_size))
        for i in range(num_steps):
            ul = min((i + 1) * self.batch_size, len(text_list))
            partial_input = text_list[i * self.batch_size:ul]
            encoding = self.tokenizer(partial_input, return_tensors='pt', padding=True, truncation=True)
            if torch.cuda.is_available():
                encoding = encoding.to('cuda')
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            out = self.model(input_ids, attention_mask=attention_mask)
            encoding = encoding.to('cpu')

            hidden_states = out.hidden_states

            arr = hidden_states[-1].to('cpu')
            arr = arr.detach().numpy()

            attention_mask = attention_mask.to('cpu')
            att_mask = attention_mask.detach().numpy()

            zeros = first_zero(att_mask)
            array = []
            attentions = []
            output = []
            for entry in range(len(partial_input)):
                attention_masked_non_zero_entries = arr[entry]
                if zeros[entry] > 0:
                    attention_masked_non_zero_entries = attention_masked_non_zero_entries[:zeros[entry]]
                array.append(np.mean(attention_masked_non_zero_entries, axis=0))
                if method == 'cosine-similarity':
                    cosines = compute_cosine_similarities(attention_masked_non_zero_entries, array[-1])
                    indicies = find_maxes(cosines, token_per_embedding)
                    tokens = [self.tokenizer.decode([input_ids[entry][i]]) for i in indicies]
                    output.append(tokens)
                elif method == 'projection':
                    pass

            embedding_output = np.asarray(output)

            outputs.append(embedding_output)
            out = out.logits
            out = out.to('cpu')

            del encoding
            del partial_input
            del input_ids
            del attention_mask
            del out
            torch.cuda.empty_cache()
            if self.verbose and i % 100 == 0:
                print("at step", i, "of", num_steps)

        return np.vstack(outputs)
