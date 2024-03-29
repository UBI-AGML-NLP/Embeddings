from .embedder import Embedder
import numpy as np
import math

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

import torch
from torch.nn import functional as F
from tqdm import tqdm
import sklearn
from sklearn.metrics import precision_recall_fscore_support


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.data.items()}

    def __len__(self):
        return len(self.data.input_ids)


class BertHuggingface(Embedder):

    def __init__(self, num_labels, model_name=None, batch_size=16, verbose=False):
        self.model = None
        self.tokenizer = None
        self.num_labels = num_labels
        super().__init__(model_name=model_name, batch_size=batch_size, verbose=verbose)

    @staticmethod
    def __first_zero(arr):
        mask = arr == 0
        return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)

    def __switch_to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print('Using Bert with CUDA/GPU')
        else:
            print('WARNING! Using Bert on CPU!')

    def prepare(self, **kwargs):
        model_name = kwargs.pop('model_name') or 'bert-base-uncased'

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True,
                                                                        num_labels=self.num_labels,
                                                                        output_hidden_states=True,
                                                                        output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.__switch_to_cuda()
        self.model.eval()

    def embed(self, text_list):
        # in case we get a tuple instead of a list:
        text_list = list(text_list)

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

            attentions = out.attentions[-1].to('cpu')

            hidden_states = out.hidden_states
            arr = hidden_states[-1].to('cpu')
            arr = arr.detach().numpy()

            attention_mask = attention_mask.to('cpu')
            att_mask = attention_mask.detach().numpy()

            zeros = self.__first_zero(att_mask)
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
            del attentions
            torch.cuda.empty_cache()
            if self.verbose and i % 100 == 0:
                print("at step", i, "of", num_steps)

        return np.vstack(outputs)

    def embed_generator(self, text_list_generator):
        for raw_texts in text_list_generator:
            yield self.embed(raw_texts)

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model = self.model.to('cpu')
        print('Loading existing model...')
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.__switch_to_cuda()
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

    def eval(self, texts, labels):
        values = self.predict(texts)
        values = [x.argmax() for x in values]
        f1 = precision_recall_fscore_support(labels, values, average='weighted')
        return f1

    def retrain(self, texts, labels, epochs=2):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')
        inputs['labels'] = torch.tensor(labels)
        dataset = BertDataset(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        for ep in range(epochs):
            losses.append(self.retrain_one_epoch(loader, epoch=ep))
        return losses

    def retrain_one_epoch(self, loader, epoch=0):
        overall_loss = 0
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        # pull all tensor batches required for training
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # extract loss
            loss = F.cross_entropy(outputs.logits, labels)

            outputs.logits.to('cpu')

            # calculate loss for every parameter that needs grad update
            loss.backward()

            # update parameters
            optimizer.step()

            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

            loss = loss.detach().item()
            overall_loss += loss

            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')
            labels = labels.to('cpu')
            del input_ids
            del attention_mask
            del labels

        self.model.eval()
        torch.cuda.empty_cache()
        return overall_loss
