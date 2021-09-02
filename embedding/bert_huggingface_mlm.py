from .embedder import Embedder
import numpy as np
import math
from transformers import BertForMaskedLM, BertTokenizer, AdamW, pipeline
import torch
from tqdm import tqdm


def first_zero(arr):
    mask = arr == 0
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


class MLMDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class BertHuggingfaceMLM(Embedder):

    def __init__(self, model_name=None, batch_size=16, verbose=False):
        self.model = None
        self.tokenizer = None
        super().__init__(model_name=model_name, batch_size=batch_size, verbose=verbose)

    def __switch_to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print('Using Bert with CUDA/GPU')
        else:
            print('WARNING! Using Bert on CPU!')

    def prepare(self, **kwargs):
        model_name = kwargs.pop('model_name') or 'bert-base-uncased'

        self.model = BertForMaskedLM.from_pretrained(model_name, return_dict=True, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.__switch_to_cuda()
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
        print('Loading existing model...')
        self.model = BertForMaskedLM.from_pretrained(path, return_dict=True, output_hidden_states=True)
        self.__switch_to_cuda()
        self.model.eval()

    def unmask_pipeline(self):
        return pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer)

    def create_tokens(self, masked_texts, labels):
        inputs = self.tokenizer(masked_texts, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')
        inputs['labels'] = self.tokenizer(labels, return_tensors='pt', max_length=512, truncation=True,
                                          padding='max_length').input_ids.detach().clone()
        return inputs

    def retrain(self, masked_texts, labels, epochs=2):
        inputs = self.create_tokens(masked_texts, labels)
        dataset = MLMDataset(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        optim = AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                # pull all tensor batches required for training
                device = 'cpu'
                if torch.cuda.is_available():
                    device = 'cuda'
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # process
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                # extract loss
                loss = outputs.loss
                if device == 'cuda':
                    outputs.logits.to('cpu')
                    loss.to('cpu')

                # calculate loss for every parameter that needs grad update
                loss.backward()

                # update parameters
                optim.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
                loss = loss.detach().item()

                # put everything back on the cpu
                if device == 'cuda':
                    input_ids = input_ids.to('cpu')
                    attention_mask = attention_mask.to('cpu')
                    labels = labels.to('cpu')
                del input_ids
                del attention_mask
                del labels
                del outputs

        optim.zero_grad()
        self.model.eval()
        torch.cuda.empty_cache()

