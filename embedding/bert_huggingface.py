from .embedder import Embedder
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


class DatasetForTransformer(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class BertHuggingface(Embedder):

    def __init__(self, num_labels: int, model_name: str = None, batch_size: int = 16, verbose: bool = False,
                 pooling: str = 'mean', optimizer: torch.optim.Optimizer = None,
                 loss_function: torch.nn.modules.loss._Loss = None, lr=1e-5):
        self.model = None
        self.tokenizer = None
        self.num_labels = num_labels
        super().__init__(model_name=model_name, batch_size=batch_size, verbose=verbose)

        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if optimizer is not None:
            print("use custom optimizer")
            print(optimizer)
            self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

        self.loss_function = torch.nn.CrossEntropyLoss()
        if loss_function is not None:
            print("use custom loss function")
            self.loss_function = loss_function()

        if pooling not in ['mean', 'cls', 'pooling_layer']:
            print("pooling strategy %s not supported, default to mean pooling" % pooling)
            pooling = 'mean'
        self.pooling = pooling
        # TODO: which models have a distinct pooling module?
        if self.pooling == 'cls':
            if self.tokenizer.cls_token is None:
                print("this model does not support the cls token, default to mean pooling instead")
                self.pooling = 'mean'
                # TODO: for GPT add cls token at end of each input like this: (need to append this manually to end of input and determine the position for each sample
                # tokenizer.add_special_tokens({'cls_token': '[CLS]'})
                # model.resize_token_embeddings(len(tokenizer))
            else:
                self.default_cls_pos = 0  # default for BERT-like models

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=self.model.config.max_position_embeddings,
                                                       truncation=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.__switch_to_cuda()
        self.model.eval()

    def embed(self, texts: list[str], verbose=False):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        outputs = []

        for batch in tqdm(loader, leave=True):
            if torch.cuda.is_available():
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda')

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            out = self.model(input_ids, attention_mask=attention_mask)

            token_emb = out.hidden_states[-1]

            # pool embedding
            if self.pooling == 'cls':
                pooled_emb = out.last_hidden_state[:, 0, :]  # cls is first token

            elif self.pooling == 'pooling_layer':
                pooled_emb = out.pooler_output  # same name for all models that support this?

            else:  # pooling='mean'
                attention_repeat = torch.repeat_interleave(attention_mask, token_emb.size()[2]).reshape(
                    token_emb.size())
                pooled_emb = torch.sum(token_emb * attention_repeat, dim=1) / torch.sum(attention_repeat, dim=1)

                attention_repeat = attention_repeat.to('cpu')
                del attention_repeat

            outputs.append(pooled_emb.to('cpu').detach().numpy())

            out = out.logits.to('cpu')
            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')

            del input_ids
            del attention_mask
            del pooled_emb
            del out

            torch.cuda.empty_cache()
        return np.vstack(outputs)

    def embed_generator(self, text_list_generator):
        for raw_texts in text_list_generator:
            yield self.embed(raw_texts)

    def save(self, path: str):
        self.model.save_pretrained(path)

    def load(self, path: str):
        self.model = self.model.to('cpu')
        print('Loading existing model...')
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.__switch_to_cuda()
        self.model.eval()

    def predict(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        outputs = []

        for batch in tqdm(loader, leave=True):
            if torch.cuda.is_available():
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda')

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                out = self.model(input_ids, attention_mask=attention_mask)
                encoding = encoding.to('cpu')
                out = out.logits
                out = out.to('cpu')

                out = out.detach().numpy()
                outputs.append(out)

                del input_ids
                del attention_mask
                del out
                torch.cuda.empty_cache()
        return np.vstack(outputs)

    def eval(self, texts: list[str], labels):
        values = self.predict(texts)
        values = [x.argmax() for x in values]
        f1 = precision_recall_fscore_support(labels, values, average='weighted')
        return f1

    def retrain(self, texts: list[str], labels, epochs: int = 2):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')
        inputs['labels'] = torch.tensor(labels)
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        for ep in range(epochs):
            losses.append(self.retrain_one_epoch(loader, epoch=ep))
        return losses

    def retrain_one_epoch(self, loader: torch.utils.data.DataLoader, epoch: int = 0):
        overall_loss = 0
        self.model.train()

        # pull all tensor batches required for training
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # extract loss
            loss = self.loss_function(outputs.logits, labels)

            outputs.logits.to('cpu')

            # calculate loss for every parameter that needs grad update
            loss.backward()

            # update parameters
            self.optimizer.step()

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
