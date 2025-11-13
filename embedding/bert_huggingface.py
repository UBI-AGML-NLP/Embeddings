from .embedder import Embedder

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support


class DatasetForTransformer(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['index'] = idx
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


class BertHuggingface(Embedder):

    def __init__(self, num_labels: int, model_name: str = None, batch_size: int = 16, verbose: bool = False,
                 pooling: str = 'mean', optimizer: torch.optim.Optimizer = None,
                 loss_function: torch.nn.modules.loss._Loss = None, lr=1e-5, class_weights=None):
        self.model = None
        self.tokenizer = None
        self.num_labels = num_labels
        super().__init__(model_name=model_name, batch_size=batch_size, verbose=verbose)

        self.lr = lr
        if optimizer is not None:
            print("use custom optimizer")
            print(optimizer)
            self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        else:
            print("no optmizer specified, default to AdamW")
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if loss_function is None:
            print("no loss function specified, default to cross entropy")
            loss_function = torch.nn.CrossEntropyLoss

        if class_weights is not None:
            print("use positive class weights")
            class_weights = torch.tensor(class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.to('cuda')
            self.loss_function = loss_function(pos_weight=class_weights)
        else:
            self.loss_function = loss_function()

        self.pooling = 'mean'
        self.default_cls_pos = 0  # default for BERT-like models
        self.set_pooling(pooling)

    def set_pooling(self, pooling: str = 'mean'):
        if pooling not in ['mean', 'cls', 'pooling_layer']:
            print("pooling strategy %s not supported, default to mean pooling" % pooling)
            self.pooling = 'mean'
        self.pooling = pooling

        if self.pooling == 'cls':
            if self.tokenizer.cls_token is None:
                self.tokenizer.cls_token_id = self.tokenizer.eos_token_id
                self.default_cls_pos = -1

        if self.pooling == 'pooling_layer':
            # verfiy that the model has a pooling layer
            key, self.main_module = next(iter(self.model._modules.items()))
            self.pooling_layers = []
            for k in self.main_module._modules.keys():
                if 'pooler' in k:
                    self.pooling_layers.append(k)

            if len(self.pooling_layers) == 0:
                print("did not find a pooling layer, default to mean pooling")
                self.pooling = 'mean'

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
        
        # Different model families use different names for the 'max position' field
        typical_fields = ["max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"]  
        context_windows = [getattr(self.model.config, field) for field in typical_fields if field in dir(self.model.config)]
        print(context_windows[0]) if len(context_windows) else print(f"No Max input variable found for {model_name}")
        max_pos = context_windows.pop()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_pos,
                                                       truncation=True)
        if self.tokenizer.pad_token is None:
            print("manually define padding token for model %s" % model_name)
            if self.tokenizer.eos_token_id is None:
                print("no <eos> token set for this model, use <unk> token as <pad> token")
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                self.model.config.pad_token_id = self.tokenizer.unk_token_id
            else:
                print("use <eos> token as <pad> token")
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.tokenizer.padding_side = "left"  # for generator models
        self.__switch_to_cuda()
        self.model.eval()

    def save(self, path: str):
        self.model.save_pretrained(path)

    def load(self, path: str):
        self.model = self.model.to('cpu')
        print('Loading existing model...')
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.__switch_to_cuda()
        self.model.eval()

    def embed(self, texts: list[str], verbose=False):
        max_length = 512
        if self.default_cls_pos == -1:  # gpt
            # need to manually set the cls token at the end
            inputs = self.tokenizer(texts, return_tensors='pt', max_length=max_length-1, truncation=True,
                                    padding='max_length')
            cls_emb = (torch.ones(inputs['input_ids'].size()[0], 1) * self.tokenizer.cls_token_id).type(inputs['input_ids'].type())
            inputs['input_ids'] = torch.cat((inputs['input_ids'], cls_emb), 1)
            attention = torch.zeros(inputs['input_ids'].size()[0], 1).type(inputs['attention_mask'].type())
            inputs['attention_mask'] = torch.cat((inputs['attention_mask'], attention), 1)
        else:
            inputs = self.tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True,
                                    padding='max_length')
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        outputs = np.zeros((len(texts), self.model.config.hidden_size))
        for batch in tqdm(loader, leave=True):
            if torch.cuda.is_available():
                for key in batch.keys():
                    if key != 'index':
                        batch[key] = batch[key].to('cuda')

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            indices = batch['index']

            # pool embedding
            if self.pooling == 'cls':
                out = self.model(input_ids, attention_mask=attention_mask)
                if self.model.config.is_encoder_decoder:
                    token_emb = out.encoder_hidden_states[-1]
                else:
                    token_emb = out.hidden_states[-1]
                pooled_emb = token_emb[:, self.default_cls_pos, :]

            elif self.pooling == 'pooling_layer':
                out = self.main_module(batch['input_ids'], attention_mask=batch['attention_mask'])
                pooled_emb = out.pooler_output  # same name for all models that support this?

            else:  # pooling='mean'
                out = self.model(input_ids, attention_mask=attention_mask)
                if self.model.config.is_encoder_decoder:
                    token_emb = out.encoder_hidden_states[-1]
                else:
                    token_emb = out.hidden_states[-1]
                attention_repeat = torch.repeat_interleave(attention_mask, token_emb.size()[2]).reshape(
                    token_emb.size())
                pooled_emb = torch.sum(token_emb * attention_repeat, dim=1) / torch.sum(attention_repeat, dim=1)

                attention_repeat = attention_repeat.to('cpu')
                del attention_repeat

            outputs[indices.numpy()] = pooled_emb.to('cpu').detach().numpy()

            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')

            del input_ids
            del attention_mask
            del pooled_emb
            torch.cuda.empty_cache()
        return outputs

    def embed_generator(self, text_list_generator):
        for raw_texts in text_list_generator:
            yield self.embed(raw_texts)

    def predict(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        outputs = np.zeros((len(texts), self.num_labels))
        for batch in tqdm(loader, leave=True):
            if torch.cuda.is_available():
                for key in batch.keys():
                    if key != 'index':
                        batch[key] = batch[key].to('cuda')

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                indices = batch['index']

                out = self.model(input_ids, attention_mask=attention_mask)
                out = out.logits.to('cpu')

                out = out.detach().numpy()
                outputs[indices.numpy()] = pooled_emb.to('cpu').detach().numpy()

                input_ids = input_ids.to('cpu')
                attention_mask = attention_mask.to('cpu')

                del input_ids
                del attention_mask
                torch.cuda.empty_cache()
        return outputs

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

            # calculate loss for every parameter that needs grad update
            loss.backward()

            # update parameters
            self.optimizer.step()

            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

            loss = loss.detach().item()
            overall_loss += loss

            outputs.logits.to('cpu')
            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')
            labels = labels.to('cpu')
            del input_ids
            del attention_mask
            del labels

        self.model.eval()
        torch.cuda.empty_cache()
        return overall_loss
