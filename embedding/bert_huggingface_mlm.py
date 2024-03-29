from .embedder import Embedder
import numpy as np
import math
from transformers import AutoModelForMaskedLM, AutoTokenizer, AdamW, pipeline
import torch
from tqdm import tqdm
import random
import string
import types


def first_zero(arr):
    mask = arr == 0
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


class MLMDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

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

        self.model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512, truncation=True)
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
        self.model = AutoModelForMaskedLM.from_pretrained(path, return_dict=True, output_hidden_states=True)
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

    def retrain(self, masked_texts, labels, epochs=2, insert_masks=False):
        losses = []
        inputs = self.create_tokens(masked_texts, labels)
        if insert_masks:
            rand = torch.rand(inputs.input_ids.shape)
            mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
            selection = []

            for i in range(inputs.input_ids.shape[0]):
                selection.append(
                    torch.flatten(mask_arr[i].nonzero()).tolist()
                )
            for i in range(inputs.input_ids.shape[0]):
                inputs.input_ids[i, selection[i]] = 103

        dataset = MLMDataset(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        optim = AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)
            epoch_loss = 0
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
                # calculate loss for every parameter that needs grad update
                loss.backward()

                # update parameters
                optim.step()

                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
                loss = loss.detach().item()
                epoch_loss += loss

                # put everything back on the cpu
                if device == 'cuda':
                    outputs.logits.to('cpu')
                    input_ids = input_ids.to('cpu')
                    attention_mask = attention_mask.to('cpu')
                    labels = labels.to('cpu')
                del input_ids
                del attention_mask
                del labels
                del outputs
            losses.append(epoch_loss)

        self.model.eval()
        torch.cuda.empty_cache()
        return losses

    def lazy_retrain(self, texts, epochs=2):
        """
        Retrain on any texts with automated, but random placed masks
        """

        def mask_random_word(doc):
            doc = doc.strip(' ')  # remove leading/tailing whitespaces
            MASK = self.tokenizer.mask_token
            words = doc.split(' ')
            mask = int(random.random() * len(words))
            if words[mask][-1] in string.punctuation:
                MASK += words[mask][-1]
            words[mask] = MASK
            masked = ' '.join(words)
            return masked

        def clear(doc):
            words = doc.split(' ')
            words = [x for x in words if x]
            s_doc = ' '.join(words)
            return s_doc

        masked_texts = []
        labels = texts
        for text in texts:
            text = clear(text)
            masked_texts.append(mask_random_word(text))

        return self.retrain(masked_texts, labels, epochs=epochs)

    def eval(self, texts, labels, top_k=1):
        preds = self.predict(texts, top_k=top_k)
        in_top_k = [0] * len(texts)
        for i, pred_single  in enumerate(preds):
            #
            for elem in pred_single:
                if elem['sequence'] == labels[i]:
                    in_top_k[i] = 1
        acc = sum(in_top_k) / len(in_top_k)
        return acc

    def predict(self, text_list, top_k=1):
        MAX_LENGTH = self.model.config.max_position_embeddings
        def _my_preprocess(self, inputs, return_tensors=None, **preprocess_parameters):
            if return_tensors is None:
                return_tensors = self.framework
            model_inputs = self.tokenizer(inputs, truncation=True, max_length=MAX_LENGTH, return_tensors=return_tensors)
            self.ensure_exactly_one_mask_token(model_inputs)
            return model_inputs

        if torch.cuda.is_available():
            unmasker = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, device=0)
            unmasker.preprocess = types.MethodType(_my_preprocess, unmasker)
        else:
            unmasker = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, device=-1)
            unmasker.preprocess = types.MethodType(_my_preprocess, unmasker)
        predictions = unmasker(text_list, top_k=top_k)
        return predictions