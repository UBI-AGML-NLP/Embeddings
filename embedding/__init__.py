import importlib.util

from .bert_huggingface import BertHuggingface
from .bert_huggingface_mlm import BertHuggingfaceMLM

spec = importlib.util.find_spec('sentence_transformers')
if spec is not None:
    from .bert_ukplab import BertUKPLab
else:
    from .dummy_classes import BertUKPLab

spec = importlib.util.find_spec('tensorflow_hub')
if spec is not None:
    from .use_embedder import USEEmbedder
else:
    from .dummy_classes import USEEmbedder

spec = importlib.util.find_spec('gensim')
if spec is not None:
    from .doc2vec import Doc2Vec
else:
    from .dummy_classes import Doc2Vec

