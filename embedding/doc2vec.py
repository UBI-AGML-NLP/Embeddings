from .embedder import Embedder
import numpy as np



class Doc2Vec(Embedder):
    def __init__(self):
        self.embedder = None
        super().__init__()

    def prepare(self, **kwargs):
        pass

    def embed(self, text_list):
        pass

