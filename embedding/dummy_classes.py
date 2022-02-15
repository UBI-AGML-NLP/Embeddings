class Dummyclass:
    __install_shortcut__ = 'short'

    def __init__(self):
        text = '{name} embedder is not available. Install it\'s dependencies via \"pip install .[shortcut]\"'
        print(text.format(name=type(self).__name__, shortcut=self.__install_shortcut__))


class BertUKPLab(Dummyclass):
    __install_shortcut__ = 'ukplab'

class USEEmbedder(Dummyclass):
    __install_shortcut__ = 'use'

class Doc2Vec(Dummyclass):
    __install_shortcut__ = 'doc2vec'
