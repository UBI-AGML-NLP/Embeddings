from __future__ import print_function
import sys
from setuptools import setup

INSTALL_REQUIRES = [
    'setuptools~=50.3.2',
    'numpy>=1.19.0',
    'torch==1.10.0',
    'tqdm>=4.62.3',
    'tensorflow>=2.5.2',
    'transformers==4.12.2',
    'scikit-learn>=1.2.1'
]

version = '1.2'

setup(name='agml-embedding',
      version=version,
      description='Wrapper for various embedders for NLP',
      long_description_content_type="text/markdown",
      long_description = open('README.md').read(),
      url='https://github.com/UBI-AGML-NLP/Embedding',
      packages=['embedding'],
      install_requires=INSTALL_REQUIRES,
      extras_require={
          'ukplab': ['sentence_transformers==2.1.0'],
          'use': ['tensorflow_hub==0.12.0'],
          'doc2vec': ['gensim==3.8.3']
      }
      )
