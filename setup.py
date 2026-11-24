from __future__ import print_function
import sys
from setuptools import setup

INSTALL_REQUIRES = [
    'setuptools>=65.5.1',
    'numpy>=1.19.0',
    'torch>=1.13.1',
    'tqdm>=4.62.3',
    'transformers>=5.1.0',
    'sentence-transformers>=5.6.0',
    'scikit-learn>=1.2.1'
]

version = '2.0'

setup(name='agml-embedding',
      version=version,
      description='Wrapper for various embedders for NLP',
      long_description_content_type="text/markdown",
      long_description = open('README.md').read(),
      url='https://github.com/UBI-AGML-NLP/Embedding',
      packages=['embedding'],
      install_requires=INSTALL_REQUIRES
      )
