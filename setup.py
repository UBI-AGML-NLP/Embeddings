from __future__ import print_function
import sys
from setuptools import setup

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

version = '1.0'

setup(name='agml-embedding',
      version=version,
      description='Wrapper for various embedders for NLP',
      url='https://github.com/UBI-AGML-NLP/Embedding',
      packages=['embedding'],
      install_requires=INSTALL_REQUIRES,
      )
