#!/usr/bin/env python
from setuptools import find_packages, setup

setup(name='ApproPO',
      version='1.0',
      author='Kianté Brantely, Miro Dudík, Hal Daumé III',
      packages=find_packages(),
      install_requires=[
          'gymnasium',
          'matplotlib',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
          'six',
          'torch',
      ],
)
