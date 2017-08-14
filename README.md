# nnmnkwii

[![PyPI](https://img.shields.io/pypi/v/nnmnkwii.svg)](https://pypi.python.org/pypi/nnmnkwii)
[![Build Status](https://travis-ci.org/r9y9/nnmnkwii.svg?branch=master)](https://travis-ci.org/r9y9/nnmnkwii)

Library to build speech synthesis systems designed for easy and fast prototyping.

## Documentation

See https://r9y9.github.io/nnmnkwii/ for reference manual and tutorials.

## Installation

The latest release is availabe on pypi. Assuming you have already ``numpy`` installed, you can install nnmnkwii by:

    pip install nnmnkwii

If you want the latest development version, run:

    pip install git+https://github.com/r9y9/nnmnkwii

or:

    git clone https://github.com/r9y9/nnmnkwii
    cd nnmnkwii
    python setup.py develop # or install

This should resolve the package dependencies and install ``nnmnkwii`` property.

At the moment, `nnmnkwii.autograd` package depends on [PyTorch](http://pytorch.org/).
If you need autograd features, please install PyTorch as well.
