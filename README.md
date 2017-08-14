# nnmnkwii

[![wercker status](https://app.wercker.com/status/95168587f096665567ecd2033a43d20a/s/master "wercker status")](https://app.wercker.com/project/byKey/95168587f096665567ecd2033a43d20a)

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
