Installation
============

The latest release is availabe on pypi. Assuming you have already ``numpy`` installed, you can install nnmnkwii by:

.. code:: shell

    pip install nnmnkwii

If yout want the latest development version, run:

.. code:: shell

   pip install git+https://github.com/r9y9/nnmnkwii

or:

.. code:: shell

   git clone https://github.com/r9y9/nnmnkwii
   cd nnmnkwii
   python setup.py develop # or install

This should resolve the package dependencies and install ``nnmnkwii`` property.

At the moment, :obj:`nnmnkwii.autograd` package depends on PyTorch. If you need
autograd features, please install PyTorch as well. Running tests also requires PyTorch.
