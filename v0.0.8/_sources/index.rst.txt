.. nnmnkwii documentation master file, created by
   sphinx-quickstart on Sat Jul 29 16:55:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/r9y9/nnmnkwii

nnmnkwii (nanami) documentation
================================

Library to build speech synthesis systems designed for easy and fast prototyping.

Github: https://github.com/r9y9/nnmnkwii

You can find tutorial notebooks in the document. Full code is available at `nnmnkwii_gallery`_. Also advanced examples can be found at the following repositories:

- `tacotron_pytorch`_: PyTorch implementation of `Tacotron`_ speech synthesis model.
- `gantts`_: PyTorch implementation of GAN-based text-to-speech synthesis and voice conversion (VC).

.. _`tacotron_pytorch`: https://github.com/r9y9/tacotron_pytorch
.. _`gantts`: https://github.com/r9y9/gantts
.. _`Tacotron`: https://arxiv.org/abs/1703.10135

.. _nnmnkwii_gallery: https://github.com/r9y9/nnmnkwii_gallery

.. toctree::
  :maxdepth: 1
  :caption: Notes

  design
  design_jp
  nnmnkwii_gallery/notebooks/00-Quick start guide.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   nnmnkwii_gallery/notebooks/tts/01-DNN-based statistical speech synthesis (en).ipynb
   nnmnkwii_gallery/notebooks/tts/02-Bidirectional-LSTM based RNNs for speech synthesis (en).ipynb
   nnmnkwii_gallery/notebooks/vc/01-GMM voice conversion (en).ipynb

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Package references

    references/*

.. toctree::
    :maxdepth: 1
    :caption: Meta information

    changelog


.. only:: html

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
