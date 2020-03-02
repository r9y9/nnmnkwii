.. nnmnkwii documentation master file, created by
   sphinx-quickstart on Sat Jul 29 16:55:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/r9y9/nnmnkwii

nnmnkwii (nanamin kawaii) documentation
=======================================

Library to build speech synthesis systems designed for easy and fast prototyping.

- Github: https://github.com/r9y9/nnmnkwii
- Tutorial notebooks: https://github.com/r9y9/nnmnkwii_gallery

For advanced applications using the library, see :ref:`external-links-label`.

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
   nnmnkwii_gallery/notebooks/tts/02-Bidirectional-LSTM based RNNs for speech synthesis using OpenJTalk (ja).ipynb
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

.. _external-links-label:

External links
--------------

- `wavenet_vocoder`_: WaveNet vocoder [6]_ [7]_
- `deepvoice3_pytorch`_: PyTorch implementation of convolutional networks-based text-to-speech synthesis models. [4]_ [5]_
- `tacotron_pytorch`_: PyTorch implementation of Tacotron speech synthesis model. [3]_
- `gantts`_: PyTorch implementation of GAN-based text-to-speech synthesis and voice conversion (VC). [1]_  [2]_
- `icassp2020-espnet-tts-merlin-baseline`_: ICASSP 2020 ESPnet-TTS: Merlin baseline system [8]_

.. _`wavenet_vocoder`: https://github.com/r9y9/wavenet_vocoder
.. _`deepvoice3_pytorch`: https://github.com/r9y9/deepvoice3_pytorch
.. _`tacotron_pytorch`: https://github.com/r9y9/tacotron_pytorch
.. _`gantts`: https://github.com/r9y9/gantts
.. _`icassp2020-espnet-tts-merlin-baseline`: https://github.com/r9y9/icassp2020-espnet-tts-merlin-baseline

.. [1] Saito, Yuki, Shinnosuke Takamichi, and Hiroshi Saruwatari. "Statistical Parametric Speech Synthesis Incorporating Generative Adversarial Networks." IEEE/ACM Transactions on Audio, Speech, and Language Processing 26.1 (2018): 84-96.
.. [2] Shan Yang, Lei Xie, Xiao Chen, Xiaoyan Lou, Xuan Zhu, Dongyan Huang, Haizhou Li, " Statistical Parametric Speech Synthesis Using Generative Adversarial Networks Under A Multi-task Learning Framework", arXiv:1707.01670, Jul 2017.
.. [3] Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton et al, "Tacotron: Towards End-to-End Speech Synthesis", 	arXiv:1703.10135, Mar 2017.
.. [4] Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.
.. [5] Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara, “Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention”. arXiv:1710.08969, Oct 2017.
.. [6] Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", arXiv:1609.03499, Sep 2016.
.. [7] Tamamori Akira, Tomoki Hayashi, Kazuhiro Kobayashi, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.
.. [8] 'T. Hayashi, R. Yamamoto, K. Inoue, T. Yoshimura, S. Watanabe,T. Toda, K. Takeda, Y. Zhang, and X. Tan, ESPnet-TTS: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit,” arXiv:1910.10909, 2019.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
