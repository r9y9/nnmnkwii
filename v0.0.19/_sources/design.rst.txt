General design documentation
============================

.. note::
    Design issue tracker https://github.com/r9y9/nnmnkwii/issues/8

The underlying design philosophy
--------------------------------

- Getting better experience on rich REPL (i.e, IPython, Jupyter) boosts research productivity.

Background
----------

- Statistical speech synthesis system typically involves many area of research; roughly speaking, text processsing, speech analysis / synthesis and machine learning, etc. Sometimes it's hard to prototype new ideas since it usually requires many deep knowledge among these areas. I think it's worth to create common reusable software for ease of future research.
- In HMM speech synthesis, HTS_ have been widely used. They provide various command line tools to allow users to create their own HMM-based statistical speech synthesis systems. However, it doesn't meet the requirements nowadays, since it's been known that DNN-based speech synthesis system can outperform HMM-based ones.
- Merlin_, which is an open source DNN-based speech synthesis toolkit, is a successor of HTS. Merlin was created to satisfy needs for DNN-based speech synthesis. One of their purpose is to help research reproducibility. From their reports, lots of research uses the toolkit to do research. That's great. However, from my perspective, it lacks flexibility. From their design, the main entry point for users is run_merlin.py, which does *everything* for you. We use Merlin to provide configuration file to the run_merlin.py. The problem of the design is that we cannot simply reuse Merlin's part of functionality other than through the run_merlin.py. Similarly, Merlin is built on top of Theano and keras as their computational backends, we cannot simply use other computational backends (PyTorch, tensorflow, etc) with Merlin.

.. _HTS: http://hts.sp.nitech.ac.jp/
.. _Merlin: https://github.com/CSTR-Edinburgh/merlin

From the background described above, I think I need a new flexible and modular library. This is why I started to create the library.

Goal
----

In my philosophy, I believe getting better experience on rich REPL (IPython, Jupyter) boosts research productivity. From this in mind, my goal is to create a modular and reusable library focused on

- Easy and fast prototyping

The libray is MIT-licensed. I hope this will help you.

So what do we provide?
----------------------

Please correct me if I'm saying wrong things about speech synthesis. Recently there's been many progress on speech synthesis research. Besides the success of typical DNN-based speech synthesis, end-to-end speech synthesis (e.g., Char2Wav_), vocoder-less speech synthesis (e.g., WaveNet_) have been investigated. To design software, we cannot practically cover all the techniques. If we want to do this, I think it complicates the software overly. In my option, it's important to focus on generic algorithms, that can be used as building blocks.

.. _Char2Wav: http://www.josesotelo.com/speechsynthesis/
.. _WaveNet: https://deepmind.com/blog/wavenet-generative-model-raw-audio/

From the success of deep learning, many computational backends (e.g., tensorflow, PyTorch, etc) have been created to help research.
I think we should provide a library which bridges generic computational backends and speech data. Hence, in this library, we provide

- Dataset and data iteration abstractions, considering arbitrary large datasets. :obj:`nnmnkwii.datasets`
- Generic functions for speech synthesis. :obj:`nnmnkwii.autograd`
- Pre-processsing, parameter generation and post-processsing utilities. :obj:`nnmnkwii.preprocessing`, :obj:`nnmnkwii.paramgen`, :obj:`nnmnkwii.postfilters`

As I believe visualization is important to understand what happens, I plan to provide visualization package (:obj:`nnmnkwii.display`) in the near future.


Design decisions
----------------

1. We provide our library as python packages that can be used in REPL. Command line tools, which would be useful for batch processing, are not included. Users are expected to create their own command line tools if necessary.
2. We use in-memory IO as possible, except for loading dataset from files.
3. We don't provide duration/acoustic models, opposite to Merlin. Users are expected to implement their own ones. For generic models may go in :obj:`nnmnkwii.baseline` module, but it's not meant to cover all the models.
4. We don't provide linguistic feature extraction frontend, except for utilities to convert structural linguistic information (e.g., HTS full-context labels ) to its numeric forms.
5. We don't provide speech analysis/synthesis backend. Users are expected to use another packages for this purpose. e.g., :obj:`pysptk`, :obj:`pyworld` and :obj:`librosa`.

We will try to keep the library to be modular, easy to understand and reusable.

Development guidelines
----------------------

-  **Do not reinvent the wheel**: Avoid reinventing the wheel as possible.
-  **Fully unit tested**: There's no software that has no bugs.
-  **Documentation**: Well documented software will help users to get stared.
