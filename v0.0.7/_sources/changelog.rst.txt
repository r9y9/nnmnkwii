Change log
==========

v0.0.7 <2017-10-09>
-------------------

- `#12`_: [experimental] Add :obj:`nnmnkwii.metrics` package
- `#42`_: Fix installation failsure on no-utf-8 environments

v0.0.6 <2017-10-01>
-------------------

- `#38`_: Add parameter trajectory smoothing.
- `#37`_: Add ``tqdm`` as dependency. Dataset's ``asarray`` now report progress if ``verbose > 0``.
- `#37`_: Add further support for incremental mean/var computation.
- `#37`_: Add and improve normalization utilities, :func:`nnmnkwii.preprocessing.inv_scale`, :func:`nnmnkwii.preprocessing.inv_minmax_scale` and :func:`nnmnkwii.preprocessing.minmax_scale_params`.
- Add builtin data source for Voice Conversion Challenge (VCC) 2016 dataset.
- `#34`_: Add :func:`nnmnkwii.preprocessing.adjast_frame_length`.
- `#34`_: ``adjast_frame_lengths`` now supports ``divisible_by`` parameter. ``ensure_even`` is deprecated.
- `#34`_: Rename ``adjast_frame_length`` to ``adjast_Frame_lengths``
- Add references to :func:`nnmnkwii.postfilters.merlin_post_filter`.

v0.0.5 <2017-09-19>
-------------------

- `#19`_: Achieved 80% test coverage
- `#31`_: Cleanup data source implementations and add docs.
- Fix example data wasn't included in release tar ball.
- Support ``padded_length`` is ``None`` for :obj:`nnmnkwii.datasets.FileSourceDataset`.
- Automatic frame length adjastment for DTWAligner / IterativeDTWAligner

v0.0.4 <2017-09-01>
-------------------

- `#28`_: Setuptools improvements. 1) __version__ now includes git commit hash. 2) description read README.rst using pandoc.
- `#27`_: Add preemphasis / inv_preemphasis
- `#26`_: Add tests for GMM based voice conversion if swap=True
- `#25`_: fix typo in nnmnkwii/baseline/gmm.py

v0.0.3 <2017-08-26>
-------------------

- Add tests, achieve 75% test coverage.
- `#23`_, `#22`_: Preprocess rewrite & module restructure.
- `#21`_: Add new function :obj:`nnmnkwii.autograd.UnitVarianceMLPG` that can run on CPU/GPU.

v0.0.2 <2017-08-18>
-------------------

* hts io: Add support for full-context only label files
* `#17`_: ts io: Fix  wildcard handling bug
* Use pack_pad_sequence for RNN training and add tests for this
* Faster MLPG gradient computation

v0.0.1 <2017-08-14>
-------------------

* Initial release


.. _#12: https://github.com/r9y9/nnmnkwii/issues/12
.. _#17: https://github.com/r9y9/nnmnkwii/pull/17
.. _#19: https://github.com/r9y9/nnmnkwii/issues/19
.. _#21: https://github.com/r9y9/nnmnkwii/pull/21
.. _#22: https://github.com/r9y9/nnmnkwii/issues/22
.. _#23: https://github.com/r9y9/nnmnkwii/pull/23
.. _#25: https://github.com/r9y9/nnmnkwii/pull/25
.. _#26: https://github.com/r9y9/nnmnkwii/issues/26
.. _#27: https://github.com/r9y9/nnmnkwii/pull/27
.. _#28: https://github.com/r9y9/nnmnkwii/pull/28
.. _#31: https://github.com/r9y9/nnmnkwii/pull/31
.. _#34: https://github.com/r9y9/nnmnkwii/pull/34
.. _#37: https://github.com/r9y9/nnmnkwii/pull/37
.. _#38: https://github.com/r9y9/nnmnkwii/issues/38
.. _#42: https://github.com/r9y9/nnmnkwii/issues/42
