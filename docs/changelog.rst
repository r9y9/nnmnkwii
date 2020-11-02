Change log
==========

v0.0.22 <2020-xx-xx>
--------------------

- `#108`_: Fix label time overflow on environment which size of np.int is 4bytes
- `#109`_: io: fix bug of handling pitch diff in the SVS full context format

v0.0.21 <2020-08-13>
--------------------

- `#99`_: Fix future warning from sklearn
- `#101`_: [hts][frontend] various fixes for singing voice synthesis. Our frontend now supports MIDI number extraction. HTSLabelFile supports list and slice indexing.


v0.0.20 <2020-03-02>
--------------------
- `#90`_: Expand available speakers list for CMU Arctic dataset
- `#91`_: Add support for JVS dataset: a free Japanese multi-speaker voice corpus.
- `#98`_: Enable python 3.8 tests on travis
- `#95`_: Fixed parameter generation error of MLPG.

--------------------

v0.0.19 <2019-07-06>
--------------------

- `#88`_: Tentative fix: make bandmat optional requirement as it is causing installation errors on python 3.7. See [here](https://github.com/MattShannon/bandmat/issues/10) for details.
- `#85`_: Fixed rounding error in caluculating number of frames.
- `#87`_: Fixed :func:`nnmnkwii.preprocessing.trim_zeros_frames` issue. Support passing ``trim`` argument.

v0.0.18 <2019-05-31>
--------------------

- Fix for python 3.7.

v0.0.17 <2018-12-25>
--------------------

- `#79`_: ModSpec update for Pytorch 1.0.0. Fix deprecations.
- `#40`_: Add logo
- `#76`_: MLPG update for PyTorch 1.0.0.

v0.0.16 <2018-08-23>
--------------------

- `#73`_: Add Japanese TTS demo to docs and fix some typos.
- `#72`_: Add TranscriptionDataSource for VoiceStatistics (ja) dataset
- `#72`_: Deprecate NormalizedTranscriptionDataSource for LJSpeech dataset.
- ``hts.load`` now accepts ``lines`` argment.

v0.0.15 <2018-07-12>
--------------------

- Fix pypi release to render markdown property

v0.0.14 <2018-06-06>
--------------------

- `#67`_: Windows support (experimental)
- `#66`_: Fixed window installation issue
- `#71`_: Bump bandmat version to v0.7

v0.0.13 <2018-01-24>
--------------------

- `#65`_: Part of example data was not installed by setuptools. Fixed.
- `#63`_: Windows CI

v0.0.12 <2018-01-04>
--------------------

- Fix typo: ``adjast_frame_length`` and ``adjast_frame_lengths`` are renamed to ``adjust_frame_length`` and ``adjust_frame_lengths``, respectively,
- `#63`_: Improved support for :func:`nnmnkwii.preprocessing.adjast_frame_length` and :func:`nnmnkwii.preprocessing.adjast_frame_lengths`. Padding for 1d array is now supported.
- BUG FIX: example audio data is now included in the release tar.gz

v0.0.11 <2017-12-22>
--------------------

- Fix RuntimeError when HTS label file has white spaces between fields. Skip comments when reading HTS labels.

v0.0.10 <2017-12-05>
--------------------

- `#61`_: Misc dataset improvements. Unified `max_files=None` from `max_files=50` and add `max_files` args for VCTK data sources.
- `#59`_: Bug fix for memory re-allocations when num frames exceed padded_initial_guess
- `#60`_: FileSourceDataset: better descriptive error messages
- `#57`_: Add ``append`` method to HTSLabelFile and simplify structure. ``frame_shift_in_micro_sec`` was removed from its property.
- `#55`_: Add mu-law companding/expansion
- Add support for JSUT dataset ver 1.1
- `#20`_: Support for mono phone labels and fix bug of ``silence_phone_indices()`` for non-state level alignment label files.

v0.0.9 <2017-11-14>
-------------------

- `#53`_: Add builtin data sources for VCTK dataset
- `#50`_: Add builtin data sources for JSUT dataset
- `#51`_: Fix modspec autograd bug for ``norm='ortho'``

v0.0.8 <2017-10-25>
-------------------

- `#49`_: Add support for build without cython
- `#46`_: Cleanup frontnend implementation

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
.. _#20: https://github.com/r9y9/nnmnkwii/issues/20
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
.. _#40: https://github.com/r9y9/nnmnkwii/issues/40
.. _#42: https://github.com/r9y9/nnmnkwii/issues/42
.. _#46: https://github.com/r9y9/nnmnkwii/pull/46
.. _#49: https://github.com/r9y9/nnmnkwii/issues/49
.. _#50: https://github.com/r9y9/nnmnkwii/issues/50
.. _#51: https://github.com/r9y9/nnmnkwii/pull/51
.. _#53: https://github.com/r9y9/nnmnkwii/issues/53
.. _#55: https://github.com/r9y9/nnmnkwii/pull/55
.. _#57: https://github.com/r9y9/nnmnkwii/pull/57
.. _#59: https://github.com/r9y9/nnmnkwii/issues/59
.. _#60: https://github.com/r9y9/nnmnkwii/pull/60
.. _#61: https://github.com/r9y9/nnmnkwii/pull/61
.. _#63: https://github.com/r9y9/nnmnkwii/pull/63
.. _#65: https://github.com/r9y9/nnmnkwii/issues/65
.. _#66: https://github.com/r9y9/nnmnkwii/issues/66
.. _#67: https://github.com/r9y9/nnmnkwii/issues/67
.. _#68: https://github.com/r9y9/nnmnkwii/pull/68
.. _#71: https://github.com/r9y9/nnmnkwii/pull/71
.. _#72: https://github.com/r9y9/nnmnkwii/pull/72
.. _#73: https://github.com/r9y9/nnmnkwii/pull/73
.. _#76: https://github.com/r9y9/nnmnkwii/pull/76
.. _#79: https://github.com/r9y9/nnmnkwii/pull/79
.. _#85: https://github.com/r9y9/nnmnkwii/issues/85
.. _#87: https://github.com/r9y9/nnmnkwii/pull/87
.. _#88: https://github.com/r9y9/nnmnkwii/pull/88
.. _#90: https://github.com/r9y9/nnmnkwii/pull/90
.. _#91: https://github.com/r9y9/nnmnkwii/issues/91
.. _#95: https://github.com/r9y9/nnmnkwii/issues/95
.. _#98: https://github.com/r9y9/nnmnkwii/pull/98
.. _#99: https://github.com/r9y9/nnmnkwii/issues/99
.. _#101: https://github.com/r9y9/nnmnkwii/pull/101
.. _#108: https://github.com/r9y9/nnmnkwii/pull/108
.. _#109: https://github.com/r9y9/nnmnkwii/pull/109