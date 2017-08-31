Change log
==========

v0.0.5
------

- Fix example data wasn't included in release tar ball.

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


.. _#17: https://github.com/r9y9/nnmnkwii/pull/17
.. _#21: https://github.com/r9y9/nnmnkwii/pull/21
.. _#22: https://github.com/r9y9/nnmnkwii/issues/22
.. _#23: https://github.com/r9y9/nnmnkwii/pull/23
.. _#25: https://github.com/r9y9/nnmnkwii/pull/25
.. _#26: https://github.com/r9y9/nnmnkwii/issues/26
.. _#27: https://github.com/r9y9/nnmnkwii/pull/27
.. _#28: https://github.com/r9y9/nnmnkwii/pull/28
