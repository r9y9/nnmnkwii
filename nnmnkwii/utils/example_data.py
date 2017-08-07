"""
Example data
============

All the example data was taken or generated from CMU Arctic data
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import pkg_resources

from nnmnkwii.datasets import DataSource, Dataset

EXAMPLE_AUDIO = 'example_audio_data/arctic_a0007.wav'


def example_acoustic_features():
    pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)
    np.fromfile("")
    pass


def example_linguistic_features():
    pass


def example_duration_features():
    pass
