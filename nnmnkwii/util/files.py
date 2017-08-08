"""
Files
=====

"""
from __future__ import division, print_function, absolute_import

import pkg_resources

__all__ = [
    "example_label_file",
    "example_audio_file",
    "example_linguistic_acoustic_pairs_file",
]


def example_label_file():
    name = "arctic_a0009"
    label_path = pkg_resources.resource_filename(
        __name__, '_example_data/{}_state.lab'.format(name))
    return label_path


def example_audio_file():
    name = "arctic_a0009"
    wav_path = pkg_resources.resource_filename(
        __name__, '_example_data/{}.wav'.format(name))
    return wav_path


def example_linguistic_acoustic_pairs_file():
    """Get file path of example linguistic/acoustic features path.

    Returns:
        str: File path of example linguistic/acoustic features.

    """
    return pkg_resources.resource_filename(__name__, '_example_data/foobar.npz')
