from __future__ import division, print_function, absolute_import

import pkg_resources


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
