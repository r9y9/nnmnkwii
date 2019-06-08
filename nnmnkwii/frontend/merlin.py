# coding: utf-8

# Part of code here is adapted from Merlin. Their license follows:
##########################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://github.com/CSTR-Edinburgh/merlin
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
##########################################################################

from __future__ import division, print_function, absolute_import

import numpy as np

from nnmnkwii.io import hts


def get_frame_feature_size(subphone_features="full"):
    if subphone_features is None:
        # the phoneme level features only
        return 0
    subphone_features = subphone_features.strip().lower()
    if subphone_features == "none":
        raise ValueError(
            "subphone_features = 'none' is deprecated, use None instead")
    if subphone_features == 'full':
        return 9  # zhizheng's original 5 state features + 4 phoneme features
    elif subphone_features == 'minimal_frame':
        # the minimal features necessary to go from a state-level to
        # frame-level model
        return 2
    elif subphone_features == 'state_only':
        return 1  # this is equivalent to a state-based system
    elif subphone_features == 'frame_only':
        # this is equivalent to a frame-based system without relying on
        # state-features
        return 1
    elif subphone_features == 'uniform_state':
        # this is equivalent to a frame-based system with uniform
        # state-features
        return 2
    elif subphone_features == 'minimal_phoneme':
        # this is equivalent to a frame-based system with minimal features
        return 3
    elif subphone_features == 'coarse_coding':
        # this is equivalent to a frame-based positioning system reported in
        # Heiga Zen's work
        return 4
    else:
        raise ValueError(
            'Unknown value for subphone_features: %s' % (subphone_features))
    assert False


def compute_coarse_coding_features(num_states=3, npoints=600):
    # TODO
    assert num_states == 3
    cc_features = np.zeros((num_states, npoints))

    x1 = np.linspace(-1.5, 1.5, npoints)
    x2 = np.linspace(-1.0, 2.0, npoints)
    x3 = np.linspace(-0.5, 2.5, npoints)

    mu1 = 0.0
    mu2 = 0.5
    mu3 = 1.0

    sigma = 0.4

    from scipy.stats import norm
    cc_features[0, :] = norm(mu1, sigma).pdf(x1)
    cc_features[1, :] = norm(mu2, sigma).pdf(x2)
    cc_features[2, :] = norm(mu3, sigma).pdf(x3)

    return cc_features


def extract_coarse_coding_features_relative(cc_features, phone_duration):
    dur = int(phone_duration)

    cc_feat_matrix = np.zeros((dur, 3), dtype=np.float32)

    for i in range(dur):
        # TODO: does the magic number really make sense?
        # need to investigate
        rel_indx = int((200 / float(dur)) * i)
        cc_feat_matrix[i, 0] = cc_features[0, 300 + rel_indx]
        cc_feat_matrix[i, 1] = cc_features[1, 200 + rel_indx]
        cc_feat_matrix[i, 2] = cc_features[2, 100 + rel_indx]

    return cc_feat_matrix


def pattern_matching_binary(binary_dict, label):
    dict_size = len(binary_dict)
    lab_binary_vector = np.zeros((1, dict_size), dtype=np.int)

    for i in range(dict_size):
        current_question_list = binary_dict[i]
        binary_flag = 0
        for iq in range(len(current_question_list)):
            current_compiled = current_question_list[iq]

            ms = current_compiled.search(label)
            if ms is not None:
                binary_flag = 1
                break
        lab_binary_vector[0, i] = binary_flag

    return lab_binary_vector


def pattern_matching_continous_position(continuous_dict, label):
    dict_size = len(continuous_dict)

    lab_continuous_vector = np.zeros((1, dict_size), dtype=np.float32)
    for i in range(dict_size):

        continuous_value = -1.0

        current_compiled = continuous_dict[i]

        ms = current_compiled.search(label)
        if ms is not None:
            continuous_value = ms.group(1)

        lab_continuous_vector[0, i] = continuous_value

    return lab_continuous_vector


def load_labels_with_phone_alignment(hts_labels,
                                     binary_dict,
                                     continuous_dict,
                                     subphone_features=None,
                                     add_frame_features=False,
                                     frame_shift_in_micro_sec=50000):
    dict_size = len(binary_dict) + len(continuous_dict)
    frame_feature_size = get_frame_feature_size(subphone_features)
    dimension = frame_feature_size + dict_size

    assert isinstance(hts_labels, hts.HTSLabelFile)
    if add_frame_features:
        label_feature_matrix = np.empty((hts_labels.num_frames(), dimension))
    else:
        label_feature_matrix = np.empty((hts_labels.num_phones(), dimension))

    label_feature_index = 0

    if subphone_features == "coarse_coding":
        cc_features = compute_coarse_coding_features()

    for idx, (start_time, end_time, full_label) in enumerate(hts_labels):
        frame_number = int(end_time / frame_shift_in_micro_sec) - int(start_time / frame_shift_in_micro_sec)

        label_binary_vector = pattern_matching_binary(
            binary_dict, full_label)

        # if there is no CQS question, the label_continuous_vector will
        # become to empty
        label_continuous_vector = pattern_matching_continous_position(
            continuous_dict, full_label)
        label_vector = np.concatenate(
            [label_binary_vector, label_continuous_vector], axis=1)

        if subphone_features == "coarse_coding":
            cc_feat_matrix = extract_coarse_coding_features_relative(cc_features,
                                                                     frame_number)

        if add_frame_features:
            current_block_binary_array = np.zeros(
                (frame_number, dict_size + frame_feature_size))
            for i in range(frame_number):
                current_block_binary_array[i,
                                           0:dict_size] = label_vector

                if subphone_features == 'minimal_phoneme':
                    # features which distinguish frame position in phoneme
                    # fraction through phone forwards
                    current_block_binary_array[i, dict_size] = float(
                        i + 1) / float(frame_number)
                    # fraction through phone backwards
                    current_block_binary_array[i, dict_size + 1] = float(
                        frame_number - i) / float(frame_number)
                    # phone duration
                    current_block_binary_array[i,
                                               dict_size + 2] = float(frame_number)

                elif subphone_features == 'coarse_coding':
                    # features which distinguish frame position in phoneme
                    # using three continous numerical features
                    current_block_binary_array[i,
                                               dict_size + 0] = cc_feat_matrix[i, 0]
                    current_block_binary_array[i,
                                               dict_size + 1] = cc_feat_matrix[i, 1]
                    current_block_binary_array[i,
                                               dict_size + 2] = cc_feat_matrix[i, 2]
                    current_block_binary_array[i,
                                               dict_size + 3] = float(frame_number)

                elif subphone_features is None:
                    pass
                else:
                    raise ValueError(
                        "Combination of subphone_features and add_frame_features is not supported: {}, {}".format(
                            subphone_features, add_frame_features))

            label_feature_matrix[label_feature_index:label_feature_index +
                                 frame_number, ] = current_block_binary_array
            label_feature_index = label_feature_index + frame_number

        elif subphone_features is None:
            current_block_binary_array = label_vector
            label_feature_matrix[label_feature_index:label_feature_index +
                                 1, ] = current_block_binary_array
            label_feature_index = label_feature_index + 1
        else:
            pass

    # omg
    if label_feature_index == 0:
        raise ValueError("Combination of subphone_features and add_frame_features is not supported: {}, {}".format(
            subphone_features, add_frame_features))

    label_feature_matrix = label_feature_matrix[0:label_feature_index, ]

    return label_feature_matrix


def load_labels_with_state_alignment(hts_labels,
                                     binary_dict,
                                     continuous_dict,
                                     subphone_features=None,
                                     add_frame_features=False,
                                     frame_shift_in_micro_sec=50000):
    dict_size = len(binary_dict) + len(continuous_dict)
    frame_feature_size = get_frame_feature_size(subphone_features)
    dimension = frame_feature_size + dict_size

    assert isinstance(hts_labels, hts.HTSLabelFile)
    if add_frame_features:
        label_feature_matrix = np.empty((hts_labels.num_frames(), dimension))
    else:
        label_feature_matrix = np.empty((hts_labels.num_phones(), dimension))

    label_feature_index = 0
    state_number = hts_labels.num_states()

    if subphone_features == "coarse_coding":
        cc_features = compute_coarse_coding_features()

    phone_duration = 0
    state_duration_base = 0
    for current_index, (start_time, end_time,
                        full_label) in enumerate(hts_labels):
        # remove state information [k]
        assert full_label[-1] == "]"
        full_label_length = len(full_label) - 3
        state_index = full_label[full_label_length + 1]

        state_index = int(state_index) - 1
        state_index_backward = state_number + 1 - state_index
        full_label = full_label[0:full_label_length]

        frame_number = (end_time - start_time) // frame_shift_in_micro_sec

        if state_index == 1:
            current_frame_number = 0
            phone_duration = frame_number
            state_duration_base = 0

            label_binary_vector = pattern_matching_binary(
                binary_dict, full_label)

            # if there is no CQS question, the label_continuous_vector will
            # become to empty
            label_continuous_vector = pattern_matching_continous_position(
                continuous_dict, full_label)
            label_vector = np.concatenate(
                [label_binary_vector, label_continuous_vector], axis=1)

            for i in range(state_number - 1):
                s, e, _ = hts_labels[current_index + i + 1]
                phone_duration += (e - s) // frame_shift_in_micro_sec

            if subphone_features == "coarse_coding":
                cc_feat_matrix = extract_coarse_coding_features_relative(
                    cc_features, phone_duration)

        if add_frame_features:
            current_block_binary_array = np.zeros(
                (frame_number, dict_size + frame_feature_size))
            for i in range(frame_number):
                current_block_binary_array[i,
                                           0: dict_size] = label_vector

                if subphone_features == 'full':
                    # Zhizheng's original 9 subphone features:
                    # fraction through state (forwards)
                    current_block_binary_array[i, dict_size] = float(
                        i + 1) / float(frame_number)
                    # fraction through state (backwards)
                    current_block_binary_array[i, dict_size + 1] = float(
                        frame_number - i) / float(frame_number)
                    # length of state in frames
                    current_block_binary_array[i,
                                               dict_size + 2] = float(frame_number)
                    # state index (counting forwards)
                    current_block_binary_array[i,
                                               dict_size + 3] = float(state_index)
                    # state index (counting backwards)
                    current_block_binary_array[i, dict_size +
                                               4] = float(state_index_backward)

                    # length of phone in frames
                    current_block_binary_array[i,
                                               dict_size + 5] = float(phone_duration)
                    # fraction of the phone made up by current state
                    current_block_binary_array[i, dict_size +
                                               6] = float(frame_number) / float(phone_duration)
                    # fraction through phone (backwards)
                    current_block_binary_array[i, dict_size + 7] = float(
                        phone_duration - i - state_duration_base) / float(phone_duration)
                    # fraction through phone (forwards)
                    current_block_binary_array[i, dict_size + 8] = float(
                        state_duration_base + i + 1) / float(phone_duration)

                elif subphone_features == 'state_only':
                    # features which only distinguish state:
                    current_block_binary_array[i, dict_size] = float(
                        state_index)  # state index (counting forwards)

                elif subphone_features == 'frame_only':
                    # features which distinguish frame position in phoneme:
                    current_frame_number += 1
                    # fraction through phone (counting forwards)
                    current_block_binary_array[i, dict_size] = float(
                        current_frame_number) / float(phone_duration)

                elif subphone_features == 'uniform_state':
                    # features which distinguish frame position in phoneme:
                    current_frame_number += 1
                    # fraction through phone (counting forwards)
                    current_block_binary_array[i, dict_size] = float(
                        current_frame_number) / float(phone_duration)
                    new_state_index = max(
                        1, round(float(current_frame_number) / float(phone_duration) * 5))
                    # state index (counting forwards)
                    current_block_binary_array[i,
                                               dict_size + 1] = float(new_state_index)

                elif subphone_features == "coarse_coding":
                    # features which distinguish frame position in phoneme
                    # using three continous numerical features
                    current_block_binary_array[i, dict_size +
                                               0] = cc_feat_matrix[current_frame_number, 0]
                    current_block_binary_array[i, dict_size +
                                               1] = cc_feat_matrix[current_frame_number, 1]
                    current_block_binary_array[i, dict_size +
                                               2] = cc_feat_matrix[current_frame_number, 2]
                    current_block_binary_array[i,
                                               dict_size + 3] = float(phone_duration)
                    current_frame_number += 1

                elif subphone_features == 'minimal_frame':
                    # features which distinguish state and minimally frame
                    # position in state:
                    current_block_binary_array[i, dict_size] = float(
                        i + 1) / float(frame_number)  # fraction through state (forwards)
                    # state index (counting forwards)
                    current_block_binary_array[i,
                                               dict_size + 1] = float(state_index)
                elif subphone_features is None:
                    pass
                else:
                    assert False

            label_feature_matrix[label_feature_index:label_feature_index +
                                 frame_number] = current_block_binary_array
            label_feature_index = label_feature_index + frame_number
        elif subphone_features == 'state_only' and state_index == state_number:
            # TODO: this pass seems not working
            current_block_binary_array = np.zeros(
                (state_number, dict_size + frame_feature_size))
            for i in range(state_number):
                current_block_binary_array[i,
                                           0:dict_size] = label_vector
                current_block_binary_array[i, dict_size] = float(
                    i + 1)  # state index (counting forwards)
            label_feature_matrix[label_feature_index:label_feature_index +
                                 state_number, ] = current_block_binary_array
            label_feature_index = label_feature_index + state_number
        elif subphone_features is None and state_index == state_number:
            current_block_binary_array = label_vector
            label_feature_matrix[label_feature_index:label_feature_index +
                                 1, ] = current_block_binary_array
            label_feature_index = label_feature_index + 1
        else:
            pass

        state_duration_base += frame_number

    # omg
    if label_feature_index == 0:
        raise ValueError("Combination of subphone_features and add_frame_features is not supported: {}, {}".format(
            subphone_features, add_frame_features))

    label_feature_matrix = label_feature_matrix[0:label_feature_index, ]
    return label_feature_matrix


def linguistic_features(hts_labels, *args, **kwargs):
    """Linguistic features from HTS-style full-context labels.

    This converts HTS-style full-context labels to it's numeric representation
    given feature extraction regexes which should be constructed from
    HTS-style question set. The input full-context must be aligned with
    phone-level or state-level.　

    .. note::
        The implementation is adapted from Merlin, but no internal algorithms are
        changed. Unittests ensure this can get same results with Merlin
        for several typical settings.

    Args:
        hts_label (hts.HTSLabelFile): Input full-context label file
        binary_dict (dict): Dictionary used to extract binary features
        continuous_dict (dict): Dictionary used to extrract continuous features
        subphone_features (dict): Type of sub-phone features. According
          to the Merlin's source code, None, ``full``, ``state_only``,
          ``frame_only``, ``uniform_state``, ``minimal_phoneme`` and
          ``coarse_coding`` are supported. **However**, None, ``full`` (for state
          alignment) and ``coarse_coding`` (phone alignment) are only tested in
          this library. Default is None.
        add_frame_features (dict): Whether add frame-level features or not.
          Default is False.
        frame_shift_in_micro_sec (int) : Frame shift of alignment in micro seconds.

    Returns:
        numpy.ndarray: Numpy array representation of linguistic features.

    Examples:
        For state-level labels

        >>> from nnmnkwii.frontend import merlin as fe
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_label_file, example_question_file
        >>> labels = hts.load(example_label_file(phone_level=False))
        >>> binary_dict, continuous_dict = hts.load_question_set(example_question_file())
        >>> features = fe.linguistic_features(labels, binary_dict, continuous_dict,
        ...     subphone_features="full", add_frame_features=True)
        >>> features.shape
        (615, 425)
        >>> features = fe.linguistic_features(labels, binary_dict, continuous_dict,
        ...     subphone_features=None, add_frame_features=False)
        >>> features.shape
        (40, 416)

        For phone-level labels

        >>> from nnmnkwii.frontend import merlin as fe
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_label_file, example_question_file
        >>> labels = hts.load(example_label_file(phone_level=True))
        >>> binary_dict, continuous_dict = hts.load_question_set(example_question_file())
        >>> features = fe.linguistic_features(labels, binary_dict, continuous_dict,
        ...     subphone_features="coarse_coding", add_frame_features=True)
        >>> features.shape
        (615, 420)
        >>> features = fe.linguistic_features(labels, binary_dict, continuous_dict,
        ...     subphone_features=None, add_frame_features=False)
        >>> features.shape
        (40, 416)

    """
    if hts_labels.is_state_alignment_label():
        return load_labels_with_state_alignment(hts_labels, *args, **kwargs)
    else:
        return load_labels_with_phone_alignment(hts_labels, *args, **kwargs)


def extract_dur_from_state_alignment_labels(hts_labels,
                                            feature_type="numerical",
                                            unit_size="state",
                                            feature_size="phoneme",
                                            frame_shift_in_micro_sec=50000):
    if feature_type not in ["binary", "numerical"]:
        raise ValueError("Not supported")
    if unit_size not in ["phoneme", "state"]:
        raise ValueError("Not supported")
    if feature_size not in ["phoneme", "frame"]:
        raise ValueError("Not supported")

    dur_dim = hts_labels.num_states() if unit_size == "state" else 1
    if feature_size == "phoneme":
        dur_feature_matrix = np.empty(
            (hts_labels.num_phones(), dur_dim), dtype=np.int)
    else:
        dur_feature_matrix = np.empty(
            (hts_labels.num_frames(), dur_dim), dtype=np.int)

    current_dur_array = np.zeros((dur_dim, 1))
    state_number = hts_labels.num_states()
    dur_dim = state_number

    dur_feature_index = 0
    for current_index, (start_time, end_time,
                        full_label) in enumerate(hts_labels):
        # remove state information [k]
        full_label_length = len(full_label) - 3
        state_index = full_label[full_label_length + 1]
        state_index = int(state_index) - 1

        frame_number = (
            end_time - start_time) // frame_shift_in_micro_sec

        if state_index == 1:
            phone_duration = frame_number

            for i in range(state_number - 1):
                s, e, _ = hts_labels[current_index + i + 1]
                phone_duration += (e - s) // frame_shift_in_micro_sec

        if feature_type == "binary":
            current_block_array = np.zeros((frame_number, 1))
            if unit_size == "state":
                current_block_array[-1] = 1
            elif unit_size == "phoneme":
                if state_index == state_number:
                    current_block_array[-1] = 1
            else:
                assert False
        elif feature_type == "numerical":
            if unit_size == "state":
                current_dur_array[current_index % 5] = frame_number
                if feature_size == "phoneme" and state_index == state_number:
                    current_block_array = current_dur_array.transpose()
                if feature_size == "frame":
                    current_block_array = np.tile(
                        current_dur_array.transpose(), (frame_number, 1))
            elif unit_size == "phoneme":
                current_block_array = np.array([phone_duration])
            else:
                assert False

        # writing into dur_feature_matrix
        if feature_size == "frame":
            dur_feature_matrix[dur_feature_index:dur_feature_index +
                               frame_number, ] = current_block_array
            dur_feature_index = dur_feature_index + frame_number
        elif feature_size == "phoneme" and state_index == state_number:
            dur_feature_matrix[dur_feature_index:dur_feature_index +
                               1, ] = current_block_array
            dur_feature_index = dur_feature_index + 1
        else:
            pass

    # dur_feature_matrix = dur_feature_matrix[0:dur_feature_index, ]
    return dur_feature_matrix


def extract_dur_from_phone_alignment_labels(hts_labels,
                                            feature_type="numerical",
                                            unit_size="phoneme",
                                            feature_size="phoneme",
                                            frame_shift_in_micro_sec=50000):
    if feature_type not in ["binary", "numerical"]:
        raise ValueError("Not supported")
    if unit_size != "phoneme":
        raise ValueError("Not supported")
    if feature_size not in ["phoneme", "frame"]:
        raise ValueError("Not supported")
    if feature_size == "phoneme":
        dur_feature_matrix = np.empty(
            (hts_labels.num_phones(), 1), dtype=np.int)
    else:
        dur_feature_matrix = np.empty(
            (hts_labels.num_frames(), 1), dtype=np.int)
    dur_feature_index = 0
    for current_index, (start_time, end_time, _) in enumerate(hts_labels):
        frame_number = (end_time - start_time) / frame_shift_in_micro_sec

        phone_duration = frame_number

        if feature_type == "binary":
            current_block_array = np.zeros((frame_number, 1))
            current_block_array[-1] = 1
        elif feature_type == "numerical":
            current_block_array = np.array([phone_duration])
        else:
            assert False

        # writing into dur_feature_matrix
        if feature_size == "frame":
            dur_feature_matrix[dur_feature_index:dur_feature_index +
                               frame_number] = current_block_array
            dur_feature_index = dur_feature_index + frame_number
        elif feature_size == "phoneme":
            dur_feature_matrix[dur_feature_index:dur_feature_index +
                               1] = current_block_array
            dur_feature_index = dur_feature_index + 1
        else:
            assert False

    # dur_feature_matrix = dur_feature_matrix[0:dur_feature_index]
    return dur_feature_matrix


def duration_features(hts_labels, *args, **kwargs):
    """Duration features from HTS-style full-context labels.

    The input full-context must be aligned with phone-level or state-level.

    .. note::
        The implementation is adapted from Merlin, but no internal algorithms are
        changed. Unittests ensure this can get same results with Merlin
        for several typical settings.

    Args:
        hts_labels (hts.HTSLabelFile): HTS label file.
        feature_type (str): ``numerical`` or ``binary``. Default is ``numerical``.
        unit_size (str): ``phoneme`` or ``state``. Default for state-level and
          phone-level alignment is ``state`` and ``phoneme``, respectively.
        feature_size (str): ``frame`` or ``phoneme``. Default is ``phoneme``.
          ``frame`` is only supported for state-level alignments.
        frame_shift_in_micro_sec (int) : Frame shift of alignment in micro seconds.

    Returns:
        numpy.ndarray: numpy array representation of duration features.

    Examples:
        For state-level alignments

        >>> from nnmnkwii.frontend import merlin as fe
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_label_file
        >>> labels = hts.load(example_label_file(phone_level=False))
        >>> features = fe.duration_features(labels)
        >>> features.shape
        (40, 5)

        For phone-level alignments

        >>> from nnmnkwii.frontend import merlin as fe
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_label_file
        >>> labels = hts.load(example_label_file(phone_level=True))
        >>> features = fe.duration_features(labels)
        >>> features.shape
        (40, 1)

    """
    if hts_labels.is_state_alignment_label():
        return extract_dur_from_state_alignment_labels(
            hts_labels, *args, **kwargs)
    else:
        return extract_dur_from_phone_alignment_labels(
            hts_labels, *args, **kwargs)
