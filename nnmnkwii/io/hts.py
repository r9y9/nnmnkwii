"""
HTS IO
======

The code here is initally taken from merlin/src/label_normalisation.py and
refactored to be stateless and functional APIs.

https://github.com/CSTR-Edinburgh/merlin

.. autosummary::
    :toctree: generated/

    load
    load_question_set
    linguistic_features
    durations_features

Classes
-------

.. autoclass:: HTSLabelFile
    :members:
"""

# TODO: should define data structure that represents full-context labels?
# and add a method something likes `asarray`? This design can avoid loading
# label twice to compute both linguistic and duration features.

from __future__ import division, print_function, absolute_import

import numpy as np
import re

# TODO: consider two label alignmetn format


class HTSLabelFile(object):
    """Memory representation for HTS-style context labels file

    Attributes:
        frame_shift_in_ms (int): Frame shift in micro seconds
        start_times (ndarray): Start times
        end_times (ndarray): End times
        contexts (nadarray): Contexts.
    """

    def __init__(self, frame_shift_in_micro_sec=50000):
        self.start_times = []
        self.end_times = []
        self.contexts = []
        self.frame_shift_in_micro_sec = frame_shift_in_micro_sec

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx):
        return self.start_times[idx], self.end_times[idx], self.contexts[idx]

    def __str__(self):
        ret = ""
        for s, e, context in self:
            ret += "{} {} {}\n".format(s, e, context)
        return ret

    def __repr__(self):
        return str(self)

    def load(self, path):
        """Load labels from file

        Args:
            path (str): File path
        """
        with open(path) as f:
            lines = f.readlines()

        start_times = np.empty(len(lines), dtype=np.int)
        end_times = np.empty(len(lines), dtype=np.int)
        contexts = []
        # TODO: consider comments?
        for idx, line in enumerate(lines):
            start_time, end_time, context = line[:-1].split(" ")
            start_times[idx] = int(start_time)
            end_times[idx] = int(end_time)
            contexts.append(context)

        self.start_times = start_times
        self.end_times = end_times
        self.contexts = np.array(contexts)

    def silence_label_indices(self, regex=None):
        """Returns silence label indices

        Args:
            regex (re(optional)): Compiled regex to find silence labels.

        Returns:
            1darray: Silence label indices
        """
        if regex is None:
            regex = re.compile(".*-sil+.*")
        return np.where(list(map(regex.match, self.contexts)))[0]

    def silence_phone_indices(self, regex=None):
        """Returns phone-level frame indices

        Args:
            regex (re(optional)): Compiled regex to find silence labels.

        Returns:
            1darray: Silence label indices
        """
        if regex is None:
            regex = re.compile(".*-sil+.*")
        state_number = 5  # TODO
        return np.unique(self.silence_label_indices(regex) // state_number)

    def silence_frame_indices(self, regex=None):
        """Returns silence frame indices

        Similar to :func:`silence_label_indices`, but returns indices in frame-level.

        Args:
            regex (re(optional)): Compiled regex to find silence labels.

        Returns:
            1darray: Silence frame indices
        """
        if regex is None:
            regex = re.compile(".*-sil+.*")
        indices = self.silence_label_indices(regex)
        if len(indices) == 0:
            return np.empty(0)
        s = self.start_times[indices] // self.frame_shift_in_micro_sec
        e = self.end_times[indices] // self.frame_shift_in_micro_sec
        return np.unique(np.concatenate(
            [np.arange(a, b) for (a, b) in zip(s, e)], axis=0)).astype(np.int)

    def is_state_alignment_label(self):
        return self.contexts[0][-1] == ']' and self.contexts[0][-3] == '['

    def num_states(self):
        """Returnes number of states exclusing special begin/end states.

        Returns
        """
        if not self.is_state_alignment_label():
            return 1

        assert len(self) > 0
        initial_state_num = int(self.contexts[0][-2])
        largest_state_num = initial_state_num
        for label in self.contexts[1:]:
            n = int(label[-2])
            if n > largest_state_num:
                largest_state_num = n
            else:
                break
        return largest_state_num - initial_state_num + 1

    def num_phones(self):
        if self.is_state_alignment_label():
            return len(self) // self.num_states()
        else:
            return len(self)

    def num_frames(self):
        return self.end_times[-1] // self.frame_shift_in_micro_sec


def load(path, frame_shift_in_micro_sec=50000):
    """Load HTS-style label file and preserve it in memory.

    Args:
        path (str): Path of file.
        frame_shift_in_micro_sec (optional[int]): Frame shift in micro seconds.
            Default is 50000.

    Returns:
        labels (HTSLabelFile): Instance of HTSLabelFile.
    """
    labels = HTSLabelFile(frame_shift_in_micro_sec)
    labels.load(path)

    return labels


def is_state_alignment_label(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        assert len(lines) > 0
        first_line = lines[0][:-1]
        if first_line[-1] == ']' and first_line[-3] == '[':
            return True
    return False


def get_frame_feature_size(subphone_features="full"):
    if subphone_features is None:
        # the phoneme level features only
        return 0
    subphone_features = subphone_features.strip().lower()
    if subphone_features == "none":
        raise RuntimeError(
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
        raise RuntimeError(
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
        current_question_list = binary_dict[str(i)]
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

        current_compiled = continuous_dict[str(i)]

        ms = current_compiled.search(label)
        if ms is not None:
            continuous_value = ms.group(1)

        lab_continuous_vector[0, i] = continuous_value

    return lab_continuous_vector


def wildcards2regex(question, convert_number_pattern=False):
    """subphone_features
    Convert HTK-style question into regular expression for searching labels.
    If convert_number_pattern, keep the following sequences unescaped for
    extracting continuous values):
    (\d+)       -- handles digit without decimal point
    ([\d\.]+)   -- handles digits with and without decimal point
    """

    # handle HTK wildcards (and lack of them) at ends of label:
    if '*' in question:
        if not question.startswith('*'):
            question = '\A' + question
        if not question.endswith('*'):
            question = question + '\Z'
    question = question.strip('*')
    question = re.escape(question)
    # convert remaining HTK wildcards * and ? to equivalent regex:
    question = question.replace('\\*', '.*')

    if convert_number_pattern:
        question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
        question = question.replace(
            '\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
    return question


def load_question_set(qs_file_name):
    """Load HTS-style question and convert it to binary/continuous feature
    extraction regexes.

    Args:
        qs_file_name (str): Input HTS-style question file path

    Returns:
        (binary_dict, continuous_dict): Binary/continuous feature extraction
        regexes.
    """
    with open(qs_file_name) as f:
        lines = f.readlines()
    binary_qs_index = 0
    continuous_qs_index = 0
    binary_dict = {}
    continuous_dict = {}
    LL = re.compile(re.escape('LL-'))

    for line in lines:
        line = line.replace('\n', '')

        if len(line) > 5:
            temp_list = line.split('{')
            temp_line = temp_list[1]
            temp_list = temp_line.split('}')
            temp_line = temp_list[0]
            temp_line = temp_line.strip()
            question_list = temp_line.split(',')

            temp_list = line.split(' ')
            question_key = temp_list[1]
            if temp_list[0] == 'CQS':
                assert len(question_list) == 1
                processed_question = wildcards2regex(
                    question_list[0], convert_number_pattern=True)
                continuous_dict[str(continuous_qs_index)] = re.compile(
                    processed_question)  # save pre-compiled regular expression
                continuous_qs_index = continuous_qs_index + 1
            elif temp_list[0] == 'QS':
                re_list = []
                for temp_question in question_list:
                    processed_question = wildcards2regex(temp_question)
                    if LL.search(question_key):
                        processed_question = '^' + processed_question
                    re_list.append(re.compile(processed_question))

                binary_dict[str(binary_qs_index)] = re_list
                binary_qs_index = binary_qs_index + 1
            else:
                raise RuntimeError("Not supported question format")
    return binary_dict, continuous_dict


def load_labels_with_phone_alignment(hts_labels,
                                     binary_dict,
                                     continuous_dict,
                                     subphone_features=None,
                                     add_frame_features=False,
                                     manual_dur_data=None):
    dict_size = len(binary_dict) + len(continuous_dict)
    frame_feature_size = get_frame_feature_size(subphone_features)
    dimension = frame_feature_size + dict_size

    assert isinstance(hts_labels, HTSLabelFile)
    if add_frame_features:
        label_feature_matrix = np.empty((hts_labels.num_frames(), dimension))
    else:
        label_feature_matrix = np.empty((hts_labels.num_phones(), dimension))

    label_feature_index = 0

    if subphone_features == "coarse_coding":
        cc_features = compute_coarse_coding_features()

    for idx, (start_time, end_time, full_label) in enumerate(hts_labels):

        # to do - support different frame shift - currently hardwired to 5msec
        # currently under beta testing: support different frame shift
        if manual_dur_data is not None:
            frame_number = manual_dur_data[idx]
        else:
            frame_number = int((end_time - start_time) / 50000)

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
                    raise RuntimeError('unknown subphone_features type')

            label_feature_matrix[label_feature_index:label_feature_index +
                                 frame_number, ] = current_block_binary_array
            label_feature_index = label_feature_index + frame_number

        elif subphone_features is None:
            current_block_binary_array = label_vector
            label_feature_matrix[label_feature_index:label_feature_index +
                                 1, ] = current_block_binary_array
            label_feature_index = label_feature_index + 1
        else:
            # TODO
            assert False

    label_feature_matrix = label_feature_matrix[0:label_feature_index, ]

    return label_feature_matrix


def load_labels_with_state_alignment(hts_labels,
                                     binary_dict,
                                     continuous_dict,
                                     subphone_features=None,
                                     add_frame_features=False):
    dict_size = len(binary_dict) + len(continuous_dict)
    frame_feature_size = get_frame_feature_size(subphone_features)
    dimension = frame_feature_size + dict_size

    assert isinstance(hts_labels, HTSLabelFile)
    if add_frame_features:
        label_feature_matrix = np.empty((hts_labels.num_frames(), dimension))
    else:
        label_feature_matrix = np.empty((hts_labels.num_phones(), dimension))

    label_feature_index = 0

    # TODO
    state_number = 5

    if subphone_features == "coarse_coding":
        cc_features = compute_coarse_coding_features()

    frame_shift_in_micro_sec = hts_labels.frame_shift_in_micro_sec
    phone_duration = 0
    state_duration_base = 0
    for current_index, (start_time, end_time, full_label) in enumerate(hts_labels):
        # remove state information [k]
        assert full_label[-1] == "]"
        full_label_length = len(full_label) - 3
        state_index = full_label[full_label_length + 1]

        state_index = int(state_index) - 1
        state_index_backward = 6 - state_index  # TODO
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

        state_duration_base += frame_number

    label_feature_matrix = label_feature_matrix[0:label_feature_index, ]
    return label_feature_matrix


def linguistic_features(hts_labels, *args, **kwargs):
    """Compute linguistic features from HTS-style full-context labels

    This converts HTS-style full-context labels to it's numeric representation
    given feature extraction regexes which should be constructed from
    HTS-style question set. The input full-context must be aligned with
    phone-level or state-level.

    Args:
        hts_label (HTSLabelFile): Input full-context label file
        binary_dict (dict): Dictionary used to extract binary features
        continuous_dict (dict): Dictionary used to extrract continuous features
        subphone_features (dict): Type of sub-phone features we use.
        add_frame_features (dict): Whether add frame-level features or not.

    Returns:
        ndarray: Numpy array representation of linguistic features.
    """
    if hts_labels.is_state_alignment_label():
        return load_labels_with_state_alignment(hts_labels, *args, **kwargs)
    else:
        return load_labels_with_phone_alignment(hts_labels, *args, **kwargs)


def extract_dur_from_state_alignment_labels(hts_labels,
                                            feature_type="numerical",
                                            unit_size="state",
                                            feature_size="phoneme"):
    if not feature_type in ["binary", "numerical"]:
        raise ValueError("Not supported")
    if not unit_size in ["phoneme", "state"]:
        raise ValueError("Not supported")
    if not feature_size in ["phoneme", "frame"]:
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
    for current_index, (start_time, end_time, full_label) in enumerate(hts_labels):
        # remove state information [k]
        full_label_length = len(full_label) - 3
        state_index = full_label[full_label_length + 1]
        state_index = int(state_index) - 1

        frame_number = (
            end_time - start_time) // hts_labels.frame_shift_in_micro_sec

        if state_index == 1:
            phone_duration = frame_number

            for i in range(state_number - 1):
                s, e, _ = hts_labels[current_index + i + 1]
                phone_duration += (e - s) // hts_labels.frame_shift_in_micro_sec

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

        ### writing into dur_feature_matrix ###
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
                                            feature_size="phoneme"):
    if not feature_type in ["binary", "numerical"]:
        raise ValueError("Not supported")
    if unit_size != "phoneme":
        raise ValueError("Not supported")
    if not feature_size in ["phoneme", "frame"]:
        raise ValueError("Not supported")
    if feature_size == "phoneme":
        dur_feature_matrix = np.empty(
            (hts_labels.num_phones(), 1), dtype=np.int)
    else:
        dur_feature_matrix = np.empty(
            (hts_labels.num_frames(), 1), dtype=np.int)
    dur_feature_index = 0
    for current_index, (start_time, end_time, _) in enumerate(hts_labels):
        frame_number = (end_time - start_time) / \
            hts_labels.frame_shift_in_micro_sec

        phone_duration = frame_number

        if feature_type == "binary":
            current_block_array = np.zeros((frame_number, 1))
            current_block_array[-1] = 1
        elif feature_type == "numerical":
            current_block_array = np.array([phone_duration])
        else:
            assert False

        ### writing into dur_feature_matrix ###
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
    """Extract durations from HTS-style full-context label.

    The input full-context must be aligned with phone-level or state-level.


    Args:
        file_name (str): Input full-context label path
        feature_type (str): ``numerical`` or ``binary``
        unit_size (str): ``phoneme`` or ``state``
        feature_size (str): ``frame`` or ``phoneme``

    Returns:
        duration_features (ndarray): numpy array representation of linguistic features.
    """
    if hts_labels.is_state_alignment_label():
        return extract_dur_from_state_alignment_labels(hts_labels, *args, **kwargs)
    else:
        return extract_dur_from_phone_alignment_labels(hts_labels, *args, **kwargs)
