"""
HTS IO
======

.. autosummary::
    :toctree: generated/

    load
    load_question_set

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

    Examples:
        >>> from nnmnkwii.io import hts
        >>> labels = hts.load("path_to_labels.lab")
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

    def set_durations(self, durations):
        """Set start/end times from duration features

        TODO: this should be refactored
        """
        # Unwrap state-axis
        end_times = np.cumsum(
            durations.reshape(-1, 1) * self.frame_shift_in_micro_sec).astype(np.int)
        if len(end_times) != len(self.end_times):
            raise RuntimeError("Unexpected input, maybe")
        # Assuming first label starts with time `0`
        # Is this really true? probably no
        start_times = np.hstack((0, end_times[:-1])).astype(np.int)
        self.start_times, self.end_times = start_times, end_times

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
