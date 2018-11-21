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
import re

# TODO: consider two label alignmetn format


class HTSLabelFile(object):
    """Memory representation for HTS-style context labels (a.k.a HTK alignment).

    Indexing is supported. It returns tuple of
    (``start_time``, ``end_time``, ``label``).

    Attributes:
        start_times (list): Start times in micro seconds.
        end_times (list): End times in micro seconds.
        contexts (list): Contexts. Each value should have either phone or
          full-context annotation.

    Examples:

        Load from file

        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_label_file
        >>> labels = hts.load(example_label_file())
        >>> print(labels[0])
        (0, 50000, 'x^x-sil+hh=iy@x_x/A:0_0_0/B:x-x-x@x-x&x-x#x-x$x-x!x-x;x-x|x\
/C:1+1+2/D:0_0/E:x+x@x+x&x+x#x+x/F:content_1/G:0_0/H:x=x@1=2|0/I:4=3/\
J:13+9-2[2]')

        Create memory representation of label

        >>> labels = hts.HTSLabelFile()
        >>> labels.append((0, 3125000, "silB"))
        0 3125000 silB
        >>> labels.append((3125000, 3525000, "m"))
        0 3125000 silB
        3125000 3525000 m
        >>> labels.append((3525000, 4325000, "i"))
        0 3125000 silB
        3125000 3525000 m
        3525000 4325000 i

        Save to file

        >>> from tempfile import TemporaryFile
        >>> with TemporaryFile("w") as f:
        ...     f.write(str(labels))
        50

    """

    def __init__(self, frame_shift_in_micro_sec=50000):
        self.start_times = []
        self.end_times = []
        self.contexts = []
        frame_shift_in_micro_sec = frame_shift_in_micro_sec

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx):
        return self.start_times[idx], self.end_times[idx], self.contexts[idx]

    def __str__(self):
        ret = ""
        if len(self.start_times) == 0:
            return ret
        for s, e, context in self:
            ret += "{} {} {}\n".format(s, e, context)
        return ret[:-1]

    def __repr__(self):
        return str(self)

    def append(self, label):
        """Append a single alignment label

        Args:
            label (tuple): tuple of (start_time, end_time, context).

        Returns:
            self

        Raises:
            ValueError: if start_time >= end_time
            ValueError: if last end time doesn't match start_time
        """
        start_time, end_time, context = label
        start_time = int(start_time)
        end_time = int(end_time)

        if start_time >= end_time:
            raise ValueError(
                "end_time ({}) must be larger than start_time ({}).".format(
                    end_time, start_time))
        if len(self.end_times) > 0 and start_time != self.end_times[-1]:
            raise ValueError(
                "start_time ({}) must be equal to the last end_time ({}).".format(
                    start_time, self.end_times[-1]))

        self.start_times.append(start_time)
        self.end_times.append(end_time)
        self.contexts.append(context)
        return self

    def set_durations(self, durations, frame_shift_in_micro_sec=50000):
        """Set start/end times from duration features

        TODO:
            this should be refactored
        """
        # Unwrap state-axis
        end_times = np.cumsum(
            durations.reshape(-1, 1) * frame_shift_in_micro_sec).astype(np.int)
        if len(end_times) != len(self.end_times):
            raise RuntimeError("Unexpected input, maybe")
        # Assuming first label starts with time `0`
        # Is this really true? probably no
        start_times = np.hstack((0, end_times[:-1])).astype(np.int)
        self.start_times, self.end_times = start_times, end_times

    def load(self, path=None, lines=None):
        """Load labels from file

        Args:
            path (str): File path
            lines (list): Content of label file. If not None, construct HTSLabelFile
                directry from it instead of loading a file.
        """
        assert path is not None or lines is not None
        if lines is None:
            with open(path) as f:
                lines = f.readlines()

        start_times = []
        end_times = []
        contexts = []
        for line in lines:
            if line[0] == "#":
                continue
            cols = line.strip().split()
            if len(cols) == 3:
                start_time, end_time, context = cols
                start_time = int(start_time)
                end_time = int(end_time)
            elif len(cols) == 1:
                start_time = -1
                end_time = -1
                context = cols[0]
            else:
                raise RuntimeError("Not supported for now")

            start_times.append(start_time)
            end_times.append(end_time)
            contexts.append(context)

        self.start_times = start_times
        self.end_times = end_times
        self.contexts = contexts

        return self

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
        return np.unique(self.silence_label_indices(regex) // self.num_states())

    def silence_frame_indices(self, regex=None, frame_shift_in_micro_sec=50000):
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
        start_times = np.array(self.start_times)
        end_times = np.array(self.end_times)
        s = start_times[indices] // frame_shift_in_micro_sec
        e = end_times[indices] // frame_shift_in_micro_sec
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

    def num_frames(self, frame_shift_in_micro_sec=50000):
        return self.end_times[-1] // frame_shift_in_micro_sec


def load(path=None, lines=None):
    """Load HTS-style label file

    Args:
        path (str): Path of file.
        lines (list): Content of label file. If not None, construct HTSLabelFile
            directry from it instead of loading a file.

    Returns:
        labels (HTSLabelFile): Instance of HTSLabelFile.

    Examples:
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_label_file
        >>> labels = hts.load(example_label_file())
    """
    labels = HTSLabelFile()
    return labels.load(path, lines)


def wildcards2regex(question, convert_number_pattern=False):
    """subphone_features
    Convert HTK-style question into regular expression for searching labels.
    If convert_number_pattern, keep the following sequences unescaped for
    extracting continuous values):
    (\d+)       -- handles digit without decimal point
    ([\d\.]+)   -- handles digits with and without decimal point
    """

    # handle HTK wildcards (and lack of them) at ends of label:
    prefix = ""
    postfix = ""
    if '*' in question:
        if not question.startswith('*'):
            prefix = "\A"
        if not question.endswith('*'):
            postfix = "\Z"
    question = question.strip('*')
    question = re.escape(question)
    # convert remaining HTK wildcards * and ? to equivalent regex:
    question = question.replace('\\*', '.*')
    question = prefix + question + postfix

    if convert_number_pattern:
        question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
        question = question.replace(
            '\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
    return question


def load_question_set(qs_file_name):
    """Load HTS-style question and convert it to binary/continuous feature
    extraction regexes.

    This code was taken from Merlin.

    Args:
        qs_file_name (str): Input HTS-style question file path

    Returns:
        (binary_dict, continuous_dict): Binary/continuous feature extraction
        regexes.

    Examples:
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_question_file
        >>> binary_dict, continuous_dict = hts.load_question_set(example_question_file())
    """
    with open(qs_file_name) as f:
        lines = f.readlines()
    binary_qs_index = 0
    continuous_qs_index = 0
    binary_dict = {}
    continuous_dict = {}
    # I guess `LL` means Left-left, but it doesn't seem to be docmented
    # anywhere
    LL = re.compile(re.escape('LL-'))

    for line in lines:
        line = line.replace('\n', '')
        temp_list = line.split()
        if len(line) <= 0:
            continue
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
            continuous_dict[continuous_qs_index] = re.compile(
                processed_question)  # save pre-compiled regular expression
            continuous_qs_index = continuous_qs_index + 1
        elif temp_list[0] == 'QS':
            re_list = []
            # import ipdb; ipdb.set_trace()
            for temp_question in question_list:
                processed_question = wildcards2regex(temp_question)
                if LL.search(question_key) and processed_question[0] != '^':
                    processed_question = '^' + processed_question
                re_list.append(re.compile(processed_question))

            binary_dict[binary_qs_index] = re_list
            binary_qs_index = binary_qs_index + 1
        else:
            raise RuntimeError("Not supported question format")
    return binary_dict, continuous_dict
