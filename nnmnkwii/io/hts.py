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

import re
from copy import copy

import numpy as np


class HTSLabelFile(object):
    """Memory representation for HTS-style context labels (a.k.a HTK alignment).

    Indexing is supported. It returns tuple of
    (``start_time``, ``end_time``, ``label``).

    Attributes:
        start_times (list): Start times in 100ns units.
        end_times (list): End times in 100ns units.
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

    def __init__(self, frame_shift=50000):
        self.start_times = []
        self.end_times = []
        self.contexts = []
        self.frame_shift = frame_shift

    @classmethod
    def create_from_contexts(cls, contexts):
        return cls().load(None, contexts)

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # yes, this is inefficient and there will probably a bette way
            # but this is okay for now
            current, stop, _ = idx.indices(len(self))
            obj = copy(self)
            obj.start_times = obj.start_times[current:stop]
            obj.end_times = obj.end_times[current:stop]
            obj.contexts = obj.contexts[current:stop]
            return obj
        elif isinstance(idx, list):
            obj = copy(self)
            obj.start_times = list(np.asarray(obj.start_times)[idx])
            obj.end_times = list(np.asarray(obj.end_times)[idx])
            obj.contexts = list(np.asarray(obj.contexts)[idx])
            return obj
        else:
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

    def round_(self):
        s = self.frame_shift
        self.start_times = list(
            np.round(np.asarray(self.start_times) / s).astype(np.int64) * s
        )
        self.end_times = list(
            np.round(np.asarray(self.end_times) / s).astype(np.int64) * s
        )
        return self

    def append(self, label, strict=True):
        """Append a single alignment label

        Args:
            label (tuple): tuple of (start_time, end_time, context).
            strict (bool): strict mode.

        Returns:
            self

        Raises:
            ValueError: if start_time >= end_time
            ValueError: if last end time doesn't match start_time
        """
        start_time, end_time, context = label
        start_time = int(start_time)
        end_time = int(end_time)

        if strict:
            if start_time >= end_time:
                raise ValueError(
                    "end_time ({}) must be larger than start_time ({}).".format(
                        end_time, start_time
                    )
                )
            if len(self.end_times) > 0 and start_time != self.end_times[-1]:
                raise ValueError(
                    "start_time ({}) must be equal to the last end_time ({}).".format(
                        start_time, self.end_times[-1]
                    )
                )

        self.start_times.append(start_time)
        self.end_times.append(end_time)
        self.contexts.append(context)
        return self

    def set_durations(self, durations, frame_shift=50000):
        """Set start/end times from duration features

        TODO:
            this should be refactored
        """
        offset = self.start_times[0]

        # Unwrap state-axis
        end_times = offset + np.cumsum(durations.reshape(-1, 1) * frame_shift).astype(
            np.int64
        )
        if len(end_times) != len(self.end_times):
            raise RuntimeError("Unexpected input, maybe")
        start_times = np.hstack((offset, end_times[:-1])).astype(np.int64)
        self.start_times, self.end_times = start_times, end_times

    def load(self, path=None, lines=None):
        """Load labels from file

        Args:
            path (str): File path
            lines (list): Content of label file. If not None, construct HTSLabelFile
                directry from it instead of loading a file.

        Raises:
            ValueError: if the content of labels is empty.
        """
        assert path is not None or lines is not None
        if lines is None:
            with open(path) as f:
                lines = f.readlines()
        else:
            if len(lines) == 0:
                raise ValueError(
                    "Empty label is specifid! Please check if input contains a content."
                )

        is_sec_format = False
        start_times = []
        end_times = []
        contexts = []
        for line in lines:
            if line[0] == "#":
                continue
            cols = line.strip().split()
            if len(cols) == 3:
                start_time, end_time, context = cols
                if "." in start_time or "." in end_time:
                    is_sec_format = True
                if is_sec_format:
                    # convert sec to 100ns (HTS format)
                    start_time = int(1e7 * float(start_time))
                    end_time = int(1e7 * float(end_time))
                else:
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

    def silence_frame_indices(self, regex=None, frame_shift=50000):
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
        s = start_times[indices] // frame_shift
        e = end_times[indices] // frame_shift
        return np.unique(
            np.concatenate([np.arange(a, b) for (a, b) in zip(s, e)], axis=0)
        ).astype(np.int64)

    def is_state_alignment_label(self):
        return self.contexts[0][-1] == "]" and self.contexts[0][-3] == "["

    def num_states(self):
        """Returnes number of states exclusing special begin/end states."""
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

    def num_frames(self, frame_shift=50000):
        return self.end_times[-1] // frame_shift


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


def wildcards2regex(question, convert_number_pattern=False, convert_svs_pattern=True):
    r"""subphone_features
    Convert HTK-style question into regular expression for searching labels.
    If convert_number_pattern, keep the following sequences unescaped for
    extracting continuous values):
    (\d+)       -- handles digit without decimal point
    ([\d\.]+)   -- handles digits with and without decimal point
    ([-\d]+)    -- handles positive and negative numbers
    """

    # handle HTK wildcards (and lack of them) at ends of label:
    prefix = ""
    postfix = ""
    if "*" in question:
        if not question.startswith("*"):
            prefix = "\\A"
        if not question.endswith("*"):
            postfix = "\\Z"
    question = question.strip("*")
    question = re.escape(question)
    # convert remaining HTK wildcards * and ? to equivalent regex:
    question = question.replace("\\*", ".*")
    question = prefix + question + postfix

    if convert_number_pattern:
        question = question.replace("\\(\\\\d\\+\\)", "(\\d+)")
        question = question.replace("\\(\\[\\-\\\\d\\]\\+\\)", "([-\\d]+)")
        question = question.replace("\\(\\[\\\\d\\\\\\.\\]\\+\\)", "([\\d\\.]+)")
    # NOTE: singing voice synthesis specific handling
    if convert_svs_pattern:
        question = question.replace(
            "\\(\\[A\\-Z\\]\\[b\\]\\?\\[0\\-9\\]\\+\\)", "([A-Z][b]?[0-9]+)"
        )
        question = question.replace("\\(\\\\NOTE\\)", "([A-Z][b]?[0-9]+)")
        question = question.replace("\\(\\[pm\\]\\\\d\\+\\)", "([pm]\\d+)")

    return question


def load_question_set(qs_file_name, append_hat_for_LL=True, convert_svs_pattern=True):
    """Load HTS-style question and convert it to binary/continuous feature
    extraction regexes.

    This code was taken from Merlin.

    Args:
        qs_file_name (str): Input HTS-style question file path
        append_hat_for_LL (bool): Append ^ for LL regex search.
            Note that the most left context is assumed to be phoneme identity
            before the previous phoneme (i.e. LL-xx). This parameter should be False
            for the HTS-demo_NIT-SONG070-F001 demo.
        convert_svs_pattern (bool): Convert SVS specific patterns.

    Returns:
        (binary_dict, numeric_dict): Binary/numeric feature extraction
        regexes.

    Examples:
        >>> from nnmnkwii.io import hts
        >>> from nnmnkwii.util import example_question_file
        >>> binary_dict, numeric_dict = hts.load_question_set(example_question_file())
    """
    with open(qs_file_name) as f:
        lines = f.readlines()
    binary_qs_index = 0
    continuous_qs_index = 0
    binary_dict = {}
    numeric_dict = {}

    LL = re.compile(re.escape("LL-"))

    for line in lines:
        line = line.replace("\n", "")
        temp_list = line.split()
        if len(line) <= 0 or line.startswith("#"):
            continue
        name = temp_list[1].replace('"', "").replace("'", "")
        temp_list = line.split("{")
        temp_line = temp_list[1]
        temp_list = temp_line.split("}")
        temp_line = temp_list[0]
        temp_line = temp_line.strip()
        question_list = temp_line.split(",")

        temp_list = line.split(" ")
        question_key = temp_list[1]
        if temp_list[0] == "CQS":
            assert len(question_list) == 1
            processed_question = wildcards2regex(
                question_list[0],
                convert_number_pattern=True,
                convert_svs_pattern=convert_svs_pattern,
            )
            numeric_dict[continuous_qs_index] = (
                name,
                re.compile(processed_question),
            )  # save pre-compiled regular expression
            continuous_qs_index = continuous_qs_index + 1
        elif temp_list[0] == "QS":
            re_list = []
            for temp_question in question_list:
                processed_question = wildcards2regex(temp_question)
                if (
                    append_hat_for_LL
                    and LL.search(question_key)
                    and processed_question[0] != "^"
                ):
                    processed_question = "^" + processed_question
                re_list.append(re.compile(processed_question))

            binary_dict[binary_qs_index] = (name, re_list)
            binary_qs_index = binary_qs_index + 1
        else:
            raise RuntimeError("Not supported question format")
    return binary_dict, numeric_dict


def write_audacity_labels(dst_path, labels):
    """Write audacity labels from HTS-style labels

    Args:
        dst_path (str): The output file path.
        labels (HTSLabelFile): HTS style labels
    """
    with open(dst_path, "w") as of:
        for s, e, l in labels:
            s, e = s * 1e-7, e * 1e-7
            if "-" in l and "+" in l:
                ph = l.split("-")[1].split("+")[0]
            else:
                ph = l
            of.write("{:.4f}\t{:.4f}\t{}\n".format(s, e, ph))


def write_textgrid(dst_path, labels):
    """Write TextGrid from HTS-style labels

    Args:
        dst_path (str): The output file path.
        labels (HTSLabelFile): HTS style labels
    """
    template = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {xmax}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phoneme"
        xmin = 0
        xmax = {xmax}
        intervals: size = {size}"""
    template = template.format(xmax=labels.end_times[-1] * 1e-7, size=len(labels))

    for idx, (s, e, l) in enumerate(labels):
        s, e = s * 1e-7, e * 1e-7
        if "-" in l and "+" in l:
            ph = l.split("-")[1].split("+")[0]
        else:
            ph = l

        template += """
        intervals [{idx}]:
            xmin = {s}
            xmax = {e}
            text = "{ph}" """.format(
            idx=idx + 1, s=s, e=e, ph=ph
        )
    template += "\n"

    with open(dst_path, "w") as of:
        of.write(template)
