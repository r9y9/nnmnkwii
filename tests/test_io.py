# coding: utf-8
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from os.path import dirname, join
import copy
from nnmnkwii.util import example_question_file
import re
from nose.tools import raises

DATA_DIR = join(dirname(__file__), "data")


def test_labels_number_of_frames():
    # https://github.com/r9y9/nnmnkwii/issues/85
    binary_dict, continuous_dict = hts.load_question_set(
        join(DATA_DIR, "jp.hed"))
    labels = hts.load(join(DATA_DIR, "BASIC5000_0619.lab"))
    linguistic_features = fe.linguistic_features(
        labels, binary_dict, continuous_dict, add_frame_features=True)
    assert labels.num_frames() == linguistic_features.shape[0]


def test_load_question_set():
    binary_dict, continuous_dict = hts.load_question_set(
        example_question_file())
    assert len(binary_dict) + len(continuous_dict) == 416


def test_htk_style_question_basics():
    binary_dict, continuous_dict = hts.load_question_set(
        join(DATA_DIR, "test_question.hed"))
    # sil k o n i ch i w a sil
    input_phone_label = join(DATA_DIR, "hts-nit-atr503", "phrase01.lab")
    labels = hts.load(input_phone_label)

    # Test if we can handle wildcards correctly
    # also test basic phon contexts (LL, L, C, R, RR)
    """
QS "LL-Phone_Muon1"  {sil^,pau^}    # without wildcards (*)
QS "LL-Phone_Muon2"  {sil^*,pau^*}  # with *, should be equivalent with above
QS "L-Phone_Muon1"   {*^sil-*,*^pau-*}
QS "C-Phone_sil"     {*-sil+*}
QS "R-Phone_o"       {*+o=*}
QS "RR-Phone_o"      {*=o/A:*}
    """
    LL_muon1 = binary_dict[0][0]
    LL_muon2 = binary_dict[1][0]
    L_muon1 = binary_dict[2][0]
    C_sil = binary_dict[3][0]
    R_phone_o = binary_dict[4][0]
    RR_phone_o = binary_dict[5][0]

    # xx^xx-sil+k=o
    label = labels[0][-1]
    assert LL_muon1.search(label) is None
    assert LL_muon2.search(label) is None
    assert L_muon1.search(label) is None
    assert C_sil.search(label)
    assert R_phone_o.search(label) is None
    assert RR_phone_o.search(label)

    # xx^sil-k+o=N
    label = labels[1][-1]
    assert LL_muon1.search(label) is None
    assert LL_muon2.search(label) is None
    assert L_muon1.search(label)
    assert C_sil.search(label) is None
    assert R_phone_o.search(label)
    assert RR_phone_o.search(label) is None

    # sil^k-o+N=n
    label = labels[2][-1]
    assert LL_muon1.search(label)
    assert LL_muon2.search(label)
    assert L_muon1.search(label) is None
    assert C_sil.search(label) is None
    assert R_phone_o.search(label) is None
    assert RR_phone_o.search(label) is None


def test_state_alignment_label_file():
    input_state_label = join(DATA_DIR, "label_state_align", "arctic_a0001.lab")
    labels = hts.load(input_state_label)
    with open(input_state_label) as f:
        l = f.read()
        l = l[:-1] if l[-1] == "\n" else l
        assert l == str(labels)

    print(labels.num_states())
    assert labels.num_states() == 5

    # Get and restore durations
    durations = fe.duration_features(labels)
    labels_copy = copy.deepcopy(labels)
    labels_copy.set_durations(durations)

    assert str(labels) == str(labels_copy)


def test_phone_alignment_label():
    input_state_label = join(DATA_DIR, "label_phone_align", "arctic_a0001.lab")
    labels = hts.load(input_state_label)
    assert not labels.is_state_alignment_label()


def test_label_without_times():
    input_phone_label = join(DATA_DIR, "hts-nit-atr503", "phrase01.lab")
    labels = hts.load(input_phone_label)
    assert not labels.is_state_alignment_label()


def test_mono():
    lab_path = join(DATA_DIR, "BASIC5000_0001.lab")
    labels = hts.load(lab_path)
    assert not labels.is_state_alignment_label()

    # Should detect begin/end sil regions
    sil_regex = re.compile("sil")

    for indices in [
            labels.silence_label_indices(sil_regex),
            labels.silence_phone_indices(sil_regex)]:
        assert len(indices) == 2
        assert indices[0] == 0
        assert indices[1] == len(labels) - 1


def test_hts_append():
    lab_path = join(DATA_DIR, "BASIC5000_0001.lab")
    test_labels = hts.load(lab_path)
    print("\n{}".format(test_labels))

    # should get same string representation
    labels = hts.HTSLabelFile()
    assert str(labels) == ""
    for label in test_labels:
        labels.append(label)
    assert str(test_labels) == str(labels)

    @raises(ValueError)
    def test_invalid_start_time():
        l = hts.HTSLabelFile()
        l.append((100000, 0, "NG"))

    def test_succeeding_times():
        l = hts.HTSLabelFile()
        l.append((0, 1000000, "OK"))
        l.append((1000000, 2000000, "OK"))

    @raises(ValueError)
    def test_non_succeeding_times():
        l = hts.HTSLabelFile()
        l.append((0, 1000000, "OK"))
        l.append((1500000, 2000000, "NG"))

    test_invalid_start_time()
    test_succeeding_times()
    test_non_succeeding_times()


# shouldn't raise RuntimeError
def test_hts_labels_contains_multiple_whitespaces():
    lab_path = join(DATA_DIR, "p225_001.lab")
    labels = hts.load(lab_path)
    print(labels)
