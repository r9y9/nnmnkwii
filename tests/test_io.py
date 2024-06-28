import copy
import re
from os.path import dirname, join

import pytest
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.util import example_question_file

try:
    import pyopenjtalk  # noqa
except ImportError:
    pass

DATA_DIR = join(dirname(__file__), "data")


def test_labels_number_of_frames():
    # https://github.com/r9y9/nnmnkwii/issues/85
    binary_dict, numeric_dict = hts.load_question_set(join(DATA_DIR, "jp.hed"))
    labels = hts.load(join(DATA_DIR, "BASIC5000_0619.lab"))
    linguistic_features = fe.linguistic_features(
        labels, binary_dict, numeric_dict, add_frame_features=True
    )
    assert labels.num_frames() == linguistic_features.shape[0]


def test_load_question_set():
    binary_dict, numeric_dict = hts.load_question_set(example_question_file())
    assert len(binary_dict) + len(numeric_dict) == 416


def test_htk_style_question_basics():
    binary_dict, _ = hts.load_question_set(join(DATA_DIR, "test_question.hed"))
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
    LL_muon1 = binary_dict[0][1][0]
    LL_muon2 = binary_dict[1][1][0]
    L_muon1 = binary_dict[2][1][0]
    C_sil = binary_dict[3][1][0]
    R_phone_o = binary_dict[4][1][0]
    RR_phone_o = binary_dict[5][1][0]

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

    # Slice/list indexing
    assert str(labels[:2]) == str(labels[[0, 1]])


def test_singing_voice_question():
    # Test SVS case
    """
    QS "L-Phone_Yuusei_Boin"           {*^a-*,*^i-*,*^u-*,*^e-*,*^o-*}
    CQS "e1" {/E:(\\NOTE)]}
    """
    binary_dict, numeric_dict = hts.load_question_set(
        join(DATA_DIR, "test_jp_svs.hed"),
        append_hat_for_LL=False,
        convert_svs_pattern=True,
    )
    input_phone_label = join(DATA_DIR, "song070_f00001_063.lab")
    labels = hts.load(input_phone_label)
    feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
    assert feats.shape == (74, 3)

    # CQS e1: get the current MIDI number
    C_e1 = numeric_dict[0][1]
    for idx, lab in enumerate(labels):
        context = lab[-1]
        if C_e1.search(context) is not None:
            from nnmnkwii.frontend import NOTE_MAPPING

            assert NOTE_MAPPING[C_e1.findall(context)[0]] == feats[idx, 1]

    # CQS e57: get pitch diff
    # In contrast to other continous features, the pitch diff has a prefix "m" or "p"
    # to indiecate th sign of numbers.
    C_e57 = numeric_dict[1][1]
    for idx, lab in enumerate(labels):
        context = lab[-1]
        if "~p2+" in context:
            assert C_e57.search(context).group(1) == "p2"
            assert feats[idx, 2] == 2
        if "~m2+" in context:
            assert C_e57.search(context).group(1) == "m2"
            assert feats[idx, 2] == -2


def test_state_alignment_label_file():
    input_state_label = join(DATA_DIR, "label_state_align", "arctic_a0001.lab")
    labels = hts.load(input_state_label)
    with open(input_state_label) as f:
        line = f.read()
        line = line[:-1] if line[-1] == "\n" else line
        assert line == str(labels)

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
        labels.silence_phone_indices(sil_regex),
    ]:
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

    def test_invalid_start_time():
        labels = hts.HTSLabelFile()
        labels.append((100000, 0, "NG"))

    def test_succeeding_times():
        labels = hts.HTSLabelFile()
        labels.append((0, 1000000, "OK"))
        labels.append((1000000, 2000000, "OK"))

    def test_non_succeeding_times():
        labels = hts.HTSLabelFile()
        labels.append((0, 1000000, "OK"))
        labels.append((1500000, 2000000, "NG"))

    def test_non_succeeding_times_wo_strict():
        labels = hts.HTSLabelFile()
        labels.append((0, 1000000, "OK"), strict=False)
        labels.append((1500000, 2000000, "OK"), strict=False)

    with pytest.raises(ValueError):
        test_invalid_start_time()
    test_succeeding_times()
    with pytest.raises(ValueError):
        test_non_succeeding_times()
    test_non_succeeding_times_wo_strict()


# shouldn't raise RuntimeError
def test_hts_labels_contains_multiple_whitespaces():
    lab_path = join(DATA_DIR, "p225_001.lab")
    labels = hts.load(lab_path)
    print(labels)


def test_create_from_contexts():
    lab_path = join(DATA_DIR, "BASIC5000_0001.lab")
    labels = hts.load(lab_path)

    with open(lab_path) as f:
        contexts = f.readlines()

    labels2 = hts.HTSLabelFile.create_from_contexts(contexts)
    assert str(labels), str(labels2)

    def test_empty_context():
        hts.HTSLabelFile.create_from_contexts("")

    def test_empty_context2():
        contexts = pyopenjtalk.extract_fullcontext("")
        hts.HTSLabelFile.create_from_contexts(contexts)

    with pytest.raises(ValueError):
        test_empty_context()
    try:
        import pyopenjtalk  # noqa

        with pytest.raises(ValueError):
            test_empty_context2()
    except ImportError:
        pass


def test_lab_in_sec():
    labels1 = hts.load(join(DATA_DIR, "BASIC5000_0619_head.lab"))
    labels2 = hts.load(join(DATA_DIR, "BASIC5000_0619_head_sec.lab"))

    assert str(labels1) == str(labels2)
