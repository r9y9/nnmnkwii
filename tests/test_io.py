from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from os.path import dirname, join
import copy

DATA_DIR = join(dirname(__file__), "data")


def test_state_alignment_label_file():
    input_state_label = join(DATA_DIR, "label_state_align", "arctic_a0001.lab")
    labels = hts.load(input_state_label)
    with open(input_state_label) as f:
        assert f.read() == str(labels)

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
