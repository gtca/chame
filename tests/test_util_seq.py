import numpy as np
from chame.util.seq import *


def test_sequence_to_onehot():
    xs = "ATGC"
    mx = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    assert np.array_equal(sequence_to_onehot(xs), mx)
