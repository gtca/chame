import numpy as np
from chame.tl import chromvar


def test_compute_deviations():
    # Adopted from:
    # chromVAR/tests/testthat/test_compute_deviations.R

    # obs x peaks
    counts = np.array([[0, 1, 2, 0], [1, 1, 0, 1], [2, 0, 1, 1]]).T
    # TFs x peaks
    anno_mx = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]]).astype(bool).T
    anno_list = [np.where(x) for x in anno_mx]
    # niter x peaks
    bg = np.array([[1, 1, 1], [0, 0, 2], [1, 1, 0]]).T

    true_deviations = np.array(
        [
            [0.1728395, -0.4444444, 0.1728395, -0.07407407],
            [-0.2910053, 0.3174603, 0.2116402, -0.19841270],
            [0.7407407, -0.6349206, -0.7407407, 0.63492063],
        ]
    ).T
    true_z = np.array(
        [
            [1.1547005, -1.1547005, 1.1547005, -1.1547005],
            [-0.5773503, 0.5773503, 0.5773503, -0.5773503],
            [3.2331615, -1.1547005, -4.0414519, 9.2376043],
        ]
    ).T

    expectation = chromvar._compute_expectations_core(counts)
    dev = chromvar._compute_deviations_core(counts, anno_mx, bg, expectation)

    assert np.allclose(dev["deviations"], true_deviations, equal_nan=True)
    assert np.allclose(dev["z"], true_z, equal_nan=True)
