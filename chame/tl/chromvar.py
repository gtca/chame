#
# ChromVAR:
#   Schep, Wu, Buenrostro & Greenleaf, 2017
#   DOI: 10.1038/nmeth.4401
#
# Original implementation in R:
#   https://github.com/GreenleafLab/chromVAR
#

from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from anndata import AnnData


def _compute_deviations_core(
    counts, peak_indices, background_peaks, expectation
) -> Dict[str, np.ndarray]:
    assert all(
        counts.sum(axis=0) > 0
    ), "There should be at least one count for each peak"
    assert (
        counts.shape[1] == background_peaks.shape[1]
    ), "Number of background peaks should match"
    assert (
        len(expectation) == counts.shape[1]
    ), "Expectation length should equal the number of peaks"

    # TODO: optimize
    deviations_list, z_list = [], []
    for row in peak_indices:
        dev = _compute_deviations_single(row, counts, background_peaks, expectation)
        deviations_list.append(dev["dev"])
        z_list.append(dev["z"])

    deviations = np.column_stack(deviations_list)
    z = np.column_stack(z_list)

    return {"deviations": deviations, "z": z}


def _compute_deviations_single(
    peak_set, counts, background_peaks, expectation, threshold=1
):
    fragments_per_sample = counts.sum(axis=1)
    tf_count = len(np.where(peak_set)[0])

    if tf_count == 1:
        raise NotImplementedError()

    tf_vec = lil_matrix((counts.shape[1], 1))
    tf_vec[peak_set] = 1

    observed = (counts @ tf_vec).reshape(1, -1)
    expected = (expectation @ tf_vec).reshape(-1, 1) @ fragments_per_sample.reshape(
        1, -1
    )
    observed_deviation = ((observed - expected) / expected).squeeze()

    niter = background_peaks.shape[0]
    # sample_mx = csr_matrix(([], ([], [])), shape=(niter,counts.shape[1]))
    sample_list = []

    # TODO: optimize
    for peak in range(counts.shape[1]):
        sample_list.append((background_peaks[:, peak_set] == peak).sum(axis=1))
    sample_mx = csr_matrix(np.column_stack(sample_list))

    sampled = counts @ sample_mx.T
    sampled_expected = (expectation @ sample_mx.T).reshape(
        -1, 1
    ) @ fragments_per_sample.reshape(1, -1)
    sampled_deviation = (sampled.T - sampled_expected) / sampled_expected

    bg_overlap = (sample_mx @ tf_vec) / tf_count

    fail_filter = np.where(expected < threshold)[0]

    mean_sampled_deviation = sampled_deviation.mean(axis=0)
    sd_sampled_deviation = sampled_deviation.std(axis=0, ddof=1)

    normdev = observed_deviation - mean_sampled_deviation
    z = normdev / sd_sampled_deviation

    if len(fail_filter) > 0:
        z[fail_filter] = np.nan
        normdev[fail_filter] = np.nan

    return {
        "z": z,
        "dev": normdev,
        "matches": tf_count / counts.shape[1],
        "overlap": bg_overlap,
    }


def _compute_expectations_core(counts, norm=False, group=None) -> np.ndarray:
    if group is not None:
        raise NotImplementedError()
    if norm is not False:
        raise NotImplementedError()

    return counts.sum(axis=0) / counts.sum()
