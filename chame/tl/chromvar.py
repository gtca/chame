#
# ChromVAR:
#   Schep, Wu, Buenrostro & Greenleaf, 2017
#   DOI: 10.1038/nmeth.4401
#
# Original implementation in R:
#   https://github.com/GreenleafLab/chromVAR
#

from typing import Dict, Literal, List, Union

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData

from logging import log


def chromvar(
    adata: AnnData,
    annotations,
    bias: Union[str, List[str], np.ndarray] = "gc",
    background_peaks=None,
    expectation=None,
    dev_key: str = "deviations",
    z_key: str = "z",
    norm: bool = False,
    group: str = None,
    threshold: float = 1,
):

    # TODO: implement an option to use layers
    counts = adata.X

    if bias is None:
        log.info(f"Using counts per peak as bias as none provided.")
        bias = counts.sum(axis=0)

    if background_peaks is None:
        if isinstance(bias, str):
            try:
                bias = adata.var[bias]
            except KeyError:
                raise KeyError(f"Calculate GC bias per peak with chame.util.seq.count_gc "
                               "or provide it as an array")
        background_peaks = _get_background_peaks(counts, bias)

    if expectation is None:
        expectation = _compute_expectations_core(counts, norm=norm, group=group)

    dev = _compute_deviations_core(
        counts,
        annotations,
        background_peaks=background_peaks,
        expectation=expectation,
        threshold=threshold,
    )

    adata.obsm[dev_key] = dev["deviations"]
    log.info(f"Added key '{dev_key}' to adata.obsm.")

    adata.obsm[z_key] = dev["z"]
    log.info(f"Added key '{z_key}' to adata.obsm.")


def _compute_deviations_core(
    counts,
    peak_indices,
    background_peaks,
    expectation,
    threshold=1,
) -> Dict[str, np.ndarray]:
    assert all(
        counts.sum(axis=0) > 0
    ), "There should be at least one count for each peak"
    assert (
        counts.shape[1] == background_peaks.shape[0]
    ), "Number of background peaks should match"
    assert (
        len(expectation) == counts.shape[1]
    ), "Expectation length should equal the number of peaks"

    # TODO: optimize
    deviations_list, z_list = [], []
    dev = np.apply_along_axis(
        lambda row: _compute_deviations_single(
            row, counts, background_peaks, expectation, threshold=threshold
        ),
        axis=1,
        arr=peak_indices,
    )

    deviations_list = [d["dev"] for d in dev]
    z_list = [d["z"] for d in dev]

    deviations = np.column_stack(deviations_list)
    z = np.column_stack(z_list)

    return {"deviations": deviations, "z": z}


def _compute_deviations_single(
    peak_set, counts, background_peaks, expectation, threshold=1
):
    if len(peak_set) == counts.shape[1]:
        peak_set = np.where(peak_set)[0]

    fragments_per_sample = counts.sum(axis=1)
    tf_count = len(peak_set)

    tf_vec = lil_matrix((counts.shape[1], 1))
    tf_vec[peak_set] = 1

    observed = (counts @ tf_vec).reshape(1, -1)
    expected = (expectation @ tf_vec).reshape(-1, 1) @ fragments_per_sample.reshape(
        1, -1
    )
    observed_deviation = ((observed - expected) / expected).squeeze()

    niter = background_peaks.shape[1]
    # sample_mx = csr_matrix(([], ([], [])), shape=(niter,counts.shape[1]))
    sample_list = []

    # TODO: optimize
    for peak in range(counts.shape[1]):
        sample_list.append(
            (background_peaks[peak_set, :] == peak).sum(axis=0).squeeze()
        )
    sample_mx = csr_matrix(np.column_stack(sample_list))

    sampled = counts @ sample_mx.T
    sampled_expected = (expectation @ sample_mx.T).reshape(
        -1, 1
    ) @ fragments_per_sample.reshape(1, -1)
    sampled_deviation = (sampled.T - sampled_expected) / sampled_expected

    bg_overlap = np.mean(sample_mx @ tf_vec) / tf_count

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


def _get_background_peaks(
    counts, bias, niterations: int = 50, w: float = 0.1, bs: int = 50
):
    """
    Get a set of background peaks for each peak based on GC content
    and number of fragments across all cells
    """
    fragments_per_peak = counts.sum(axis=0)
    assert bias.shape[0] == len(fragments_per_peak)

    if np.min(fragments_per_peak) <= 0:
        raise ValueError("All peaks should have at least 1 count across all cells")

    intensity = np.log10(fragments_per_peak)
    norm_mx = np.column_stack([intensity, bias])

    chol_cov_mx = np.linalg.cholesky(np.cov(norm_mx, rowvar=False)).T
    trans_norm_mx = np.linalg.solve(chol_cov_mx.T, norm_mx.T).T

    # Make bins
    bins1 = np.linspace(
        np.min(trans_norm_mx[:, 0]), np.max(trans_norm_mx[:, 0]), num=bs
    )
    bins2 = np.linspace(
        np.min(trans_norm_mx[:, 1]), np.max(trans_norm_mx[:, 1]), num=bs
    )

    bin_data = np.meshgrid(bins1, bins2)
    bin_data = np.column_stack([bin_data[0].flatten(), bin_data[1].flatten()])

    bin_dist = np.linalg.norm(bin_data[:, None, :] - bin_data[None, :, :], axis=-1)

    dist = stats.norm(loc=0, scale=w)
    bin_p = dist.pdf(bin_dist)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(bin_data)
    _, bin_membership = nbrs.kneighbors(trans_norm_mx)
    bin_membership = bin_membership.squeeze()

    bin_density = np.bincount(bin_membership, minlength=bs**2)

    n = len(bin_membership)
    out = np.zeros(shape=(n, niterations))
    for i in range(len(bin_density)):
        ix = np.where(bin_membership == i)[0]
        if len(ix) == 0:
            continue
        p = bin_p[:, i][bin_membership] / bin_density[bin_membership]
        p /= p.sum()
        s = np.random.choice(np.arange(n), size=niterations * len(ix), p=p)
        for j in range(len(ix)):
            out[ix[j], :] = s[j * niterations : (j + 1) * niterations]

    return out
