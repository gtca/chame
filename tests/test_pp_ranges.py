import numpy as np
from numpy.random import default_rng
from anndata import AnnData
import chame as ch

def setup_anndata_with_ranges(col: str = "interval", n_obs: int = 50, n_vars: int = 100) -> AnnData:
    adata = AnnData(default_rng(42).integers(10, size=(n_obs, n_vars)))
    chromosomes = default_rng(42).choice(n_vars, size=n_vars, replace=False) + 1
    starts = default_rng(42).choice(1_000_000, size=n_vars, replace=False) + 10
    sizes = default_rng(42).choice(100_000, size=n_vars, replace=False) + 100
    ranges = [f"chr{chrom}:{start}-{start+size}" for chrom, start, size in zip(chromosomes, starts, sizes)]
    adata.var[col] = ranges
    return adata


def test_ranges_slice_one_string():
    adata = setup_anndata_with_ranges()
    ranges = adata.var.interval.values
    for r in ranges:
        assert ch.pp.filter_var_by_ranges(adata, ranges=r).shape == (adata.shape[0], 1)
    assert ch.pp.filter_var_by_ranges(adata, ranges=ranges[0]).is_view


def test_ranges_slice_all_ranges():
    adata = setup_anndata_with_ranges()
    ranges = adata.var.interval.values
    assert ch.pp.filter_var_by_ranges(adata, ranges=ranges).shape == adata.shape


def test_ranges_slice_full_overlap():
    adata = setup_anndata_with_ranges()
    ranges = adata.var.interval.values
    assert ch.pp.filter_var_by_ranges(
        adata, ranges=ranges, min_var_coverage=1.0
    ).shape == adata.shape

