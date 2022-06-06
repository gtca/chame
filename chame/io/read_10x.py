from os import PathLike
from typing import Optional, Literal
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def read_10x_h5(filename: PathLike, atac_only: bool = True, *args, **kwargs) -> AnnData:
    """Read 10x Genomics .h5 file with features
    such as filtered_peak_bc_matrix.h5 or raw_peak_bc_matrix.h5.

    With atac_only, this will read only the ATAC part of multimodal datasets
    (features with feature_types set to Peaks).

    Args:
      filename: str
        Path to the .h5 file
      atac_only (optional): bool
        If to only read Peaks feature type (True by default)

    Returns:
      AnnData object.
    """
    adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)
    if atac_only:
        adata = adata[:, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))]
    return adata


def read_10x_mtx(path: PathLike, atac_only: bool = True, *args, **kwargs) -> AnnData:
    """Read 10x Genomics-formatted mtx directory
    such as filtered_peak_bc_matrix or raw_peak_bc_matrix.

    Args:
      path: str
        Path to the mtx directory
      atac_only (optional): bool
        If to only read Peaks feature type (True by default)

    Returns:
      AnnData object.
    """
    adata = sc.read_10x_mtx(filename, gex_only=False, *args, **kwargs)
    if atac_only:
        adata = adata[:, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))]
    return adata


def read_10x_tf_h5(
    filename: PathLike, var_names: Literal["id", "name"] = "name"
) -> AnnData:
    """
    Read TF x barcodes matrix from Cell Ranger ATAC output
    such as filtered_tf_bc_matrix.h5.

    Args:
      filename: str
        Path to the .h5 file
      var_names (optional): "id" or "name"
        The variables index.
    """
    import h5py
    from scipy.sparse import csr_matrix

    with h5py.File(filename) as f:
        m = f["matrix"]

        d, n = m["shape"]

        matrix = csr_matrix((m["data"], m["indices"], m["indptr"]), shape=(n, d))

        barcodes = {"obs_names": np.array(f["matrix"]["barcodes"]).astype(str)}
        features = {
            k: np.array(m["features"][k]).astype(str)
            for k in m["features"]
            if not k.startswith("_")
        }

        features["var_names"] = features.pop(var_names)

        adata = AnnData(
            matrix,
            obs=barcodes,
            var=features,
        )

    return adata
