import os
from os import PathLike
from typing import Optional, Literal, Any, Dict
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def read_10x(
    path: PathLike,
    raw: bool = False,
    atac_only: bool = True,
    summary: Literal["csv", "json"] = "csv",
    *args,
    **kwargs,
) -> AnnData:
    """Read 10x Genomics Cell Ranger ATAC output folder.

    Args:
      path: str
        Path to the outs/ directory
      raw (optional): bool
        If to load raw counts matrix (False by default)
      atac_only (optional): bool
        If to only read Peaks feature type (True by default)
      summary (optional): 'csv' or 'json'
        If to load summary.csv of summary.json ('csv' by default)

    Returns:
      AnnData object.
    """
    file_map = {
        "filtered_peak_bc_matrix.h5": read_10x_h5,
    }
    files = os.listdir(path)

    if len(prefix := os.path.commonprefix(files)) > 0:
        files = [f.removeprefix(prefix) for f in files]

    # Counts
    if raw:
        raise NotImplementedError
    else:
        if (counts_file := "filtered_peak_bc_matrix.h5") in files:
            adata = read_10x_h5(
                os.path.join(path, prefix + counts_file), *args, **kwargs
            )
        else:
            # TODO: read_10x_mtx
            raise NotImplementedError

    # Summary
    summary_ext = summary.strip(".").lower()
    if (summary_file := f"summary.{summary_ext}") in files:
        summary = _read_10x_summary(os.path.join(path, prefix + summary_file))
        adata.uns["summary"] = summary

    # TFs
    if (tf_file := "filtered_tf_bc_matrix.h5") in files:
        adata_tf = read_10x_tf_h5(os.path.join(path, prefix + tf_file))
        adata.obsm["tf"] = pd.DataFrame.sparse.from_spmatrix(
            adata_tf.X,
            index=adata_tf.obs_names.values,
            columns=adata_tf.var_names.values,
        )

    # TODO: peaks annotation
    # TODO: fragments file
    # TODO: peak-motif mapping
    # TODO: peaks bed file
    # TODO: genome file

    return adata


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
        adata = adata[
            :, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))
        ].copy()
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
    adata = sc.read_10x_mtx(path, gex_only=False, *args, **kwargs)
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
            dtype=matrix.dtype,
        )

    return adata


def _read_10x_summary(
    filename: PathLike,
) -> Dict[str, Any]:
    """Read summary.csv or summary.json
    in the Cell Ranger (ATAC) output folder.

    Args:
      filename: str
        Path to summary.csv or summary.json

    Returns:
      Dictionary with summary information.
    """
    ext = os.path.splitext(filename)[-1]
    if ext == ".csv":
        summary = pd.read_csv(filename).T.to_dict()[0]
    elif ext == ".json":
        import json

        with open(filename) as f:
            summary = json.load(f)
    else:
        raise NotImplementedError(
            f"Reading {ext} file with summary has not been defined"
        )
    return summary
