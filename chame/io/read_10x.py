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
    peaks_only: bool = True,
    summary: Literal["csv", "json"] = "csv",
    matrix_backend: Literal["numpy"] = "numpy",
    table_backend: Literal["pandas"] = "pandas",
    *args,
    **kwargs,
) -> AnnData:
    """Read 10x Genomics Cell Ranger ATAC output folder.

    Args:
      path: str
        Path to the outs/ directory
      raw (optional): bool
        If to load raw counts matrix (False by default)
      peaks_only (optional): bool
        If to only read Peaks feature type (True by default)
      summary (optional): 'csv' or 'json'
        If to load summary.csv of summary.json ('csv' by default)
      matrix_backend (optional): 'numpy'
        Which library to use to read matrices (arrays)
      table_backend (optional): 'pandas', 'polars'
        Which library to use to read tables (data frames)

    Returns:
      AnnData object.
    """
    file_map = {
        "filtered_peak_bc_matrix.h5": read_10x_h5,
    }
    files = os.listdir(path)

    # All output files can have the same dataset-specific prefix
    prefix = ""
    if len(os.path.commonprefix(files)) > 0:
        # Prefix should be defined so that filtered/raw counts
        # are identifiable when present
        is_matrix_file = [
            os.path.splitext(f)[0].endswith("_peak_bc_matrix") for f in files
        ]
        matrix_files = [files[i] for i in np.where(is_matrix_file)[0]]
        if len(matrix_files) > 0:
            mf = os.path.splitext(matrix_files[0])[0]
            if mf.endswith(filtered_suffix := "filtered_peak_bc_matrix"):
                prefix = mf[: (len(mf) - len(filtered_suffix))]
            elif mf.endswith(raw_suffix := "raw_peak_bc_matrix"):
                prefix = mf[: (len(mf) - len(raw_suffix))]
            else:
                raise ValueError("No filtered or raw peak matrix found")
        else:
            raise ValueError("No matrix found")

        # TODO: Use f.removeprefix since 3.9
        files = [f[len(prefix) :] for f in files]

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

    # Peak annotation
    if (peak_annotation := "peak_annotation.tsv") in files:
        peak_annotation = read_10x_peak_annotation(
            os.path.join(path, prefix + peak_annotation), backend=table_backend
        )
        if "atac" not in adata.uns:
            adata.uns["atac"] = dict()
        adata.uns["atac"]["peak_annotation"] = peak_annotation

    if (fragments := "fragments.tsv.gz") in files:
        if "files" not in adata.uns:
            adata.uns["files"] = dict()
        adata.uns["files"]["fragments"] = os.path.join(path, prefix + fragments)

    if (peaks_bed := "peaks.bed") in files:
        peaks = read_10x_peaks_bed(
            os.path.join(path, prefix + peaks_bed), backend=table_backend
        )
        # Support more backends than just pandas
        for col in "Chromosome", "Start", "End":
            adata.var[col] = peaks[col].to_numpy()

    if (peak_motif_mapping := "peak_motif_mapping.bed") in files:
        mapping = read_10x_peak_motif_mapping(
            os.path.join(path, prefix + peak_motif_mapping), backend=table_backend
        )
        if "atac" not in adata.uns:
            adata.uns["atac"] = dict()
        adata.uns["atac"]["peak_motifs_mapping"] = mapping

    # TODO: genome file

    return adata


# TODO: support v1 files as well
# TODO: consider awkward array support when it's there
def read_10x_peak_annotation(
    filename: PathLike, sep: str = "\t", backend: Literal["pandas"] = "pandas"
) -> pd.DataFrame:
    """
    Parse peak annotation file

    Parameters
    ----------
    filename
            A path to the peak annotation file (e.g. peak_annotation.tsv).
            Annotation has to contain columns: peak, gene, distance, peak_type.
    sep
            Separator for the peak annotation file. Only used if the file name is provided.
            Tab by default.
    backend
            Data frame backend such as "pandas", "polars", or "arrow".
            Currently only "pandas" is supported.
    """
    if backend == "pandas":
        pa = pd.read_csv(filename, sep=sep)

        # Convert null values to empty strings
        pa.loc[pa.gene.isnull(), "gene"] = ""
        pa.loc[pa.distance.isnull(), "distance"] = ""
        pa.loc[pa.peak_type.isnull(), "peak_type"] = ""

        # If peak name is not in the annotation table, reconstruct it:
        # peak = chrom:start-end
        if "peak" not in pa.columns:
            if "chrom" in pa.columns and "start" in pa.columns and "end" in pa.columns:
                pa["peak"] = (
                    pa["chrom"].astype(str)
                    + ":"
                    + pa["start"].astype(str)
                    + "-"
                    + pa["end"].astype(str)
                )
            else:
                raise AttributeError(
                    f"Peak annotation does not in contain neighter peak column nor chrom, start, and end columns."
                )

        # Split genes, distances, and peaks into individual records
        pa_g = pd.DataFrame(pa.gene.str.split(";").tolist(), index=pa.peak).stack()
        pa_d = pd.DataFrame(
            pa.distance.astype(str).str.split(";").tolist(), index=pa.peak
        ).stack()
        pa_p = pd.DataFrame(pa.peak_type.str.split(";").tolist(), index=pa.peak).stack()

        # Make a long dataframe indexed by gene
        pa_long = pd.concat(
            [
                pa_g.reset_index()[["peak", 0]],
                pa_d.reset_index()[[0]],
                pa_p.reset_index()[[0]],
            ],
            axis=1,
        )
        pa_long.columns = ["peak", "gene", "distance", "peak_type"]
        pa_long = pa_long.set_index("gene")

        # chrX_NNNNN_NNNNN -> chrX:NNNNN-NNNNN
        pa_long.peak = [
            peak.replace("_", ":", 1).replace("_", "-", 1) for peak in pa_long.peak
        ]

        # Make distance values integers with 0 for intergenic peaks
        # DEPRECATED: Make distance values nullable integers
        # See https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
        null_distance = pa_long.distance == ""
        pa_long.loc[null_distance, "distance"] = 0
        pa_long.distance = pa_long.distance.astype(float).astype(int)
        # DEPRECATED: Int64 is not recognized when saving HDF5 files with scanpy.write
        # pa_long.distance = pa_long.distance.astype(int).astype("Int64")
        # pa_long.distance[null_distance] = np.nan

    elif backend == "polars":
        from polars import read_csv

        pa = read_csv(filename, sep=sep)
        pa["peak"] = pa["chrom"] + ":" + pa["start"] + "-" + pa["end"]

        # Assume the long format
        pa_long = pa

    elif backend == "arrow":
        from pyarrow import csv

        pa = csv.read_csv(filename, parse_options=csv.ParseOptions(delimiter=sep))
        pa = pa.append_column(
            "peak",
            [
                pa["chrom"].to_numpy()
                + ":"
                + pa["start"].to_numpy().astype(str)
                + "-"
                + pa["end"].to_numpy().astype(str)
            ],
        )

        # Assume the long format
        pa_long = pa

    else:
        raise NotImplementedError(
            f"Support for backend {backend} has not been implemented yet"
        )

    return pa_long


def read_10x_peaks_bed(
    filename: PathLike, backend: Literal["pandas"] = "pandas"
) -> pd.DataFrame:
    if backend == "pandas":
        bed = pd.read_csv(filename, sep="\t", comment="#", header=None).iloc[:, :3]
        bed.columns = ["Chromosome", "Start", "End"]

    elif backend == "polars":
        from polars import read_csv

        bed = read_csv(filename, sep="\t", comment_char="#", has_header=False)[:, :3]
        bed.columns = ["Chromosome", "Start", "End"]

    else:
        raise NotImplementedError(
            f"Support for backend {backend} has not been implemented yet"
        )

    return bed


def read_10x_peak_motif_mapping(
    filename: PathLike, backend: Literal["pandas"] = "pandas"
) -> pd.DataFrame:
    if backend == "pandas":
        bed = pd.read_csv(filename, sep="\t", comment="#", header=None)
        bed.columns = ["Chromosome", "Start", "End", "Motif"]
        bed["Peak"] = (
            bed["Chromosome"]
            + ":"
            + bed["Start"].astype(str)
            + "-"
            + bed["End"].astype(str)
        )
        bed = bed.set_index("Peak")

    elif backend == "polars":
        from polars import read_csv

        bed = read_csv(filename, sep="\t", comment_char="#", has_header=False)
        bed.columns = ["Chromosome", "Start", "End", "Motif"]
        bed["peak"] = bed["Chromosome"] + ":" + bed["Start"] + "-" + bed["End"]

    else:
        raise NotImplementedError(
            f"Support for backend {backend} has not been implemented yet"
        )

    return bed


def read_10x_h5(
    filename: PathLike, peaks_only: bool = True, *args, **kwargs
) -> AnnData:
    """Read 10x Genomics .h5 file with features
    such as filtered_peak_bc_matrix.h5 or raw_peak_bc_matrix.h5.

    With peaks_only, this will read only the ATAC part of multimodal datasets
    (features with feature_types set to Peaks).

    Args:
      filename: str
        Path to the .h5 file
      peaks_only (optional): bool
        If to only read Peaks feature type (True by default)

    Returns:
      AnnData object.
    """
    adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)
    if peaks_only:
        # No need to copy if only peaks are present
        if len(ft := adata.var.feature_types.unique()) == 1 and ft[0] == "Peaks":
            pass
        else:
            adata = adata[
                :, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))
            ].copy()
    return adata


def read_10x_mtx(path: PathLike, peaks_only: bool = True, *args, **kwargs) -> AnnData:
    """Read 10x Genomics-formatted mtx directory
    such as filtered_peak_bc_matrix or raw_peak_bc_matrix.

    Args:
      path: str
        Path to the mtx directory
      peaks_only (optional): bool
        If to only read Peaks feature type (True by default)

    Returns:
      AnnData object.
    """
    adata = sc.read_10x_mtx(path, gex_only=False, *args, **kwargs)
    if peaks_only:
        # No need to copy if only peaks are present
        if len(ft := adata.var.feature_types.unique()) == 1 and ft[0] == "Peaks":
            pass
        else:
            adata = adata[
                :, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))
            ]
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
