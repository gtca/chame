import os
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import anndata
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
        If to load raw counts matrix (False by default).
        Only works when the root outs/ directory itself is provided as path.
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
    path = Path(path)
    files = list(path.iterdir())

    # All output files can have the same dataset-specific prefix
    prefix = ""
    if len(os.path.commonprefix(files)) > 0:
        # Prefix should be defined so that filtered/raw counts
        # are identifiable when present
        is_matrix_file = [
            f.stem.endswith("_peak_bc_matrix") for f in files
        ]
        matrix_files = [files[i] for i in np.where(is_matrix_file)[0]]
        if len(matrix_files) > 0:
            mf = matrix_files[0].stem
            if mf.endswith(filtered_suffix := "filtered_peak_bc_matrix"):
                prefix = mf[: (len(mf) - len(filtered_suffix))]
            elif mf.endswith(raw_suffix := "raw_peak_bc_matrix"):
                prefix = mf[: (len(mf) - len(raw_suffix))]
            else:
                raise ValueError("No filtered or raw peak matrix found")
        else:
            raise ValueError("No matrix found")

        files = [f.name.removeprefix(prefix) for f in files]

    # Counts
    if raw:
        if (counts_file := "raw_peak_bc_matrix.h5") in files:
            adata = read_10x_h5(
                path / (prefix + counts_file), *args, **kwargs
            )
        elif (counts_file := "matrix.mtx") in files or (counts_file := "matrix.mtx.gz") in files:
            path_mtx = path
            if not Path(path_mtx).stem != "raw_peak_bc_matrix":
                path_mtx = path / (prefix + "raw_peak_bc_matrix")
            adata = read_10x_mtx(
                path_mtx,
                *args,
                **kwargs
            )
        else:
            raise NotImplementedError
    else:
        if (counts_file := "filtered_peak_bc_matrix.h5") in files:
            adata = read_10x_h5(
                path / (prefix + counts_file), *args, **kwargs
            )
        elif (counts_file := "matrix.mtx") in files or (counts_file := "matrix.mtx.gz") in files:
            path_mtx = path
            if Path(path_mtx).stem != "filtered_peak_bc_matrix":
                path_mtx = path / (prefix + "filtered_peak_bc_matrix")
            adata = read_10x_mtx(
                path_mtx,
                *args,
                **kwargs
            )
        else:
            raise NotImplementedError
    if "gene_ids" in adata.var:
        adata.var.rename(columns={"gene_ids": "peak_ids"}, inplace=True)

    # Summary
    summary_ext = summary.strip(".").lower()
    if (summary_file := f"summary.{summary_ext}") in files:
        summary = read_10x_summary(path / (prefix + summary_file))
        adata.uns["summary"] = summary

    # TFs
    if raw:
        if (tf_file := "raw_tf_bc_matrix.h5") in files:
            adata_tf = read_10x_tf_h5(path / (prefix + tf_file))
            adata.obsm["tf"] = pd.DataFrame.sparse.from_spmatrix(
                adata_tf.X,
                index=adata_tf.obs_names.values,
                columns=adata_tf.var_names.values,
            )
    else:
        if (tf_file := "filtered_tf_bc_matrix.h5") in files:
            adata_tf = read_10x_tf_h5(path / (prefix + tf_file))
            adata.obsm["tf"] = pd.DataFrame.sparse.from_spmatrix(
                adata_tf.X,
                index=adata_tf.obs_names.values,
                columns=adata_tf.var_names.values,
            )

    # Peak annotation
    if (peak_annotation := "peak_annotation.tsv") in files:
        peak_annotation = read_10x_peak_annotation(
            path / (prefix + peak_annotation), backend=table_backend
        )
        if "atac" not in adata.uns:
            adata.uns["atac"] = dict()
        adata.uns["atac"]["peak_annotation"] = peak_annotation

    if (fragments := "fragments.tsv.gz") in files:
        if "files" not in adata.uns:
            adata.uns["files"] = dict()
        adata.uns["files"]["fragments"] = str(path / (prefix + fragments))

    if (peaks_bed := "peaks.bed") in files:
        peaks = read_10x_peaks_bed(
            path / (prefix + peaks_bed), backend=table_backend
        )
        # Support more backends than just pandas
        for col in "chrom", "start", "end":
            adata.var[col] = peaks[col].to_numpy()

    if (peak_motif_mapping := "peak_motif_mapping.bed") in files:
        mapping = read_10x_peak_motif_mapping(
            path / (prefix + peak_motif_mapping), backend=table_backend
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
        pa.loc[pa.peak_type.isnull(), "peak_type"] = ""

        # If peak name is not in the annotation table, reconstruct it:
        # peak = chrom:start-end
        if "peak" not in pa.columns:
            if "chrom" in pa.columns and "start" in pa.columns and "end" in pa.columns:
                pa["peak"] = (
                    pa["chrom"] + ":" + pa["start"].astype(str) + "-" + pa["end"].astype(str)
                )
            else:
                raise AttributeError(
                    "Peak annotation does not in contain neighter peak column nor chrom, start, and end columns."
                )

        # Split genes, distances, and peaks into individual records
        pa_g = pd.DataFrame(pa.gene.str.split(";").tolist(), index=pa.peak).stack()
        # Handle distance column properly - convert to string only for splitting
        # but preserve NaN values
        distance_series = pa.distance.copy().astype(str).replace('nan', np.nan)
        pa_d = pd.DataFrame(
            distance_series.fillna("").str.split(";").tolist(), index=pa.peak
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
        null_distance = (pa_long.distance == "") | pd.isna(pa_long.distance)
        pa_long.loc[null_distance, "distance"] = 0
        # Convert non-null values to integers
        pa_long.distance = pd.to_numeric(pa_long.distance, errors='coerce').fillna(0).astype(int)
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
    filename: PathLike, backend: Literal["pandas", "polars"] = "pandas"
) -> pd.DataFrame:
    if backend == "pandas":
        bed = pd.read_csv(filename, sep="\t", comment="#", header=None)

    elif backend == "polars":
        from polars import read_csv

        bed = read_csv(filename, separator="\t", comment_char="#", has_header=False)

    elif backend == "arrow":
        from pyarrow import csv

        bed = csv.read_csv(filename, parse_options=csv.ParseOptions(delimiter="\t", comment="#"))

    else:
        raise NotImplementedError(
            f"Support for backend {backend} has not been implemented yet"
        )

    if bed.shape[1] >= 3:
        bed.columns = ["chrom", "start", "end"] + [
            f"column_{i+4}" for i in range(bed.shape[1] - 3)
        ]
    else:
        raise ValueError(f"BED file {filename} does not have required columns.")

    return bed


def read_10x_peak_motif_mapping(
    filename: PathLike, backend: Literal["pandas", "polars", "arrow"] = "pandas"
) -> pd.DataFrame:
    if backend == "pandas":
        bed = pd.read_csv(filename, sep="\t", comment="#", header=None)
        bed.columns = ["chrom", "start", "end", "motif"]
        bed["peak"] = bed["chrom"] + ":" + bed["start"].astype(str) + "-" + bed["end"].astype(str)
        bed = bed.set_index("peak")

    elif backend == "polars":
        from polars import read_csv

        bed = read_csv(filename, separator="\t", comment_char="#", has_header=False)
        bed.columns = ["chrom", "start", "end", "motif"]
        bed["peak"] = bed["chrom"] + ":" + bed["start"].astype(str) + "-" + bed["end"].astype(str)

    elif backend == "arrow":
        from pyarrow import csv

        bed = csv.read_csv(filename, parse_options=csv.ParseOptions(delimiter="\t", comment="#"), columns=["chrom", "start", "end", "motif"])
        bed.columns = ["chrom", "start", "end", "motif"]
        bed["peak"] = bed["chrom"] + ":" + bed["start"].astype(str) + "-" + bed["end"].astype(str)

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


def read_10x_mtx(
    path: PathLike,
    feature_type: Literal["Peaks", "Motifs"] | None = "Peaks",
    feature_file: Literal["peaks.bed", "motifs.tsv"] | None = None,
    var_names: Literal["ids", "names"] = "ids",
    make_unique: bool = True,
    backend: Literal["pandas", "polars", "arrow"] = "pandas",
    *args,
    **kwargs,
) -> AnnData:
    """Read 10x-Genomics-formatted mtx directory for ATAC data.

    Parameters
    ----------
    path
        Path to directory containing the `.mtx` and `.tsv` files,
        e.g. './filtered_peak_bc_matrix/'.
    feature_type
        If to only read a certain feature type such as Peaks or Motifs,
        use None to read all feature types (Peaks by default)
    feature_file
        Name of the file containing feature information, e.g. 'peaks.bed' or 'motifs.tsv'.
    var_names
        The variables index: 'ids' or 'names' ('ids' by default -> 'peak_ids' or 'motif_ids')
    make_unique
        Whether to make the variables index unique by appending '-1', '-2' etc.
    backend
        Which library to use to read tables (pandas, polars, or arrow).
        Currently only pandas is supported.
    
    Returns
    -------
    AnnData object
    """
    from scipy.io import mmread

    if backend not in ["pandas", "polars", "arrow"]:
        raise NotImplementedError(f"Backend {backend} not supported")

    if var_names not in ["ids", "names"]:
        raise ValueError(f"Invalid value for var_names: {var_names}, expected 'ids' or 'names'")

    # Read matrix.mtx file
    mtx_file = Path(path) / "matrix.mtx"
    if not mtx_file.exists():
        mtx_file = mtx_file.with_suffix(".mtx.gz")
        if not mtx_file.exists():
            raise FileNotFoundError(f"Matrix file {mtx_file} not found")

    # Use mmread to read the sparse matrix
    X = mmread(mtx_file).T.tocsr()  # transpose to have cells as rows

    # Read barcodes.tsv file
    barcodes_file = Path(path) / "barcodes.tsv"
    if not barcodes_file.exists():
        barcodes_file = barcodes_file.with_suffix(".tsv.gz")
        if not barcodes_file.exists():
            raise FileNotFoundError(f"Barcodes file {barcodes_file} not found")

    if backend == "pandas":
        barcodes = pd.read_csv(barcodes_file, header=None).iloc[:, 0].values
    elif backend == "polars":
        import polars as pl
        barcodes = pl.read_csv(barcodes_file, separator="\t", has_header=False).columns[0].to_numpy()
    elif backend == "arrow":
        from pyarrow import csv
        barcodes = csv.read_csv(barcodes_file, parse_options=csv.ParseOptions(delimiter="\t"), columns=[0]).to_numpy()

    # Read features (peaks or motifs)
    if feature_file is None:
        possible_feature_files = "peaks.bed", "motifs.tsv", "peaks.bed.gz", "motifs.tsv.gz", "features.tsv", "genes.tsv", "features.tsv.gz", "genes.tsv.gz"
        for fname in possible_feature_files:
            feature_file_path = Path(path) / fname
            if feature_file_path.exists() or feature_file_path.with_suffix(".gz").exists():
                feature_file = fname
                break
        else:
            raise FileNotFoundError(f"Could not find feature file. Tried {', '.join(possible_feature_files)}")

    feature_is_bed = feature_file.endswith(".bed") or feature_file.endswith(".bed.gz")
    feature_is_tsv = feature_file.endswith(".tsv") or feature_file.endswith(".tsv.gz")

    features_path = Path(path) / feature_file
    if not features_path.exists():
        features_path = features_path.with_suffix(".gz")
        if not features_path.exists():
            raise FileNotFoundError(f"Feature file {features_path} not found")

    feature_types = None
    if feature_is_bed:
        # Read peaks.bed file (chr, start, end)
        features = read_10x_peaks_bed(features_path, backend=backend)

        # Create peak names in the format chrom:start-end
        peak_names = features["chrom"] + ":" + features["start"].astype(str) + "-" + features["end"].astype(str)
        feature_ids = peak_names.values
        feature_names = peak_names.values
        feature_types = np.array(["Peaks"] * len(feature_ids))

    elif feature_is_tsv:
        # Read motifs.tsv file (id, name)
        if backend == "pandas":
            features = pd.read_csv(features_path, sep="\t", header=None)
            if features.shape[1] >= 2:
                feature_ids = features[0].values
                feature_names = features[1].values
                feature_types = np.array(["Motifs"] * len(feature_ids))
            else:
                raise ValueError(f"TSV file {feature_file} does not have required columns.")
        elif backend == "polars":
            import polars as pl
            features = pl.read_csv(features_path, separator="\t", has_header=False)

        if backend == "polars" or backend == "arrow":
            if features.shape[1] >= 2:
                feature_ids = features[0]
                feature_names = features[1]
                feature_types = np.array(["Motifs"] * len(feature_ids))
            else:
                raise ValueError(f"TSV file {feature_file} does not have required columns.")
    else:
        # Check if it's a legacy format with features.tsv or genes.tsv
        for fname in "features.tsv", "genes.tsv":
            if os.path.exists(features_path) or os.path.exists(features_path + ".gz"):
                if os.path.exists(features_path + ".gz"):
                    features_path = features_path + ".gz"
                features = pd.read_csv(features_path, sep="\t", header=None)
                feature_ids = features[0].values
                feature_names = features[1].values

                # Check if we have feature_types column
                if features.shape[1] >= 3:
                    feature_types = features[2].values

    # NOTE: The code below only supports pandas at the moment

    # Make unique variable names if needed
    if make_unique:
        if var_names == "ids":
            var_names_idx = pd.Index(feature_ids)
            var_names_idx = anndata.utils.make_index_unique(var_names_idx)
            feature_ids = var_names_idx.values
        elif var_names == "names":
            var_names_idx = pd.Index(feature_names)
            var_names_idx = anndata.utils.make_index_unique(var_names_idx)
            feature_names = var_names_idx.values

    # Create var dataframe with feature information
    if backend == "pandas":
        var = pd.DataFrame(index=feature_ids)

    if feature_file.startswith("peaks.bed"):
        pass
    elif feature_file.startswith("motifs.tsv"):
        if var_names == "ids":
            var["motif_names"] = feature_names
        elif var_names == "names":
            var["motif_ids"] = feature_ids
    else:
        if var_names == "ids":
            var["feature_names"] = feature_names
        elif var_names == "names":
            var["feature_ids"] = feature_ids

    if feature_types is not None:
        var["feature_types"] = feature_types

    # Create observation dataframe with barcode information
    obs = pd.DataFrame(index=barcodes)

    # Create AnnData object
    adata = AnnData(X=X, obs=obs, var=var)

    # Filter by feature_type if requested
    if feature_type is not None and "feature_types" in adata.var:
        feature_mask = adata.var["feature_types"] == feature_type
        if feature_mask.any():
            adata = adata[:, feature_mask].copy()

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
        
        # NOTE: dtype is deprecated in newer anndata versions
        adata = AnnData(
            matrix,
            obs=barcodes,
            var=features,
        )

    return adata


def read_10x_summary(
    filename: PathLike,
) -> dict[str, Any]:
    """Read summary.csv or summary.json
    in the Cell Ranger (ATAC) output folder.

    Args:
      filename: str
        Path to summary.csv or summary.json

    Returns:
      Dictionary with summary information.
    """
    filename = Path(filename)
    ext = filename.suffix
    if ext == ".csv":
        summary = pd.read_csv(filename).T.to_dict()[0]
    elif ext == ".json":
        import json

        with filename.open() as f:
            summary = json.load(f)
    else:
        raise NotImplementedError(
            f"Reading {ext} file with summary has not been defined"
        )
    return summary
