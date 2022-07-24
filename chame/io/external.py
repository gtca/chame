from os import PathLike, path
from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import scanpy as sc
from anndata import AnnData
from mudata import MuData


def read_snap(filename: PathLike, matrix: str, bin_size: Optional[int] = None):
    """
    Read a matrix from a .snap file.

    Parameters
    ----------
    filename : str
            Path to .snap file.
    matrix : str
            Count matrix to be read, which can be
            - cell-by-peak ('peaks', 'PM'),
            - cell-by-gene ('genes', 'GM'),
            - cell-by-bin matrix ('bins', 'AM').
            In the latter case `bin_size` has to be provided.
    bin_size : Optional[int]
            Bin size, only relevant and necessary when cells x bins matrix (AM) is read.
    """

    try:
        from snaptools import snap
    except ImportError:
        raise ImportError(
            "SnapTools library is not available. Install SnapTools from PyPI (`pip install snaptools`) or from GitHub (`pip install git+https://github.com/r3fang/SnapTools`)"
        )

    from scipy.sparse import csr_matrix
    import h5py

    # Allow both PM and pm
    matrix = matrix.lower()
    assert matrix in ["pm", "gm", "am", "peaks", "genes", "bins"]
    if bin_size is not None:
        if matrix not in ["bm", "bins"]:
            warn(
                "Argument bin_size is only relevant for bins matrix (BM) and will be ignored"
            )

    f = h5py.File(filename, "r")

    if matrix == "pm" or matrix == "peaks":
        if "PM" in f:
            chrom = np.array(f["PM"]["peakChrom"]).astype(str)
            start = np.array(f["PM"]["peakStart"])
            end = np.array(f["PM"]["peakEnd"])
            idx = np.array(f["PM"]["idx"]) - 1
            idy = np.array(f["PM"]["idy"]) - 1
            count = np.array(f["PM"]["count"])

            features = (
                np.char.array(chrom)
                + ":"
                + np.char.array(start).astype("str")
                + "-"
                + np.char.array(end).astype("str")
            )
            var = pd.DataFrame(
                {"Chromosome": chrom, "Start": start, "End": end}, index=features
            )
        else:
            raise AttributeError("PM is not available in the snap file")

    elif matrix == "gm" or matrix == "genes":
        if "GM" in f:
            name = np.array(f["GM"]["name"]).astype(str)
            idx = np.array(f["GM"]["idx"]) - 1
            idy = np.array(f["GM"]["idy"]) - 1
            count = np.array(f["GM"]["count"])

            var = pd.DataFrame(index=name)
        else:
            raise AttributeError("GM is not available in the snap file")

    elif matrix == "bm" or matrix == "bins":
        if "AM" in f:
            bin_sizes = list(f["AM"]["binSizeList"])
            if bin_size is None or int(bin_size) not in bin_sizes:
                raise ValueError(
                    f"Argument bin_size has to be defined. Available bin sizes: {', '.join([str(i) for i in bin_sizes])}."
                )

            am = f["AM"][str(bin_size)]
            chrom = np.array(am["binChrom"]).astype(str)
            start = np.array(am["binStart"])
            idx = np.array(am["idx"]) - 1
            idy = np.array(am["idy"]) - 1
            count = np.array(am["count"])

            features = (
                np.char.array(chrom)
                + ":"
                + np.char.array(start - 1).astype("str")
                + "-"
                + np.char.array(start + bin_size - 1).astype("str")
            )
            var = pd.DataFrame(
                {"Chromosome": chrom, "Start": start - 1}, index=features
            )

        else:
            raise AttributeError("AM is not available in the snap file")

    f.close()

    # TODO: get barcodes manually
    bcs = snap.getBarcodesFromSnap(filename)
    obs = pd.DataFrame([bcs[i].__dict__ for i in bcs.keys()], index=bcs.keys())

    x = csr_matrix((count, (idx, idy)), shape=(obs.shape[0], var.shape[0]))

    return AnnData(X=x, obs=obs, var=var)


def read_arrow(
    filename: PathLike, fragments: bool = False, matrix: Optional[str] = None
) -> Union[AnnData, MuData]:
    """
    Read ArchR Arrow file.

    Parameters
    ----------
    filename : str
            Path to .arrow file.
    fragments : Optional[bool]
            If to read fragments as a separate modality.
            Only relevant if no matrix is provided.
    matrix : Optional[str]
            When provided, only a single matrix will be read and returned as AnnData object:
            - cell-by-tile ('tiles', 'TileMatrix'),
            - cell-by-gene ('gene_scores', 'GeneScoreMatrix'),
            - cell-by-peak matrix ('peaks', 'PeakMatrix').
            - cell-by-gene matrix ('gene_integration', 'GeneIntegrationMatrix').
    """
    import h5py

    f = h5py.File(filename)

    mdict = dict()

    # Metadata
    metadata = {k: np.array(f["Metadata"][k]) for k in f["Metadata"]}
    obs_names = metadata["CellNames"].astype(str)
    n_obs = len(obs_names)
    metadata = {
        k: (np.repeat(v[0].decode("utf-8"), n_obs) if len(v) == 1 else v)
        for k, v in metadata.items()
    }
    metadata = pd.DataFrame(metadata)
    if "CellNames" in metadata:
        metadata.CellNames = metadata.CellNames.values.astype(str)
        metadata.index = metadata.CellNames

    def sort_chromosomes(chromosomes):
        chrom_ids = chromosomes
        chrom_prefix = path.commonprefix(chromosomes)
        if len(chrom_prefix) > 0:
            chrom_ids = [chrom.replace(chrom_prefix, "") for chrom in chromosomes]
        chrom_indices = [
            j[2]
            for j in sorted(
                [
                    (float(e), str(e), i) if e.isdigit() else (float("inf"), str(e), i)
                    for i, e in enumerate(chrom_ids)
                ]
            )
        ]

        return list(np.array(chromosomes)[chrom_indices])

    # Fragments
    if "Fragments" in f and (fragments or matrix == "fragments"):
        chr_fragments = []
        chroms = [c for c in f["Fragments"] if c != "Info"]
        chroms_sorted = sort_chromosomes(chroms)
        for chrom in chroms_sorted:
            c = f["Fragments"][chrom]
            # - number of fragments per barcode per chromosome
            # - barcodes
            # - start (1-based) and length of fragments
            lengths, values, ranges = c["RGLengths"][0], c["RGValues"][0], c["Ranges"]
            j = np.repeat(values.astype(str), lengths)

            c_ranges = pd.DataFrame(
                {
                    "Chromosome": chrom,
                    "Start": ranges[0] - 1,
                    "End": ranges[0] - 1 + ranges[1],
                    "CellNames": j,
                }
            )
            chr_fragments.append(c_ranges)

            # keep pointers to the ranges in the object?

        fragments = pd.concat(chr_fragments, axis=0, ignore_index=True)
        # TODO: fix AnnData to avoid transforming to str index
        mdict["fragments"] = AnnData(var=fragments)

    def read_matrix_info(h5group):
        obs_names = np.array(h5group["CellNames"]).astype(str)

        var = pd.DataFrame(h5group["FeatureDF"][:])  # seqnames, idx, start
        for colname in ("seqnames", "name"):
            if colname in var and var[colname].dtype == "O":
                # b'chr1' -> 'chr1'
                var[colname] = var[colname].values.astype(str)

        matrix_class = h5group["Class"][0].decode("utf-8")

        params = h5group["Params"][:]
        params = [dict(zip(params.dtype.names, e)) for e in params]
        # b'chr1' -> 'chr1'
        params = [
            # TODO: use {} | {} since 3.9
            s if "seqnames" not in s else {**s, **{"seqnames": s["seqnames"].decode()}}
            for s in params
        ]

        units = h5group["Units"][:].astype(str)

        return {
            "CellNames": obs_names,
            "FeatureDF": var,
            "Class": matrix_class,
            "Params": params,
            "Units": units,
        }

    def read_matrix(h5group):
        matrices = []

        chroms = [c for c in h5group if c != "Info"]
        chroms_sorted = sort_chromosomes(chroms)

        for chrom in chroms_sorted:
            c = h5group[chrom]
            # A Matrix for a chromosome is essentially a sparse matrix
            i, jl, jv = c["i"][0] - 1, c["jLengths"][0], c["jValues"][0] - 1
            # inverse RLE
            j = np.repeat(jv, jl)
            if "x" in c:
                data = c["x"][0]
            else:
                data = np.repeat(1, len(j))

            obs_sum, var_sum = c["colSums"][0], c["rowSums"][0]
            # TODO: save sums in metadata
            n_vars = len(var_sum)

            x = csr_matrix((data, (j, i)), shape=(n_obs, n_vars))
            matrices.append(x)

            var = pd.DataFrame({"var_sum": var_sum})
            if "rowVarsLog2" in c:
                var["var_varlog2"] = c["rowVarsLog2"][0]

        matrix = hstack(matrices)
        return matrix

    # TileMatrix
    if "TileMatrix" in f and (matrix is None or matrix in ["tiles", "TileMatrix"]):
        t = f["TileMatrix"]

        # Info
        tile_info = read_matrix_info(t["Info"])
        tile_var = tile_info["FeatureDF"]
        if "seqnames" in tile_var and "start" in tile_var:
            tile_var.start = tile_var.start.astype(int)
            if "Params" in tile_info and len(tile_info["Params"]) > 0:
                tile_params = tile_info["Params"]
                if "seqnames" in tile_params[0] and "tileSize" in tile_params[0]:
                    # Optimize for the same tile size across all chromosomes
                    if len(set([par["tileSize"] for par in tile_params])) == 1:
                        tile_size = int(tile_params[0]["tileSize"])
                    else:
                        tile_size = np.array(
                            list(
                                map(
                                    lambda e: [
                                        int(par["tileSize"])
                                        for par in params
                                        if par["seqnames"] == e
                                    ][0],
                                    tile_var.seqnames.values,
                                )
                            )
                        )
                    # chr1:0-500
                    tile_var.index = (
                        tile_var.seqnames
                        + ":"
                        + tile_var.start.astype(str)
                        + "-"
                        + (tile_var.start + tile_size).astype(str)
                    )

        # TODO: Support "Sparse.Binary.Matrix", "Sparse.Integer.Matrix", "Sparse.Double.Matrix", "Sparse.Assays.Matrix"
        # TODO: use disk backing?

        tile_matrix = read_matrix(t)

        mdict["tiles"] = AnnData(tile_matrix, var=tile_var, dtype=tile_matrix.dtype)
        mdict["tiles"].obs_names = tile_info["CellNames"]
        mdict["tiles"].uns["params"] = tile_info["Params"]

    if "GeneScoreMatrix" in f and (
        matrix is None or matrix in ["gene_scores", "GeneScoreMatrix"]
    ):
        gs = f["GeneScoreMatrix"]

        # Info
        gene_score_info = read_matrix_info(gs["Info"])
        gene_score_var = gene_score_info["FeatureDF"]
        gene_score_var.index = gene_score_var.name

        # Matrix
        gene_score_matrix = read_matrix(gs)

        mdict["gene_scores"] = AnnData(
            gene_score_matrix, var=gene_score_var, dtype=gene_score_matrix.dtype
        )
        mdict["gene_scores"].obs_names = gene_score_info["CellNames"]
        mdict["gene_scores"].uns["params"] = gene_score_info["Params"]

    # TODO: PeakMatrix
    if "PeakMatrix" in f and (matrix is None or matrix in ["peaks", "PeakMatrix"]):
        print(f"PeakMatrix is present but the reader is not implemented yet")

    # TODO: GeneIntegrationMatrix
    if "GeneIntegrationMatrix" in f and (
        matrix is None or matrix in ["gene_integration", "GeneIntegrationMatrix"]
    ):
        print(f"GeneIntegrationMatrix is present but the reader is not implemented yet")

    f.close()

    if matrix is not None:
        if matrix in ["tiles", "TileMatrix"]:
            adata = mdict["tiles"]
        elif matrix in ["gene_scores", "GeneScoreMatrix"]:
            adata = mdict["gene_scores"]
        elif matrix in ["peaks", "PeakMatrix"]:
            adata = mdict["peaks"]
        elif matrix in ["gene_integration", "GeneIntegrationMatrix"]:
            adata = mdict["gene_integration"]
        elif matrix in ["fragments", "Fragments"]:
            adata = mdict["fragments"]
            return adata
        else:
            raise NotImplementedError(
                f"Reading matrix {matrix} has not been implemented yet. If you think it should be, please open an issue: https://github.com/gtca/chame/issues/new."
            )

        adata.obs = adata.obs.join(metadata)
        return adata
    else:
        mdata = MuData(mdict)
        mdata.obs = metadata
        mdata.update()

        return mdata
