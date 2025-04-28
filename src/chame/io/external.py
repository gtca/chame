from os import PathLike, path

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix, hstack


def read_arrow(
    filename: PathLike, fragments: bool = False, matrix: str | None = None
) -> AnnData | MuData:
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
            n_obs = len(obs_sum)
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
                                        for par in tile_params
                                        if par["seqnames"] == e
                                    ][0],
                                    tile_var.seqnames.values,
                                )
                            )
                        )
                    # Chromosome, Start, End
                    tile_var.rename(
                        {"seqnames": "Chromosome", "start": "Start"},
                        axis=1,
                        inplace=True,
                    )
                    tile_var["End"] = tile_var.Start + tile_size
                    # chr1:0-500
                    tile_var.index = (
                        tile_var.Chromosome
                        + ":"
                        + tile_var.Start.astype(str)
                        + "-"
                        + tile_var.End.astype(str)
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
        print("PeakMatrix is present but the reader is not implemented yet")

    # TODO: GeneIntegrationMatrix
    if "GeneIntegrationMatrix" in f and (
        matrix is None or matrix in ["gene_integration", "GeneIntegrationMatrix"]
    ):
        print("GeneIntegrationMatrix is present but the reader is not implemented yet")

    # TODO: Embeddings
    # TODO: GroupCoverages
    # TODO: IterativeLSI, IterativeLSI2

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
