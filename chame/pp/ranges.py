from __future__ import annotations
from typing import TypeVar
import numpy as np
import pandas as pd
import bioframe

from anndata import AnnData
from mudata import MuData

T = TypeVar("T", AnnData, MuData)

__all__ = ["filter_var_by_ranges"]


def filter_var_by_ranges(
    data: T,
    ranges: str | list[str] | np.ndarray | pd.DataFrame,
    ranges_columns: str | tuple[str, str, str] = "interval",
    min_var_coverage: float | None = None,
) -> T:
    """
    Subset along .var using range information.

    Parameters
    ----------
    data
        AnnData object or MuData object with ranges information stored in .var
        as an interval column ("chr1:100-200") or 
        as three columns (chrom, start, end).
    ranges
        String ("chr1:100-200") or a tuple ("chr1", "100", 200") for a single range,
        list or numpy array of strings or pandas data frame with respective columns 
        for multiple intervals.
    ranges_columns
        A single column for an interval, "interval" by default.
        Alternatively, a tuple with chromosome, start and end columns,
        ("chrom", "start", "end") by default.
    var_overlap
        Only count ranges overlaps greater than or equal to min_var_coverage
        as a fraction of the .var range covered by the ranges in `ranges`.
        Use `min_var_coverage = 1.0` to subset features fully enclosed in `ranges`.
    """
    if ranges_columns is str and ranges_columns not in data.var:
        raise KeyError(f"Ranges column {ranges_columns} was not found in .var")
    elif isinstance(ranges_columns, tuple):
        if len(ranges_columns) != 3:
            raise ValueError(
                "chromosome, start and end columns should be defined "
                f"but only {len(ranges_columns)} names were provided"
            )

    if min_var_coverage is not None and min_var_coverage is not False:
        if min_var_coverage > 1 or min_var_coverage < 0:
            raise ValueError(
                f"min_var_coverage should be from [0.0, 1.0], "
                "{min_var_coverage} was provided"
            )

    default_ranges_cols = ["chrom", "start", "end"]
    var_ids = "var_indices"

    if isinstance(ranges_columns, str):
        convertible = data.var[ranges_columns].values != "NA"
        cols = default_ranges_cols
        var = pd.DataFrame.from_records(
            data.var[ranges_columns].str.split("-|:"),
            columns=cols,
        )
        var[var_ids] = np.arange(len(var))
        var = var[convertible]
        var[cols[1]] = var[cols[1]].astype(int)
        var[cols[2]] = var[cols[2]].astype(int)
    else:
        var = data.var.loc[:, ranges_columns]
        var[var_ids] = np.arange(len(var))
        cols = ranges_columns

    if isinstance(ranges, str) or (
                isinstance(ranges, tuple) and len(ranges) == 3
            ):
        if min_var_coverage is not None and min_var_coverage is not False:
            ranges_df = pd.DataFrame.from_records(
                pd.Series(ranges).str.split("-|:"),
                columns=cols,
            )
            ranges_df[cols[1]] = ranges_df[cols[1]].astype(int)
            ranges_df[cols[2]] = ranges_df[cols[2]].astype(int)

            cov = bioframe.coverage(var, ranges_df, cols1=cols, cols2=cols)
            cov_sel = cov["coverage"] / (cov[cols[2]] - cov[cols[1]]) >= min_var_coverage
            index = cov.loc[cov_sel,:][var_ids].values

        else:
            index = bioframe.select(var, ranges, cols=cols)[var_ids].values
    else:
        if isinstance(ranges, np.ndarray) or isinstance(ranges, list):
            ranges_df = pd.DataFrame.from_records(
                pd.Series(ranges).str.split("-|:"),
                columns=cols,
            )
            ranges_df[cols[1]] = ranges_df[cols[1]].astype(int)
            ranges_df[cols[2]] = ranges_df[cols[2]].astype(int)
        else:
            ranges_df = ranges
        if min_var_coverage is not None and min_var_coverage is not False:
            cov = bioframe.coverage(var, ranges_df, cols1=cols, cols2=cols)
            cov_sel = cov["coverage"] / (cov["end"] - cov["start"]) >= min_var_coverage
            index = cov.loc[cov_sel,:][var_ids].values
        else:
            index = bioframe.overlap(
                var, ranges_df, cols1=cols, cols2=cols, how="inner"
            )[var_ids].values

    return data[:, index]
