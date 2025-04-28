from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from mudata import MuData

T = TypeVar("T", AnnData, MuData)

__all__ = ["filter_var_by_ranges"]


def _select(df, region, cols=None):
    """Select rows from dataframe that overlap with a genomic region."""
    if cols is None:
        cols = ["chrom", "start", "end"]

    if isinstance(region, str):
        # Parse "chr1:100-200" format
        chrom, positions = region.split(":")
        start, end = map(int, positions.split("-"))
    elif isinstance(region, tuple) and len(region) == 3:
        chrom, start, end = region
        start, end = int(start), int(end)
    else:
        raise ValueError("Region must be a string in format 'chr:start-end' or a tuple (chr, start, end)")

    # Convert pandas DataFrame to polars
    if isinstance(df, pd.DataFrame):
        df_pl = pl.DataFrame(df)
    else:
        df_pl = df

    # Filter rows
    result = df_pl.filter(
        (pl.col(cols[0]) == chrom) &
        (pl.col(cols[1]) < int(end)) &
        (pl.col(cols[2]) > int(start))
    )

    # Convert back to pandas for compatibility
    return result.to_pandas()


def _overlap(df1, df2, cols1=None, cols2=None, how="inner"):
    """Find overlapping regions between two dataframes."""
    if cols1 is None:
        cols1 = ["chrom", "start", "end"]
    if cols2 is None:
        cols2 = cols1

    # Convert pandas DataFrames to polars
    if isinstance(df1, pd.DataFrame):
        df1_pl = pl.DataFrame(df1)
    else:
        df1_pl = df1

    if isinstance(df2, pd.DataFrame):
        df2_pl = pl.DataFrame(df2)
    else:
        df2_pl = df2

    # Rename columns for the join
    df1_pl = df1_pl.rename({
        cols1[0]: "chrom",
        cols1[1]: "start1",
        cols1[2]: "end1"
    })

    df2_pl = df2_pl.rename({
        cols2[0]: "chrom",
        cols2[1]: "start2",
        cols2[2]: "end2"
    })

    # Join on chromosome and check for overlap
    joined = df1_pl.join(
        df2_pl,
        on="chrom",
        how=how
    ).filter(
        (pl.col("start1") < pl.col("end2")) &
        (pl.col("end1") > pl.col("start2"))
    )

    # Rename columns back
    result = joined.rename({
        "chrom": cols1[0],
        "start1": cols1[1],
        "end1": cols1[2]
    })

    # Convert back to pandas for compatibility
    return result.to_pandas()


def _coverage(df1, df2, cols1=None, cols2=None):
    """Calculate coverage of regions in df1 by regions in df2."""
    if cols1 is None:
        cols1 = ["chrom", "start", "end"]
    if cols2 is None:
        cols2 = cols1

    # Convert pandas DataFrames to polars
    if isinstance(df1, pd.DataFrame):
        df1_pl = pl.DataFrame(df1)
    else:
        df1_pl = df1

    if isinstance(df2, pd.DataFrame):
        df2_pl = pl.DataFrame(df2)
    else:
        df2_pl = df2

    # Rename columns for the join
    df1_pl = df1_pl.rename({
        cols1[0]: "chrom",
        cols1[1]: "start1",
        cols1[2]: "end1"
    })

    df2_pl = df2_pl.rename({
        cols2[0]: "chrom",
        cols2[1]: "start2",
        cols2[2]: "end2"
    })

    # Join on chromosome and check for overlap
    joined = df1_pl.join(
        df2_pl,
        on="chrom",
        how="inner"
    ).filter(
        (pl.col("start1") < pl.col("end2")) &
        (pl.col("end1") > pl.col("start2"))
    )

    # Calculate overlap length - using expressions for element-wise operations
    result = joined.with_columns([
        pl.when(pl.col("start1") > pl.col("start2"))
          .then(pl.col("start1"))
          .otherwise(pl.col("start2"))
          .alias("overlap_start"),
        pl.when(pl.col("end1") < pl.col("end2"))
          .then(pl.col("end1"))
          .otherwise(pl.col("end2"))
          .alias("overlap_end")
    ]).with_columns([
        (pl.col("overlap_end") - pl.col("overlap_start")).alias("coverage")
    ])

    # Rename columns back
    result = result.rename({
        "chrom": cols1[0],
        "start1": cols1[1],
        "end1": cols1[2]
    })

    # Convert back to pandas for compatibility
    return result.to_pandas()


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
                "min_var_coverage should be from [0.0, 1.0], "
                "{min_var_coverage} was provided"
            )

    default_ranges_cols = ["chrom", "start", "end"]
    var_ids = "var_indices"

    if isinstance(ranges_columns, str):
        convertible = data.var[ranges_columns].values != "NA"
        cols = default_ranges_cols
        var = pd.DataFrame.from_records(
            data.var.loc[:, ranges_columns].str.split("-|:").to_list(),
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

            cov = _coverage(var, ranges_df, cols1=cols, cols2=cols)
            cov_sel = cov["coverage"] / (cov[cols[2]] - cov[cols[1]]) >= min_var_coverage
            index = cov.loc[cov_sel,:][var_ids].values

        else:
            index = _select(var, ranges, cols=cols)[var_ids].values
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
            cov = _coverage(var, ranges_df, cols1=cols, cols2=cols)
            cov_sel = cov["coverage"] / (cov["end"] - cov["start"]) >= min_var_coverage
            index = cov.loc[cov_sel,:][var_ids].values
        else:
            index = _overlap(
                var, ranges_df, cols1=cols, cols2=cols, how="inner"
            )[var_ids].values

    return data[:, index]
