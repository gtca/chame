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
) -> T:
    if ranges_columns is str and ranges_columns not in data.var:
        raise KeyError(f"Ranges column {ranges_columns} was not found in .var")
    elif isinstance(ranges_columns, tuple):
        if len(ranges_columns) != 3:
            raise ValueError(
                "Chromosome, Start and End columns should be defined "
                f"but only {len(ranges_columns)} names were provided"
            )

    default_ranges_colnames = ["Chromosome", "Start", "End"]
    var_ids = "var_indices"

    if isinstance(ranges_columns, str):
        convertible = data.var[ranges_columns].values != "NA"
        var = pd.DataFrame.from_records(
            data.var[ranges_columns].str.split("-|:"),
            columns=default_ranges_colnames,
        )
        var[var_ids] = np.arange(len(var))
        var = var[convertible]
        var.Start = var.Start.astype(int)
        var.End = var.End.astype(int)
        cols = default_ranges_colnames
    else:
        var = data.var.loc[:, ranges_columns]
        var[var_ids] = np.arange(len(var))
        cols = ranges_columns

    if isinstance(ranges, str) or (
                isinstance(ranges, tuple) and len(ranges) == 3
            ):
        index = bioframe.select(var, ranges, cols=cols)[var_ids].values
    else:
        if isinstance(ranges, np.ndarray) or isinstance(ranges, list):
            ranges_df = pd.DataFrame.from_records(
                pd.Series(ranges).str.split("-|:"),
                columns=default_ranges_colnames,
            )
            ranges_df.Start = ranges_df.Start.astype(int)
            ranges_df.End = ranges_df.End.astype(int)
        else:
            ranges_df = ranges
        index = bioframe.overlap(
            var, ranges_df, cols1=cols, cols2=cols, how="inner"
        )[var_ids].values

    return data[:, index]
