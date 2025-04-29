import re
from collections.abc import Sequence

import polars as pl


def parse_region_string(region: str) -> pl.DataFrame:
    chrom, start, end = re.split("-|:", region)
    start, end = start.replace(',', ''), end.replace(',', '')
    feature_df = pl.DataFrame({"chrom": [chrom], "start": [start], "end": [end]})
    feature_df = feature_df.with_columns(
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64)
    )

    return feature_df

def parse_region_strings(regions: Sequence[str] | tuple[str]) -> pl.DataFrame:
    """
    Parse multiple region strings into a single DataFrame.

    Parameters
    ----------
    regions
        List or tuple of region strings in format `['chr1:1-2000000', ...]` or `['chr1-1-2000000', ...]`

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: chrom, start, end
    """
    chroms = []
    starts = []
    ends = []

    for region in regions:
        chrom, start, end = re.split("-|:", region)
        start, end = start.replace(',', ''), end.replace(',', '')
        chroms.append(chrom)
        starts.append(start)
        ends.append(end)

    feature_df = pl.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    feature_df = feature_df.with_columns(
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64)
    )

    return feature_df
