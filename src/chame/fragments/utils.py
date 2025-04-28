import re
from collections.abc import Sequence

import polars as pl


def parse_region_string(region: str) -> pl.DataFrame:
    chrom_start_end = re.split("-|:", region)
    feature_df = pl.DataFrame({"chrom": [chrom_start_end[0]], "start": [chrom_start_end[1]], "end": [chrom_start_end[2]]})
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
        chrom_start_end = re.split("-|:", region)
        chroms.append(chrom_start_end[0])
        starts.append(chrom_start_end[1])
        ends.append(chrom_start_end[2])

    feature_df = pl.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    feature_df = feature_df.with_columns(
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64)
    )

    return feature_df

def _get_fragment_key(
    unique: bool = True,
    cut_sites: bool = False,
    in_peaks: bool = False,
    region: str | list[str] | None = None
) -> str:
    """
    Construct a consistent key name for fragment counts based on the parameters.

    Parameters
    ----------
    unique
        If True, key will include 'unique_fragments', otherwise just 'fragments'.
    cut_sites
        If True, key will be for 'cut_sites' instead of fragments.
    in_peaks
        If True, key will include 'in_peaks'.
    region
        Region or list of regions. If provided, will be included in the key.

    Returns
    -------
    str
        The constructed key name for fragment counts.
    """
    if cut_sites:
        key = "n_cut_sites"
    elif unique:
        key = "n_unique_fragments"
    else:
        key = "n_fragments"

    if in_peaks:
        key += "_in_peaks"

    if region:
        if isinstance(region, str):
            key += f"_{region}"
        else:
            # For multiple regions, use their count
            key += f"_{len(region)}_regions"

    return key
