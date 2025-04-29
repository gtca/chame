import re
from collections.abc import Sequence

import polars as pl


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
