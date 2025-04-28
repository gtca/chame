import re

import pandas as pd
from anndata import AnnData
from mudata import MuData
from tqdm import tqdm


def count_fragments(
    data: AnnData | MuData,
    region: str | list[str] | None = None,
    in_peaks: bool = False,
    unique: bool = True,
    cut_sites: bool = False,
    barcodes: str | None = None,
    verbose: bool = True,
    add_key: str | None = None,
    min_fragment_length: int | None = None,
    max_fragment_length: int | None = None,
    shift_start: int = 0,
    shift_end: int = 0
) -> None:
    """
    Count the number of fragments per cell.

    Parameters
    ----------
    data : AnnData | MuData
        AnnData object with peak counts or multimodal MuData object with 'atac' or 'peaks' modality.
    region : str | list[str] | None, default=None
        Region or list of regions to count fragments in, in the format 'chr1' (entire chromosome)
        or 'chr1:1-100000' or 'chr1-1-100000'. If None, count fragments in the entire genome.
    in_peaks : bool, default=False
        Whether to count only fragments that overlap with peaks.
        Requires var_names in adata to be in format 'chr:start-end'.
    unique : bool, default=True
        If True, count each fragment only once per cell. If False, count fragments
        according to the score field in the fragments file (number of read pairs).
    cut_sites : bool, default=False
        If True, count individual cut sites instead of fragments. Each fragment has
        two cut sites (its 5' and 3' ends). When counting cut sites in peaks, only those
        that fall within the region are counted (0, 1, or 2 per fragment).
    barcodes : str | None, default=None
        Column name of .obs slot of the AnnData object
        with barcodes corresponding to the ones in the fragments file.
        By default, use data.obs_names as barcodes.
    verbose : bool, default=True
        Whether to show progress bars.
    add_key : str | None, default=None
        If provided, adds the fragment counts to data.obs with this key.
        For example, add_key="n_fragments" will create data.obs["n_fragments"].
        By default, will construct the key from the following parts:
        1. 'n_fragments', 'n_unique_fragments', or 'n_cut_sites',
        2. '_'
        3. 'in_peaks' if in_peaks is True,
        4. '_'
        5. region / '_'.join(region) if region is not None.
    min_fragment_length : int | None, default=None
        Minimum fragment length to include in the count. If None, no minimum length filter is applied.
    max_fragment_length : int | None, default=None
        Maximum fragment length to include in the count. If None, no maximum length filter is applied.
    shift_start : int, default=0
        Number of bases to shift the start position of each fragment.
        Use shift_start=4 to account for the sticky ends introduced by Tn5.
    shift_end : int, default=0
        Number of bases to shift the end position of each fragment.
        Use shift_end=-5 to account for the sticky ends introduced by Tn5.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. "
            "Install pysam from PyPI (`pip install pysam`) or from GitHub "
            "(`pip install git+https://github.com/pysam-developers/pysam`)"
        )

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    elif isinstance(data, MuData) and "peaks" in data.mod:
        adata = data.mod["peaks"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' or 'peaks' modality")

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError("No fragments file located. Run tools.locate_fragments first.")

    fragment_path = adata.uns["files"]["fragments"]

    # Get barcode mapping
    if barcodes and barcodes in adata.obs.columns:
        adata.obs[barcodes]
        index_to_use = adata.obs.index
    else:
        index_to_use = adata.obs.index

    # Initialize counts dictionary
    fragment_counts = {bc: 0 for bc in index_to_use}

    # Open fragments file
    fragments = pysam.TabixFile(fragment_path, parser=pysam.asBed())

    # Parse region parameter
    regions_to_query = []
    if region is not None:
        if isinstance(region, str):
            region = [region]

        for reg in region:
            # Check if region is a chromosome name only
            if ":" not in reg and "-" not in reg:
                # Full chromosome
                regions_to_query.append((reg, None, None))
            else:
                # Parse region with coordinates
                try:
                    if ":" in reg:
                        chrom, pos = reg.split(":")
                        start, end = map(int, pos.split("-"))
                    else:
                        chrom, start, end = re.split("[-]", reg)
                        start, end = int(start), int(end)
                    regions_to_query.append((chrom, start, end))
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid region format: {reg}. Use 'chr1', 'chr1:1-100000', or 'chr1-1-100000'")

    def should_count_fragment(fr):
        """Helper function to determine if a fragment should be counted based on length filters"""
        if min_fragment_length is not None or max_fragment_length is not None:
            fragment_length = fr.end - fr.start
            if min_fragment_length is not None and fragment_length < min_fragment_length:
                return False
            if max_fragment_length is not None and fragment_length > max_fragment_length:
                return False
        return True

    def get_fragment_positions(fr):
        """Helper function to get shifted fragment positions"""
        return (fr.start + shift_start, fr.end + shift_end)

    try:
        if in_peaks:
            # Parse peak regions from var_names
            peak_regions = []
            for peak in adata.var_names:
                try:
                    # Handle both chr:start-end and chr-start-end formats
                    if ":" in peak:
                        chrom, pos = peak.split(":")
                        start, end = map(int, pos.split("-"))
                    else:
                        chrom, start, end = re.split("[-]", peak)
                        start, end = int(start), int(end)

                    # If regions are specified, only include peaks that overlap with those regions
                    if regions_to_query:
                        for reg_chrom, reg_start, reg_end in regions_to_query:
                            if chrom == reg_chrom:
                                # If only chromosome is specified, include all peaks in that chromosome
                                if reg_start is None and reg_end is None:
                                    peak_regions.append((chrom, start, end))
                                    break
                                # Check if peak overlaps with region
                                elif (reg_start <= start <= reg_end) or (reg_start <= end <= reg_end) or (start <= reg_start and end >= reg_end):
                                    peak_regions.append((chrom, start, end))
                                    break
                    else:
                        peak_regions.append((chrom, start, end))
                except (ValueError, IndexError):
                    continue

            # Count fragments that overlap with peaks
            for chrom, start, end in tqdm(peak_regions, desc="Counting fragments in peaks", disable=not verbose):
                for fr in fragments.fetch(chrom, start, end):
                    if fr.name in fragment_counts and should_count_fragment(fr):
                        fr_start, fr_end = get_fragment_positions(fr)
                        if unique:
                            # Count fragment
                            fragment_counts[fr.name] += 1

                            # If counting cut sites, check which cut sites fall within the peak
                            if cut_sites:
                                # Check if start is in peak
                                start_in_peak = start <= fr_start < end
                                # Check if end is in peak
                                end_in_peak = start < fr_end <= end

                                # Add the number of cut sites in the peak (0, 1, or 2)
                                fragment_counts[fr.name] += (start_in_peak + end_in_peak)
                        else:
                            # Use score column as read count
                            read_count = int(fr.score)
                            fragment_counts[fr.name] += read_count

                            # If counting cut sites, check which cut sites fall within the peak
                            if cut_sites:
                                # Check if start is in peak
                                start_in_peak = start <= fr_start < end
                                # Check if end is in peak
                                end_in_peak = start < fr_end <= end

                                # Add the number of cut sites in the peak (0, 1, or 2) for each read
                                fragment_counts[fr.name] += read_count * (start_in_peak + end_in_peak)

        else:
            # Count all fragments in specified regions or all regions
            if regions_to_query:
                for chrom, start, end in regions_to_query:
                    if start is None and end is None:
                        # Entire chromosome
                        for fr in tqdm(fragments.fetch(chrom), desc=f"Counting fragments on {chrom}", disable=not verbose):
                            if fr.name in fragment_counts and should_count_fragment(fr):
                                if unique:
                                    # Count fragment
                                    fragment_counts[fr.name] += 1

                                    # For cut sites without peaks, we count both ends (2 per fragment)
                                    if cut_sites:
                                        fragment_counts[fr.name] += 2
                                else:
                                    # Use score column as read count
                                    read_count = int(fr.score)
                                    fragment_counts[fr.name] += read_count

                                    # For cut sites without peaks, we count both ends for each read
                                    if cut_sites:
                                        fragment_counts[fr.name] += 2 * read_count
                    else:
                        # Specific region
                        for fr in tqdm(fragments.fetch(chrom, start, end), desc=f"Counting fragments in {chrom}:{start}-{end}", disable=not verbose):
                            if fr.name in fragment_counts and should_count_fragment(fr):
                                fr_start, fr_end = get_fragment_positions(fr)
                                if unique:
                                    # Count fragment
                                    fragment_counts[fr.name] += 1

                                    # If counting cut sites, check which cut sites fall within the region
                                    if cut_sites:
                                        # Check if start is in region
                                        start_in_region = start <= fr_start < end
                                        # Check if end is in region
                                        end_in_region = start < fr_end <= end

                                        # Add the number of cut sites in the region (0, 1, or 2)
                                        fragment_counts[fr.name] += (start_in_region + end_in_region)
                                else:
                                    # Use score column as read count
                                    read_count = int(fr.score)
                                    fragment_counts[fr.name] += read_count

                                    # If counting cut sites, check which cut sites fall within the region
                                    if cut_sites:
                                        # Check if start is in region
                                        start_in_region = start <= fr_start < end
                                        # Check if end is in region
                                        end_in_region = start < fr_end <= end

                                        # Add the number of cut sites in the region (0, 1, or 2) for each read
                                        fragment_counts[fr.name] += read_count * (start_in_region + end_in_region)
            else:
                # Count all fragments in all regions
                try:
                    for contig in fragments.contigs:
                        for fr in tqdm(fragments.fetch(contig), desc=f"Counting fragments on {contig}", disable=not verbose):
                            if fr.name in fragment_counts and should_count_fragment(fr):
                                if unique:
                                    # Count fragment
                                    fragment_counts[fr.name] += 1

                                    # For cut sites without region constraints, count both ends (2 per fragment)
                                    if cut_sites:
                                        fragment_counts[fr.name] += 2
                                else:
                                    # Use score column as read count
                                    read_count = int(fr.score)
                                    fragment_counts[fr.name] += read_count

                                    # For cut sites without region constraints, count both ends for each read
                                    if cut_sites:
                                        fragment_counts[fr.name] += 2 * read_count
                except ValueError:
                    # If contigs can't be accessed, try to count all fragments
                    for fr in tqdm(fragments.fetch(), desc="Counting all fragments", disable=not verbose):
                        if fr.name in fragment_counts and should_count_fragment(fr):
                            if unique:
                                # Count fragment
                                fragment_counts[fr.name] += 1

                                # For cut sites without region constraints, count both ends (2 per fragment)
                                if cut_sites:
                                    fragment_counts[fr.name] += 2
                            else:
                                # Use score column as read count
                                read_count = int(fr.score)
                                fragment_counts[fr.name] += read_count

                                # For cut sites without region constraints, count both ends for each read
                                if cut_sites:
                                    fragment_counts[fr.name] += 2 * read_count

    except Exception as e:
        fragments.close()
        raise e

    # Convert to pandas Series
    counts_series = pd.Series(fragment_counts)
    fragments.close()

    if add_key is None:
        add_key = "n_fragments" if not unique else "n_unique_fragments" if not cut_sites else "n_cut_sites"
        if in_peaks:
            add_key += "_in_peaks"
        if region:
            if isinstance(region, str):
                add_key += "_" + region
            else:
                add_key += "_" + "_".join(region)

    # If this is a modality in MuData, add to the corresponding modality
    if isinstance(data, MuData):
        if "atac" in data.mod and data.mod["atac"] is adata:
            data.mod["atac"].obs[add_key] = counts_series
        elif "peaks" in data.mod and data.mod["peaks"] is adata:
            data.mod["peaks"].obs[add_key] = counts_series
        # Also add to the joint obs
        data.obs[add_key] = counts_series
    else:
        # Add directly to AnnData
        data.obs[add_key] = counts_series
