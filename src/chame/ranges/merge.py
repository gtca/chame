"""Merge strategies for genomic ranges.

This module provides various strategies for merging genomic ranges (peaks) with
polars DataFrames.
"""

from __future__ import annotations

import logging
from enum import Enum, auto

import polars as pl

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategies for merging genomic ranges."""

    UNION = auto()
    """Combine overlapping ranges into a larger range.

    Example: [0, 10] and [5, 15] -> [0, 15]
    """

    INTERSECTION = auto()
    """Keep only the overlapping parts of ranges.

    Example: [0, 10] and [5, 15] -> [5, 10]
    """

    PARTITION = auto()
    """Split ranges at overlap boundaries to create disjoint segments.

    Example: [0, 10] and [5, 15] -> [0, 5], [5, 10], [10, 15]
    """

    OVERLAP = auto()
    """Group overlapping ranges and select one representative (like bedtools cluster).

    Typically selects the highest scoring range from each cluster.
    """

    ITERATIVE_OVERLAP = auto()
    """Keep the most significant peak and remove all overlapping ones, repeat.

    Peaks are first ranked by their significance. The most significant peak is retained
    and any peak that directly overlaps with it is removed from further analysis.
    Then, of the remaining peaks, this process is repeated until no more peaks remain.
    This avoids daisy-chaining and still allows for use of fixed-width peaks.
    """


def merge(
    dfs: list[pl.DataFrame],
    strategy: MergeStrategy | str = MergeStrategy.UNION,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    score_col: str | None = None,
    source_col: str | None = None,
    min_overlap: int = 1,
    fraction_overlap: float = 0.0,
) -> pl.DataFrame:
    """Merge genomic ranges from multiple DataFrames.

    Args:
        dfs: List of polars DataFrames with genomic ranges
        strategy: Merge strategy to use
        chrom_col: Name of chromosome column
        start_col: Name of start position column
        end_col: Name of end position column
        score_col: Name of score column (required for OVERLAP and ITERATIVE_OVERLAP strategies)
        source_col: Name of column to create that will indicate the source of each range
                   If None, no source tracking is done
        min_overlap: Minimum number of base pairs required for ranges to be considered overlapping
        fraction_overlap: Minimum fraction of the smaller range that must overlap
                         (value between 0.0 and 1.0)

    Returns:
        DataFrame with merged ranges according to the selected strategy
    """
    if isinstance(strategy, str):
        try:
            strategy = MergeStrategy[strategy.upper()]
        except KeyError:
            valid_strategies = ", ".join(s.name for s in MergeStrategy)
            raise ValueError(
                f"Invalid strategy: {strategy}. Valid strategies are: {valid_strategies}"
            )

    # Validate inputs
    if not dfs:
        return pl.DataFrame(schema={
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64
        })

    if len(dfs) == 1:
        return dfs[0]

    # Check that all DataFrames have the required columns
    required_cols = [chrom_col, start_col, end_col]
    for i, df in enumerate(dfs):
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame at index {i} is missing required column: {col}")

    # Prepare DataFrames with source information if requested
    if source_col is not None:
        processed_dfs = []
        for i, df in enumerate(dfs):
            # Add source column with index of the DataFrame
            processed_dfs.append(df.with_columns(pl.lit(i).alias(source_col)))
        dfs = processed_dfs

    # Apply the selected merge strategy
    if strategy == MergeStrategy.UNION:
        return _merge_union(dfs, chrom_col, start_col, end_col, source_col, min_overlap, fraction_overlap)
    elif strategy == MergeStrategy.INTERSECTION:
        return _merge_intersection(dfs, chrom_col, start_col, end_col, source_col, min_overlap, fraction_overlap)
    elif strategy == MergeStrategy.PARTITION:
        return _merge_partition(dfs, chrom_col, start_col, end_col, source_col, min_overlap, fraction_overlap)
    elif strategy == MergeStrategy.OVERLAP:
        if score_col is None:
            raise ValueError("score_col is required for OVERLAP strategy")
        return _merge_overlap(dfs, chrom_col, start_col, end_col, score_col, source_col, min_overlap, fraction_overlap)
    elif strategy == MergeStrategy.ITERATIVE_OVERLAP:
        if score_col is None:
            raise ValueError("score_col is required for ITERATIVE_OVERLAP strategy")
        return _merge_iterative_overlap_faster(dfs, chrom_col, start_col, end_col, score_col, source_col, min_overlap, fraction_overlap)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


def _merge_union(
    dfs: list[pl.DataFrame],
    chrom_col: str,
    start_col: str,
    end_col: str,
    source_col: str | None,
    min_overlap: int,
    fraction_overlap: float,
) -> pl.DataFrame:
    """Merge ranges using the UNION strategy.

    Overlapping ranges are combined into larger ranges.
    Example: [0, 10] and [5, 15] -> [0, 15]
    """
    # Combine all DataFrames
    combined_df = pl.concat(dfs)

    # Group by chromosome and apply the merging logic
    result_rows = []

    # Process each chromosome separately
    for chrom in combined_df[chrom_col].unique():
        chrom_regions = combined_df.filter(pl.col(chrom_col) == chrom)

        # Sort regions by start position
        chrom_regions = chrom_regions.sort(start_col)

        # Early return if no regions for this chromosome
        if len(chrom_regions) == 0:
            continue

        # Get ranges as a list for easier processing
        ranges = chrom_regions.select([start_col, end_col, source_col] if source_col else [start_col, end_col]).to_dicts()

        # Merge overlapping ranges
        merged = []
        current = ranges[0]
        current_start, current_end = current[start_col], current[end_col]
        current_sources = {current[source_col]} if source_col else set()

        for i in range(1, len(ranges)):
            next_range = ranges[i]
            next_start, next_end = next_range[start_col], next_range[end_col]
            next_source = next_range[source_col] if source_col else None

            # Check for overlap considering min_overlap and fraction_overlap
            if _ranges_overlap(current_start, current_end, next_start, next_end, min_overlap, fraction_overlap):
                # Merge by extending the current range
                current_end = max(current_end, next_end)
                if source_col:
                    current_sources.add(next_source)
            else:
                # No overlap, store current range and start a new one
                merged_range = {chrom_col: chrom, start_col: current_start, end_col: current_end}
                if source_col:
                    merged_range[source_col] = list(current_sources)
                merged.append(merged_range)
                current_start, current_end = next_start, next_end
                current_sources = {next_source} if source_col else set()

        # Add the last range
        merged_range = {chrom_col: chrom, start_col: current_start, end_col: current_end}
        if source_col:
            merged_range[source_col] = list(current_sources)
        merged.append(merged_range)
        result_rows.extend(merged)

    if not result_rows:
        schema = {
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64
        }
        if source_col:
            schema[source_col] = pl.List(pl.Int64)
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(result_rows)


def _merge_intersection(
    dfs: list[pl.DataFrame],
    chrom_col: str,
    start_col: str,
    end_col: str,
    source_col: str | None,
    min_overlap: int,
    fraction_overlap: float,
) -> pl.DataFrame:
    """Merge ranges using the INTERSECTION strategy.

    Only regions that overlap across all input DataFrames are kept.
    Example: [0, 10] and [5, 15] -> [5, 10]
    """
    if len(dfs) < 2:
        schema = {
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64
        }
        if source_col:
            schema[source_col] = pl.List(pl.Int64)
        return dfs[0] if dfs else pl.DataFrame(schema=schema)

    # Process each chromosome separately
    result_rows = []

    # Get all unique chromosomes
    all_chroms = set()
    for df in dfs:
        all_chroms.update(df[chrom_col].unique().to_list())

    for chrom in all_chroms:
        # Get ranges for this chromosome from each DataFrame
        chrom_ranges = []
        for df in dfs:
            ranges = df.filter(pl.col(chrom_col) == chrom).sort(start_col)
            if len(ranges) > 0:
                chrom_ranges.append(ranges)

        if len(chrom_ranges) < len(dfs):
            # Skip if not all DataFrames have ranges for this chromosome
            continue

        # Convert to lists of ranges for easier processing
        range_lists = []
        for df in chrom_ranges:
            range_lists.append(df.select([start_col, end_col]).to_dicts())

        # Initialize pointers for each DataFrame
        pointers = [0] * len(range_lists)
        n_ranges = [len(ranges) for ranges in range_lists]

        # Find the first range that could potentially overlap with others
        while all(p < n for p, n in zip(pointers, n_ranges)):
            # Get current ranges
            current_ranges = [range_lists[i][pointers[i]] for i in range(len(range_lists))]

            # Find the maximum start position
            max_start = max(r[start_col] for r in current_ranges)

            # Find the minimum end position
            min_end = min(r[end_col] for r in current_ranges)

            # If there's an overlap
            if max_start < min_end:
                # Check if overlap meets minimum requirements
                overlap_size = min_end - max_start
                if overlap_size >= min_overlap:
                    if fraction_overlap > 0:
                        # Calculate the smallest range size
                        min_range_size = min(r[end_col] - r[start_col] for r in current_ranges)
                        if overlap_size / min_range_size >= fraction_overlap:
                            result_rows.append({
                                chrom_col: chrom,
                                start_col: max_start,
                                end_col: min_end
                            })
                    else:
                        result_rows.append({
                            chrom_col: chrom,
                            start_col: max_start,
                            end_col: min_end
                        })

            # Move the pointer of the range with the smallest end position
            min_end_idx = min(range(len(current_ranges)),
                            key=lambda i: current_ranges[i][end_col])
            pointers[min_end_idx] += 1

            # If any pointer reaches the end, we're done with this chromosome
            if any(p >= n for p, n in zip(pointers, n_ranges)):
                break

    if not result_rows:
        schema = {
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64
        }
        if source_col:
            schema[source_col] = pl.List(pl.Int64)
        return pl.DataFrame(schema=schema)

    result = pl.DataFrame(result_rows)

    if source_col:
        # For intersection, all sources contribute to the result
        result = result.with_columns(
            pl.lit(list(range(len(dfs)))).alias(source_col)
        )

    # Sort by chromosome and start position for consistent ordering
    return result.sort([chrom_col, start_col])


def _merge_partition(
    dfs: list[pl.DataFrame],
    chrom_col: str,
    start_col: str,
    end_col: str,
    source_col: str | None,
    min_overlap: int = 1,
    fraction_overlap: float = 0.0,
) -> pl.DataFrame:
    """Merge ranges using the PARTITION strategy.

    Split ranges at overlap boundaries to create disjoint segments.
    Each segment is labeled with the sources it belongs to.
    Example: [0, 10] and [5, 15] -> [0, 5], [5, 10], [10, 15]
    """
    if not dfs:
        return pl.DataFrame(schema={
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64
        })

    # Process each chromosome separately
    result_dfs = []

    # Get all unique chromosomes
    all_chroms = set()
    for df in dfs:
        all_chroms.update(df[chrom_col].unique().to_list())

    for chrom in sorted(all_chroms):
        # Get ranges for this chromosome from each DataFrame
        chrom_ranges = []
        for i, df in enumerate(dfs):
            ranges = df.filter(pl.col(chrom_col) == chrom).sort(start_col)
            if len(ranges) > 0:
                if source_col is not None:
                    ranges = ranges.with_columns(pl.lit(i).alias(source_col))
                chrom_ranges.append(ranges)

        if not chrom_ranges:
            continue

        # Combine all ranges for this chromosome
        combined_ranges = pl.concat(chrom_ranges)

        # Create a list of events (start/end points with their sources)
        events = []
        for row in combined_ranges.iter_rows(named=True):
            events.append((row[start_col], 'start', row[source_col] if source_col is not None else None))
            events.append((row[end_col], 'end', row[source_col] if source_col is not None else None))

        # Sort events by position
        events.sort(key=lambda x: (x[0], x[1] == 'start'))  # 'start' comes before 'end' at same position

        # Process events to create segments
        segments = []
        current_sources = set() if source_col is not None else 0  # Use counter when not tracking sources
        prev_pos = None

        for pos, event_type, source in events:
            if prev_pos is not None and pos > prev_pos:
                # Create a segment only if we have active ranges
                if (source_col is not None and current_sources) or (source_col is None and current_sources > 0):
                    segment = {
                        chrom_col: chrom,
                        start_col: prev_pos,
                        end_col: pos,
                    }
                    if source_col is not None:
                        segment[source_col] = sorted(current_sources)
                    segments.append(segment)

            if event_type == 'start':
                if source_col is not None:
                    current_sources.add(source)
                else:
                    current_sources += 1
            else:  # 'end'
                if source_col is not None:
                    current_sources.remove(source)
                else:
                    current_sources -= 1

            prev_pos = pos

        if segments:
            result_dfs.append(pl.DataFrame(segments))

    if not result_dfs:
        schema = {
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64
        }
        if source_col:
            schema[source_col] = pl.List(pl.Int64)
        return pl.DataFrame(schema=schema)

    # Combine and sort results
    result = pl.concat(result_dfs)
    return result.sort([chrom_col, start_col])


def _merge_overlap(
    dfs: list[pl.DataFrame],
    chrom_col: str,
    start_col: str,
    end_col: str,
    score_col: str,
    source_col: str | None,
    min_overlap: int,
    fraction_overlap: float,
) -> pl.DataFrame:
    """Merge ranges using the OVERLAP strategy (like bedtools cluster).

    Group overlapping ranges and select one representative per cluster
    (usually the highest scoring one).
    """
    # Process each chromosome separately
    result_dfs = []

    # Get all unique chromosomes
    all_chroms = set()
    for df in dfs:
        all_chroms.update(df[chrom_col].unique().to_list())

    for chrom in sorted(all_chroms):
        # Get ranges for this chromosome from each DataFrame
        chrom_ranges = []
        for i, df in enumerate(dfs):
            ranges = df.filter(pl.col(chrom_col) == chrom).sort(start_col)
            if len(ranges) > 0:
                if source_col is not None:
                    ranges = ranges.with_columns(pl.lit(i).alias(source_col))
                chrom_ranges.append(ranges)

        if not chrom_ranges:
            continue

        # Combine all ranges for this chromosome
        combined_ranges = pl.concat(chrom_ranges)

        # Create a list of events (start/end points with their ranges)
        events = []
        for row in combined_ranges.iter_rows(named=True):
            events.append((row[start_col], 'start', row))
            events.append((row[end_col], 'end', row))

        # Sort events by position
        events.sort(key=lambda x: (x[0], x[1] == 'start'))  # 'start' comes before 'end' at same position

        # Process events to create clusters
        active_ranges = []  # List of (range, score) tuples
        current_cluster = []  # List of ranges in current cluster

        for pos, event_type, range_data in events:
            if event_type == 'start':
                active_ranges.append(range_data)
            else:  # 'end'
                # Remove the range from active ranges
                active_ranges = [r for r in active_ranges if not (
                    r[start_col] == range_data[start_col] and
                    r[end_col] == range_data[end_col] and
                    (source_col is None or r[source_col] == range_data[source_col])
                )]

            # If we have no more active ranges, finalize the current cluster
            if not active_ranges and current_cluster:
                # Select the highest scoring range from the cluster
                best_range = max(current_cluster, key=lambda x: x[score_col])
                result_row = {
                    chrom_col: chrom,
                    start_col: best_range[start_col],
                    end_col: best_range[end_col],
                    score_col: best_range[score_col],
                }
                if source_col is not None:
                    result_row[source_col] = [best_range[source_col]]
                result_dfs.append(pl.DataFrame([result_row]))
                current_cluster = []

            # Update current cluster based on active ranges
            if active_ranges:
                # Check if any range in current cluster overlaps with active ranges
                if not current_cluster or any(
                    _ranges_overlap(
                        r[start_col], r[end_col],
                        active_ranges[0][start_col], active_ranges[0][end_col],
                        min_overlap, fraction_overlap
                    ) for r in current_cluster
                ):
                    # Add all active ranges to current cluster
                    current_cluster.extend(active_ranges)
                else:
                    # Start a new cluster
                    if current_cluster:
                        # Select the highest scoring range from the previous cluster
                        best_range = max(current_cluster, key=lambda x: x[score_col])
                        result_row = {
                            chrom_col: chrom,
                            start_col: best_range[start_col],
                            end_col: best_range[end_col],
                            score_col: best_range[score_col],
                        }
                        if source_col is not None:
                            result_row[source_col] = [best_range[source_col]]
                        result_dfs.append(pl.DataFrame([result_row]))
                    current_cluster = list(active_ranges)


        # Handle any remaining cluster
        if current_cluster:
            best_range = max(current_cluster, key=lambda x: x[score_col])
            result_row = {
                chrom_col: chrom,
                start_col: best_range[start_col],
                end_col: best_range[end_col],
                score_col: best_range[score_col],
            }
            if source_col is not None:
                result_row[source_col] = [best_range[source_col]]
            result_dfs.append(pl.DataFrame([result_row]))

    if not result_dfs:
        schema = {
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64,
            score_col: combined_ranges[score_col].dtype
        }
        if source_col:
            schema[source_col] = pl.List(pl.Int64)
        return pl.DataFrame(schema=schema)

    # Combine and sort results
    result = pl.concat(result_dfs)
    return result.sort([chrom_col, start_col])


def _merge_iterative_overlap_faster(
    dfs: list[pl.DataFrame],
    chrom_col: str,
    start_col: str,
    end_col: str,
    score_col: str,
    source_col: str | None,
    min_overlap: int,
    fraction_overlap: float,
) -> pl.DataFrame:
    """Faster version of iterative overlap removal using sweep line optimization.

    Uses sweep line to identify clusters of overlapping peaks, then applies iterative
    selection within each cluster.
    """
    # Process each chromosome separately
    result_dfs = []

    # Get all unique chromosomes
    all_chroms = set()
    for df in dfs:
        all_chroms.update(df[chrom_col].unique().to_list())

    for chrom in sorted(all_chroms):
        # Get ranges for this chromosome from each DataFrame
        chrom_ranges = []
        for i, df in enumerate(dfs):
            ranges = df.filter(pl.col(chrom_col) == chrom)
            if len(ranges) > 0:
                if source_col is not None:
                    ranges = ranges.with_columns(pl.lit(i).alias(source_col))
                chrom_ranges.append(ranges)

        if not chrom_ranges:
            continue

        # Combine all ranges for this chromosome
        combined_ranges = pl.concat(chrom_ranges)

        # Create a list of events (start/end points with their ranges)
        events = []
        for row in combined_ranges.iter_rows(named=True):
            events.append((row[start_col], 'start', row))
            events.append((row[end_col], 'end', row))

        # Sort events by position
        events.sort(key=lambda x: (x[0], x[1] == 'start'))  # 'start' comes before 'end' at same position

        # Process events to identify clusters
        active_ranges = []  # Currently active ranges
        current_cluster = []  # Accumulate ranges in the current cluster
        clusters = []  # List of all clusters
        selected_peaks = []  # Final selected peaks

        for pos, event_type, range_data in events:
            if event_type == 'start':
                # If this range doesn't overlap with any active range,
                # we've found a new cluster boundary
                if active_ranges and not any(
                    _ranges_overlap(
                        r[start_col], r[end_col],
                        range_data[start_col], range_data[end_col],
                        min_overlap, fraction_overlap
                    ) for r in active_ranges
                ):
                    # Save current cluster and start a new one
                    if current_cluster:
                        clusters.append(current_cluster)
                        current_cluster = []

                # Add to active ranges and current cluster
                active_ranges.append(range_data)
                current_cluster.append(range_data)
            else:  # 'end'
                # Remove the range from active ranges
                active_ranges = [r for r in active_ranges if not (
                    r[start_col] == range_data[start_col] and
                    r[end_col] == range_data[end_col] and
                    (source_col is None or r[source_col] == range_data[source_col])
                )]

                # If no more active ranges, save the current cluster
                if not active_ranges and current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []

        # Add any remaining cluster
        if current_cluster:
            clusters.append(current_cluster)

        # Process each cluster using iterative selection
        for cluster in clusters:
            remaining_peaks = cluster.copy()

            # Keep selecting peaks until none remain
            while remaining_peaks:
                # Sort by score and take the highest scoring peak
                remaining_peaks.sort(key=lambda x: x[score_col], reverse=True)
                best_peak = remaining_peaks[0]
                selected_peaks.append(best_peak)

                # Remove all peaks that overlap with the selected peak
                remaining_peaks = [
                    peak for peak in remaining_peaks[1:]  # Skip the peak we just selected
                    if not _ranges_overlap(
                        best_peak[start_col], best_peak[end_col],
                        peak[start_col], peak[end_col],
                        min_overlap, fraction_overlap
                    )
                ]

        # Convert selected peaks to DataFrame
        if selected_peaks:
            result_dfs.append(pl.DataFrame(selected_peaks))

    if not result_dfs:
        schema = {
            chrom_col: pl.Utf8,
            start_col: pl.Int64,
            end_col: pl.Int64,
            score_col: combined_ranges[score_col].dtype
        }
        if source_col:
            schema[source_col] = pl.List(pl.Int64)
        return pl.DataFrame(schema=schema)

    # Combine and sort results
    result = pl.concat(result_dfs)
    return result.sort([chrom_col, start_col])


def _ranges_overlap(
    start1: int,
    end1: int,
    start2: int,
    end2: int,
    min_overlap: int = 1,
    fraction_overlap: float = 0.0,
) -> bool:
    """Check if two ranges overlap according to the given criteria.

    Args:
        start1: Start position of first range
        end1: End position of first range
        start2: Start position of second range
        end2: End position of second range
        min_overlap: Minimum number of base pairs that must overlap
        fraction_overlap: Minimum fraction of the smaller range that must overlap
                         (value between 0.0 and 1.0)

    Returns:
        True if ranges overlap according to criteria, False otherwise
    """
    # Check if ranges overlap at all
    if start1 >= end2 or start2 >= end1:
        return False

    # Calculate overlap size
    overlap_size = min(end1, end2) - max(start1, start2)

    # Check minimum overlap size
    if overlap_size < min_overlap:
        return False

    # Check minimum overlap fraction if needed
    if fraction_overlap > 0:
        size1 = end1 - start1
        size2 = end2 - start2
        smaller_size = min(size1, size2)

        if overlap_size / smaller_size < fraction_overlap:
            return False

    return True
