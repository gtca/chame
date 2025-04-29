"""Tests for genomic range merging functions."""

import polars as pl

from chame.ranges import MergeStrategy, merge


def test_merge_union():
    """Test the UNION merge strategy."""
    # Create two sample DataFrames with overlapping ranges
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [0, 20, 5],
        "end": [10, 30, 15],
        "score": [5, 7, 9],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [5, 40, 10],
        "end": [15, 50, 20],
        "score": [6, 8, 10],
    })

    # Merge using UNION strategy
    result = merge([df1, df2], strategy=MergeStrategy.UNION)

    # Expected result: [0, 15], [20, 30], [40, 50], [5, 20]
    assert len(result) == 4

    # Check chr1 results
    chr1_result = result.filter(pl.col("chrom") == "chr1").sort("start")
    assert chr1_result["start"].to_list() == [0, 20, 40]
    assert chr1_result["end"].to_list() == [15, 30, 50]

    # Check chr2 results
    chr2_result = result.filter(pl.col("chrom") == "chr2")
    assert chr2_result["start"].to_list() == [5]
    assert chr2_result["end"].to_list() == [20]


def test_merge_intersection():
    """Test the INTERSECTION merge strategy."""
    # Create two sample DataFrames with overlapping ranges
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr2"],
        "start": [0, 5],
        "end": [10, 15],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr2"],
        "start": [5, 0],
        "end": [15, 10],
    })

    # Merge using INTERSECTION strategy
    result = merge([df1, df2], strategy=MergeStrategy.INTERSECTION)

    # Expected result: [5, 10], [5, 10]
    assert len(result) == 2

    # Check the intersections
    assert result.filter(pl.col("chrom") == "chr1")["start"].item() == 5
    assert result.filter(pl.col("chrom") == "chr1")["end"].item() == 10
    assert result.filter(pl.col("chrom") == "chr2")["start"].item() == 5
    assert result.filter(pl.col("chrom") == "chr2")["end"].item() == 10


def test_merge_partition():
    """Test the PARTITION merge strategy."""
    # Create two sample DataFrames
    df1 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end": [10],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [5],
        "end": [15],
    })

    # Merge using PARTITION strategy
    result = merge([df1, df2], strategy=MergeStrategy.PARTITION)

    # Expected result: [0, 5], [5, 10], [10, 15]
    assert len(result) == 3

    # Sort by start position
    result = result.sort("start")

    # Check the partitions
    assert result["start"].to_list() == [0, 5, 10]
    assert result["end"].to_list() == [5, 10, 15]


def test_merge_partition_with_source():
    """Test the PARTITION merge strategy with source tracking."""
    # Create two sample DataFrames
    df1 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end": [10],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [5],
        "end": [15],
    })

    # Merge using PARTITION strategy with source tracking
    result = merge([df1, df2], strategy=MergeStrategy.PARTITION, source_col="sources")

    # Expected result: [0, 5] -> [0], [5, 10] -> [0, 1], [10, 15] -> [1]
    assert len(result) == 3

    # Sort by start position
    result = result.sort("start")

    # Check the partitions and sources
    assert result["start"].to_list() == [0, 5, 10]
    assert result["end"].to_list() == [5, 10, 15]

    sources = result["sources"].to_list()
    assert sources[0] == [0]  # First segment only from df1
    assert sorted(sources[1]) == [0, 1]  # Middle segment from both
    assert sources[2] == [1]  # Last segment only from df2


def test_merge_overlap():
    """Test the OVERLAP merge strategy."""
    # Create sample DataFrames with overlapping ranges
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [0, 20],
        "end": [10, 30],
        "score": [5, 7],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [5, 40],
        "end": [15, 50],
        "score": [6, 8],
    })

    # Merge using OVERLAP strategy
    result = merge([df1, df2], strategy=MergeStrategy.OVERLAP, score_col="score")

    # Expected result: highest scoring range from each cluster
    # Cluster 1: [0, 10] (score 5) and [5, 15] (score 6) -> [5, 15]
    # Cluster 2: [20, 30] (score 7) -> [20, 30]
    # Cluster 3: [40, 50] (score 8) -> [40, 50]
    assert len(result) == 3

    # Sort by start position
    result = result.sort("start")

    # Check the selected ranges
    assert result["start"].to_list() == [5, 20, 40]
    assert result["end"].to_list() == [15, 30, 50]
    assert result["score"].to_list() == [6, 7, 8]  # Highest scores selected


def test_merge_iterative_overlap():
    """Test the ITERATIVE_OVERLAP merge strategy."""
    # Create sample DataFrames with multiple overlapping ranges
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
        "start": [0, 20, 50, 0, 100],
        "end": [10, 40, 70, 10, 110],
        "score": [7, 5, 8, 9, 10],
    })
    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
        "start": [5, 20, 50, 0, 85],
        "end": [15, 40, 75, 15, 125],
        "score": [9, 8, 5, 10, 9],
    })

    # Merge using ITERATIVE_OVERLAP strategy
    result = merge([df1, df2], strategy=MergeStrategy.ITERATIVE_OVERLAP, score_col="score")

    # Expected result:
    # 1. Select chr1:5-15 (score 9) - highest score
    # 2. Remove chr1:0-10 (overlaps with selection)
    # 3. Select chr1:30-50 (score 8) - next highest
    # 4. Select chr2:0-15 (score 10) - no overlap with previous selections
    # 5. Remove chr2:0-10 (overlaps with selection)
    # 6. Select chr2:100-110 (score 10) - no overlap with previous selections
    # 7. Remove chr2:85-125 (overlaps with selection)
    assert len(result) == 5
    assert result["chrom"].to_list() == ["chr1", "chr1", "chr1", "chr2", "chr2"]
    assert result["start"].to_list() == [5, 20, 50, 0, 100]
    assert result["end"].to_list() == [15, 40, 70, 15, 110]
    assert result["score"].to_list() == [9, 8, 8, 10, 10]


def test_merge_edge_cases():
    """Test edge cases for the merge function."""
    # Empty list of DataFrames
    empty_result = merge([])
    assert len(empty_result) == 0

    # Single DataFrame
    df = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end": [10],
    })
    single_result = merge([df])
    assert len(single_result) == 1

    # DataFrame with no regions
    empty_df = pl.DataFrame({
        "chrom": [],
        "start": [],
        "end": [],
    })
    empty_regions_result = merge([empty_df])
    assert len(empty_regions_result) == 0

    # String strategy
    string_strategy_result = merge([df], strategy="UNION")
    assert len(string_strategy_result) == 1


def test_merge_with_min_overlap():
    """Test merge with minimum overlap requirement."""
    df1 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end": [10],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [9],
        "end": [20],
    })

    # With default min_overlap (1 bp)
    default_result = merge([df1, df2], strategy=MergeStrategy.UNION)
    assert len(default_result) == 1
    assert default_result["start"].item() == 0
    assert default_result["end"].item() == 20

    # With min_overlap = 2 bp (the ranges only overlap by 1 bp, so they shouldn't merge)
    min_overlap_result = merge([df1, df2], strategy=MergeStrategy.UNION, min_overlap=2)
    assert len(min_overlap_result) == 2

    # Sort by start position
    min_overlap_result = min_overlap_result.sort("start")
    assert min_overlap_result["start"].to_list() == [0, 9]
    assert min_overlap_result["end"].to_list() == [10, 20]


def test_merge_with_min_overlap_2():
    """Test merging with minimum overlap requirements."""
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [0, 20, 0],
        "end": [10, 30, 10],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [9, 28, 9],
        "end": [19, 40, 19],
    })

    # Test with different minimum overlap requirements
    for min_overlap in [1, 2, 5]:
        result = merge([df1, df2], strategy=MergeStrategy.UNION, min_overlap=min_overlap)
        if min_overlap == 1:
            assert len(result) == 3  # All ranges should be merged
        elif min_overlap == 2:
            assert len(result) == 5  # Some ranges should be merged
        else:
            assert len(result) == 6  # No ranges should be merged


def test_merge_with_fraction_overlap():
    """Test merge with fraction overlap requirement."""
    df1 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end": [10],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [9],
        "end": [20],
    })

    # With default fraction_overlap (0.0)
    default_result = merge([df1, df2], strategy=MergeStrategy.UNION)
    assert len(default_result) == 1
    assert default_result["start"].item() == 0
    assert default_result["end"].item() == 20

    # With fraction_overlap = 0.2 (1 bp overlap is not enough for the smaller range)
    fraction_result = merge([df1, df2], strategy=MergeStrategy.UNION, fraction_overlap=0.2)
    assert len(fraction_result) == 2

    # Sort by start position
    fraction_result = fraction_result.sort("start")
    assert fraction_result["start"].to_list() == [0, 9]
    assert fraction_result["end"].to_list() == [10, 20]

def test_merge_with_fraction_overlap_2():
    """Test merging with fraction overlap requirements."""
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [0, 20, 0],
        "end": [10, 30, 10],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [8, 25, 5],
        "end": [18, 35, 15],
    })

    # Test with different fraction overlap requirements
    for fraction in [0.1, 0.5, 0.9]:
        result = merge([df1, df2], strategy=MergeStrategy.UNION, fraction_overlap=fraction)
        if fraction == 0.1:
            assert len(result) == 3  # Most ranges should be merged
        elif fraction == 0.5:
            assert len(result) == 4  # Some ranges should be merged
        else:
            assert len(result) == 6  # No ranges should be merged


def test_merge_complex_overlaps():
    """Test merging with complex overlapping patterns."""
    # Create DataFrames with complex overlapping patterns
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
        "start": [0, 20, 50, 0, 100],
        "end": [10, 40, 70, 10, 110],
        "score": [5, 7, 9, 6, 8],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
        "start": [5, 30, 60, 5, 105],
        "end": [15, 45, 80, 15, 115],
        "score": [6, 8, 10, 7, 9],
    })

    df3 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [25, 55, 8],
        "end": [35, 65, 12],
        "score": [7, 9, 8],
    })

    # Test UNION strategy
    union_result = merge([df1, df2, df3], strategy=MergeStrategy.UNION)
    assert len(union_result) == 5  # chr1: [0, 15], [20, 45], [50, 80], chr2: [0, 15], [100, 115]

    # Test INTERSECTION strategy
    intersection_result = merge([df1, df2, df3], strategy=MergeStrategy.INTERSECTION)
    assert len(intersection_result) == 3  # chr1: [30, 35], [60, 65] chr2: [8, 10]

    # Test PARTITION strategy
    partition_result = merge([df1, df2, df3], strategy=MergeStrategy.PARTITION)
    assert len(partition_result) == 21
    # partitions:
    # chr1: [0, 5], [5, 10], [10, 15], [20, 25], [25, 30], [30, 35], [35, 40], [40, 45], [50, 55], [55, 60], [60, 65], [65, 70], [70, 80],
    # chr2: [0, 5], [5, 8], [8, 10], [10, 12], [12, 15], [100, 105], [105, 110], [110, 115]

    # Test OVERLAP strategy
    overlap_result = merge([df1, df2, df3], strategy=MergeStrategy.OVERLAP, score_col="score")
    assert len(overlap_result) == 5  # One representative per overlapping cluster


def test_merge_multiple_chromosomes():
    """Test merging with multiple chromosomes and complex patterns."""
    df1 = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2", "chr3", "chr3"],
        "start": [0, 20, 0, 0, 30],
        "end": [10, 30, 10, 20, 40],
        "score": [5, 7, 9, 6, 8],
    })

    df2 = pl.DataFrame({
        "chrom": ["chr1", "chr2", "chr2", "chr3", "chr4"],
        "start": [5, 5, 15, 10, 0],
        "end": [15, 15, 25, 30, 10],
        "score": [6, 8, 10, 7, 9],
    })

    # Test UNION strategy
    union_result = merge([df1, df2], strategy=MergeStrategy.UNION)
    assert len(union_result) == 7  # chr1: [0, 15], [20, 30], chr2: [0, 10], [15, 25], chr3: [0, 30], [30, 40], chr4: [0, 10]

    # Test PARTITION strategy with source tracking
    partition_result = merge([df1, df2], strategy=MergeStrategy.PARTITION, source_col="sources")
    assert len(partition_result) == 13
    assert all("sources" in row for row in partition_result.iter_rows(named=True))


def test_merge_large_overlap_clusters():
    """Test merging with large clusters of overlapping ranges."""
    # Create a large cluster of overlapping ranges
    starts = list(range(0, 100, 5))
    ends = [s + 20 for s in starts]
    scores = list(range(len(starts)))

    df1 = pl.DataFrame({
        "chrom": ["chr1"] * len(starts),
        "start": starts,
        "end": ends,
        "score": scores,
    })

    # Create another set of overlapping ranges
    starts2 = list(range(10, 110, 5))
    ends2 = [s + 15 for s in starts2]
    scores2 = [s + 100 for s in range(len(starts2))]

    df2 = pl.DataFrame({
        "chrom": ["chr1"] * len(starts2),
        "start": starts2,
        "end": ends2,
        "score": scores2,
    })

    # Test ITERATIVE_OVERLAP strategy
    result = merge([df1, df2], strategy=MergeStrategy.ITERATIVE_OVERLAP, score_col="score")
    assert len(result) > 0  # Should have some non-overlapping ranges
    assert all(result["score"] >= 100)  # Should prefer higher scores from df2

