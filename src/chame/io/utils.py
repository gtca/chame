from os import PathLike
from pathlib import Path


def read_chrom_sizes(filename: PathLike, chroms: list[str] | None = None) -> tuple[dict[str, int], list[str]]:
    """Read a chromosome sizes file.

    Parameters
    ----------
    filename : PathLike
        The path to the chromosome sizes file.
    chroms : Optional[List[str]], optional
        Only include these chromosomes, by default None

    Returns
    -------
    Dict[str, int]
        A dictionary mapping chromosome names to their sizes.
    List[str]
        A list of the chromosome names.
    """
    chrom_sizes = {}
    fname = Path(filename)
    with filename.open() as f:
        for line in f:
            chrom, size = line.strip().split('\t')
            if chroms is None or chrom in chroms:
                chrom_sizes[chrom] = int(size)

    chroms_list = list(chrom_sizes.keys())
    return chrom_sizes, chroms_list
