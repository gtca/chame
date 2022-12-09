<img src="./docs/img/chame_logo.svg" data-canonical-src="./docs/img/chame_logo.svg" width="700"/>

> `chame` is work in progress. Active contributions via code or feedback are welcome.

`chame` stands for a _chromatin analysis module_. It is being developed as a Python library for working with genomic ranges and chromatin accessibility in the [`scverse`](https://scverse.org/) ecosystem.

[Documentation](https://gtca.github.io/chame) | [Examples](https://gtca.github.io/chame/examples/)

# Install `chame`

```
pip install git+https://github.com/gtca/chame
```

# Functionality

## I/O

Raw data input: [10X Genomics ARC](https://gtca.github.io/chame/examples/10x_io.html), [ArchR (arrow files)](https://gtca.github.io/chame/examples/archr_io.html).

Data model: [AnnData](https://github.com/scverse/anndata) and [MuData](https://github.com/scverse/mudata).

## Preprocessing and QC

TF-IDF / LSI. TSS enrichment, mono-nucleosome occupancy, etc.

## Methods

[Differential accessibility](https://gtca.github.io/chame/examples/chromvar.html), transcription factor activity, etc.

## Visualization

QC. Joint gene + peak visualisation.

---

# Other projects

1. [muon](https://github.com/scverse/muon)
1. [episcanpy](https://github.com/colomemaria/epiScanpy)
1. [SnapATAC2](https://github.com/kaizhang/SnapATAC2)
1. [ArchR](https://www.archrproject.com/)
1. [Signac](https://satijalab.org/signac/)

