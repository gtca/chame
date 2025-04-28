from collections.abc import Iterable

import polars as pl
from anndata import AnnData
from mudata import MuData

from .. import tools

# altair is an optional dependency
try:
    import altair as alt
    alt.data_transformers.enable("vegafusion")
    _has_altair = True
except ImportError:
    _has_altair = False


def histogram(
    data: AnnData | MuData,
    region: str = "chr1-1-2000000",
    groupby: str | None = None,
    barcodes: str | None = None,
    width: int = 600,
    height: int = 400,
    binwidth: int = 5,
    max_length: int | None = None,
):
    """
    Plot Histogram of Fragment lengths within specified region using Altair.

    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    region
        Region to plot. Specified with the format `chr1:1-2000000` or`chr1-1-2000000`.
    groupby
        Column name of .obs slot of the AnnData object according to which the plot is split.
    barcodes
        Column name of .obs slot of the AnnData object
        with barcodes corresponding to the ones in the fragments file.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    binwidth
        Bin width for the histogram.
    max_length
        Maximum length of fragments to plot.

    Returns
    -------
    Altair chart object if altair is installed, otherwise a pandas DataFrame with fragment data

    Notes
    -----
    This function requires the altair package to be installed. If not available,
    it will return a pandas DataFrame with the processed fragment data.
    """
    if not _has_altair:
        raise ImportError(
            "Altair is not installed. Install with 'pip install altair'. "
            "Returning DataFrame with fragment data instead of chart."
        )

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    fragment_path = adata.uns["files"]["fragments"]
    fragments = tools.fetch_in_regions(fragment_path=fragment_path, regions=region)

    # Calculate fragment lengths
    fragments = fragments.with_columns((pl.col("end") - pl.col("start")).alias("length"))

    # Get obs data as polars DataFrame
    if barcodes and barcodes in adata.obs.columns:
        obs_df = pl.from_pandas(adata.obs.reset_index().rename(columns={adata.obs.index.name or 'index': 'original_index'}))
        join_key = barcodes
    else:
        obs_df = pl.from_pandas(adata.obs.reset_index())
        join_key = adata.obs.index.name or 'index'

    # Join with obs data
    fragments = fragments.join(
        obs_df,
        left_on="name",
        right_on=join_key,
        how="inner"
    )

    # Convert to pandas for plotting
    fragments_pd = fragments.to_pandas()

    # Filter to fragments <= max_lengthbp
    if max_length:
        fragments_pd = fragments_pd[fragments_pd['length'] <= max_length]

    # If altair is not available, return the processed data
    if not _has_altair:
        return fragments_pd

    # Default chart settings
    base_chart = alt.Chart(fragments_pd)

    if groupby is not None:
        if isinstance(groupby, str):
            # Create a chart with faceting by the groupby column
            chart = base_chart.mark_bar().encode(
                x=alt.X('length:Q', bin=alt.Bin(step=binwidth), title='Fragment length (bp)'),
                y=alt.Y('count()', title='Count'),
                color=alt.Color(f'{groupby}:N', title=groupby),
                tooltip=['count()', f'{groupby}:N']
            ).properties(
                width=width,
                height=height,
                title=f"Fragment Length Distribution by {groupby}"
            ).facet(
                column=f'{groupby}:N'
            )
        else:
            # If groupby is a list or other iterable but not a string
            # Only use the first element for now (similar to mpl version's behavior)
            if isinstance(groupby, Iterable) and not isinstance(groupby, str):
                groupby = groupby[0]
                chart = base_chart.mark_bar().encode(
                    x=alt.X('length:Q', bin=alt.Bin(step=binwidth), title='Fragment length (bp)'),
                    y=alt.Y('count()', title='Count'),
                    color=alt.Color(f'{groupby}:N', title=groupby),
                    tooltip=['count()', f'{groupby}:N']
                ).properties(
                    width=width,
                    height=height,
                    title=f"Fragment Length Distribution by {groupby}"
                ).facet(
                    column=f'{groupby}:N'
                )
    else:
        # Simple histogram without faceting
        chart = base_chart.mark_bar().encode(
            x=alt.X('length:Q', bin=alt.Bin(step=binwidth), title='Fragment length (bp)'),
            y=alt.Y('count()', title='Count'),
            tooltip=['count()']
        ).properties(
            width=width,
            height=height,
            title="Fragment Length Distribution"
        )

    return chart
