from collections.abc import Sequence

import altair as alt
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scanpy.plotting._utils import savefig_or_show
from scipy.sparse import issparse


def histogram(
    data: AnnData | MuData,
    keys: str | Sequence[str],
    groupby: str | Sequence[str] | None = None,
    bins: int = 100,
    kde: bool = False,
    bandwidth: float = 100,
    width: int = 350,
    height: int = 250,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
) -> alt.Chart:
    """
    Plot histograms of specified features using Altair.

    Parameters
    ----------
    data
        AnnData object with feature counts or multimodal MuData object.
    keys
        Keys to plot. These can be column names in .obs or features in .var_names.
    groupby
        Column name(s) of .obs slot of the AnnData object according to which the plot is split.
    bins
        Number of bins for the histogram.
    kde
        Whether to overlay a kernel density estimate.
    bandwidth
        Bandwidth for the KDE.
    width
        Width of each plot in pixels.
    height
        Height of each plot in pixels.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.

    Returns
    -------
    Altair Chart object if show is False.
    """
    # Input validation
    if not isinstance(data, AnnData | MuData):
        raise TypeError("Expected AnnData or MuData object")

    # Normalize keys to list
    if isinstance(keys, str):
        keys = [keys]

    # Normalize groupby to list
    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
        if len(groupby) > 2:
            raise ValueError("Maximum 2 categories in groupby")
        if len(groupby) == 2 and len(keys) > 1:
            raise ValueError("Maximum 1 key when using 2 groupby variables")

    # Prepare the dataframe with all needed data
    df = prepare_dataframe(data, keys, groupby)

    # Create charts based on groupby settings
    if groupby is None:
        chart = create_simple_histograms(df, keys, bins, kde, bandwidth, width, height)
    elif len(groupby) == 1:
        chart = create_grouped_histograms(df, keys, groupby[0], bins, kde, bandwidth, width, height)
    else:  # len(groupby) == 2
        chart = create_double_grouped_histogram(df, keys[0], groupby[0], groupby[1], bins, kde, bandwidth, width, height)

    # Set title if provided
    if title is not None:
        chart = chart.properties(title=title)

    # Handle save functionality
    if save:
        filename = f"histogram_{save}" if isinstance(save, str) else "histogram"
        savefig_or_show(filename, show=False, save=filename)

    # Return or display the chart
    if show is None or show:
        return chart.display()
    else:
        return chart

def prepare_dataframe(data, keys, groupby=None):
    """Prepare dataframe with all required data for plotting."""
    # Find which keys are in obs columns vs. var_names
    obs_keys = []
    var_keys = []

    if isinstance(data, MuData):
        # For MuData, check each modality
        # FIXME: parse modality prefixes
        obs_keys = [k for k in keys if k in data.obs.columns]
        var_keys = [k for k in keys if any(k in data.mod[m].var_names for m in data.mod)]
    else:  # AnnData
        obs_keys = [k for k in keys if k in data.obs.columns]
        var_keys = [k for k in keys if k in data.var_names]

    if len(obs_keys) + len(var_keys) != len(keys):
        raise ValueError("Keys should be columns of .obs or some of .var_names")

    # Start with obs keys
    result = data.obs[obs_keys].copy() if obs_keys else pd.DataFrame(index=data.obs_names)

    # Add var keys
    if var_keys:
        if isinstance(data, MuData):
            # Extract features from appropriate modalities
            for m in data.mod:
                mod_keys = [k for k in var_keys if k in data.mod[m].var_names]
                if not mod_keys:
                    continue

                adata = data.mod[m]
                x = adata.raw[:, mod_keys].X if adata.raw is not None else adata[:, mod_keys].X
                x = x.toarray() if issparse(x) else x
                x_df = pd.DataFrame(x, index=adata.obs_names, columns=mod_keys)
                result = pd.concat([result, x_df], axis=1)
        else:  # AnnData
            x = data.raw[:, var_keys].X if data.raw is not None else data[:, var_keys].X
            x = x.toarray() if issparse(x) else x
            x_df = pd.DataFrame(x, index=data.obs_names, columns=var_keys)
            result = pd.concat([result, x_df], axis=1)

    # Add groupby columns if needed
    if groupby:
        for g in groupby:
            if g not in result.columns:
                result[g] = data.obs[g]

    return result

def create_kde_layer(chart_data, field, color="steelblue", bandwidth: float | None = None):
    """Create a KDE layer for a given field."""
    # Create density transform
    kde = chart_data.transform_density(
        field,
        as_=[field, "density"],
        bandwidth=bandwidth
    ).mark_line(color=color).encode(
        x=f"{field}:Q",
        y="density:Q"
    )
    return kde

def create_simple_histograms(df, keys, bins, kde=False, bandwidth: float | None = None, width=350, height=250):
    """Create simple histograms for each key."""
    charts = []

    for key in keys:
        # Skip if key missing or all NaN
        if key not in df.columns or df[key].isna().all():
            continue

        # Filter data for this key
        chart_data = alt.Chart(df).transform_filter(
            alt.datum[key] is not None  # Filter out None/NaN values
        )

        # Create histogram
        hist = chart_data.mark_bar(opacity=0.7).encode(
            alt.X(f'{key}:Q', bin=alt.Bin(maxbins=bins), title=key),
            alt.Y('count()', title='Count'),
            tooltip=[alt.Tooltip(f'{key}:Q', bin=alt.Bin(maxbins=bins)), 'count()']
        )

        final_chart = hist

        # Add KDE if requested
        if kde:
            # Estimate a reasonable bandwidth based on data range
            values = df[key].dropna()
            if len(values) > 1:
                data_range = values.max() - values.min()
                if bandwidth is None:
                    bandwidth = data_range / bins * 0.5  # A heuristic for reasonable bandwidth

                # Create KDE layer
                kde_layer = create_kde_layer(chart_data, key, bandwidth=bandwidth)

                # Create a dual-axis chart
                # The KDE uses a different y-axis scale than the histogram
                base = alt.layer(
                    hist,
                    kde_layer.encode(y=alt.Y('density:Q', title='Density', axis=alt.Axis(grid=False)))
                ).resolve_scale(y='independent')

                final_chart = base

        # Add title and dimensions
        final_chart = final_chart.properties(
            width=width,
            height=height,
            title=key
        )

        charts.append(final_chart)

    # Return empty chart if no valid data
    if not charts:
        return alt.Chart().mark_text(text='No data to display')

    # Combine charts horizontally
    return alt.hconcat(*charts)

def create_grouped_histograms(df, keys, groupby, bins, kde=False, bandwidth: float | None = None, width=350, height=250):
    """Create histograms for each key, with bars colored by groupby variable."""
    charts = []

    for key in keys:
        # Skip if key missing or all NaN
        if key not in df.columns or df[key].isna().all():
            continue

        # Create histogram with color encoding for group
        chart_data = alt.Chart(df).transform_filter(
            (alt.datum[key] is not None) & (alt.datum[groupby] is not None)  # Filter out None/NaN values
        )

        hist = chart_data.mark_bar(opacity=0.7).encode(
            alt.X(f'{key}:Q', bin=alt.Bin(maxbins=bins), title=key),
            alt.Y('count()', title='Count'),
            alt.Color(f'{groupby}:N', title=groupby),
            tooltip=[
                alt.Tooltip(f'{key}:Q', bin=alt.Bin(maxbins=bins)),
                'count()',
                alt.Tooltip(f'{groupby}:N')
            ]
        )

        final_chart = hist

        # Add KDE if requested
        if kde:
            # Get unique group values
            group_values = df[groupby].dropna().unique()

            # Estimate bandwidth
            values = df[key].dropna()
            if len(values) > 1:
                data_range = values.max() - values.min()
                if bandwidth is None:
                    bandwidth = data_range / bins * 0.5

                # Create a KDE layer for each group
                kde_layers = []
                for group_val in group_values:
                    # Filter data for this group
                    group_data = alt.Chart(df).transform_filter(
                        (alt.datum[key] is not None) & (alt.datum[groupby] == group_val)
                    )

                    # Get group color from the scale

                    # Create KDE
                    kde_layer = group_data.transform_density(
                        key,
                        as_=[key, "density"],
                        bandwidth=bandwidth
                    ).mark_line().encode(
                        x=f"{key}:Q",
                        y=alt.Y("density:Q", title="Density"),
                        color=alt.Color(f'{groupby}:N', title=groupby)
                    )

                    kde_layers.append(kde_layer)

                # Combine all KDE layers
                if kde_layers:
                    kde_chart = alt.layer(*kde_layers)

                    # Create dual-axis chart
                    base = alt.layer(
                        hist,
                        kde_chart
                    ).resolve_scale(y='independent')

                    final_chart = base

        # Add title and dimensions
        final_chart = final_chart.properties(
            width=width,
            height=height,
            title=key
        )

        charts.append(final_chart)

    # Return empty chart if no valid data
    if not charts:
        return alt.Chart().mark_text(text='No data to display')

    # Combine charts horizontally
    return alt.hconcat(*charts)

def create_double_grouped_histogram(df, key, groupby1, groupby2, bins, kde=False, bandwidth: float | None = None, width=350, height=250):
    """Create histograms for one key, with separate charts for each value of groupby1,
    and bars colored by groupby2."""
    # Skip if key missing or all NaN
    if key not in df.columns or df[key].isna().all():
        return alt.Chart().mark_text(text='No data to display')

    # Get unique values for first groupby (for faceting)
    group_values = df[groupby1].dropna().unique()
    if len(group_values) == 0:
        return alt.Chart().mark_text(text='No data to display')

    charts = []
    for group_val in group_values:
        # Filter data for this group value
        chart_data = alt.Chart(df).transform_filter(
            (alt.datum[key] is not None) &
            (alt.datum[groupby1] == group_val) &
            (alt.datum[groupby2] is not None)
        )

        # Create histogram
        hist = chart_data.mark_bar(opacity=0.7).encode(
            alt.X(f'{key}:Q', bin=alt.Bin(maxbins=bins), title=key),
            alt.Y('count()', title='Count'),
            alt.Color(f'{groupby2}:N', title=groupby2),
            tooltip=[
                alt.Tooltip(f'{key}:Q', bin=alt.Bin(maxbins=bins)),
                'count()',
                alt.Tooltip(f'{groupby1}:N'),
                alt.Tooltip(f'{groupby2}:N')
            ]
        )

        final_chart = hist

        # Add KDE if requested
        if kde:
            # Get unique group2 values
            group2_values = df[df[groupby1] == group_val][groupby2].dropna().unique()

            # Estimate bandwidth
            values = df[df[groupby1] == group_val][key].dropna()
            if len(values) > 1:
                data_range = values.max() - values.min()
                if bandwidth is None:
                    bandwidth = data_range / bins * 0.5

                # Create a KDE layer for each group2 value
                kde_layers = []
                for group2_val in group2_values:
                    # Filter data for this group combination
                    subgroup_data = alt.Chart(df).transform_filter(
                        (alt.datum[key] is not None) &
                        (alt.datum[groupby1] == group_val) &
                        (alt.datum[groupby2] == group2_val)
                    )

                    # Create KDE
                    kde_layer = subgroup_data.transform_density(
                        key,
                        as_=[key, "density"],
                        bandwidth=bandwidth
                    ).mark_line().encode(
                        x=f"{key}:Q",
                        y=alt.Y("density:Q", title="Density"),
                        color=alt.Color(f'{groupby2}:N', title=groupby2)
                    )

                    kde_layers.append(kde_layer)

                # Combine all KDE layers
                if kde_layers:
                    kde_chart = alt.layer(*kde_layers)

                    # Create dual-axis chart
                    base = alt.layer(
                        hist,
                        kde_chart
                    ).resolve_scale(y='independent')

                    final_chart = base

        # Add title and dimensions
        final_chart = final_chart.properties(
            width=width,
            height=height,
            title=f"{groupby1}: {group_val}"
        )

        charts.append(final_chart)

    # Return empty chart if no valid data
    if not charts:
        return alt.Chart().mark_text(text='No data to display')

    # Combine charts horizontally
    return alt.hconcat(*charts)
