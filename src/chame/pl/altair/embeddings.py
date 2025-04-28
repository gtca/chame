from collections.abc import Sequence

import altair as alt
import pandas as pd
from anndata import AnnData
from scanpy.plotting._utils import savefig_or_show
from scipy.sparse import issparse


def embeddings(
    data: AnnData,
    basis: str,
    color: str | Sequence[str] | None = None,
    components: str | Sequence[int] = (0, 1),
    width: int = 400,
    height: int = 400,
    color_map: dict | None = None,
    opacity: float = 1.0,
    size: int = 20,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs
) -> alt.Chart:
    """
    Plot embeddings using Altair.

    Parameters
    ----------
    data
        AnnData object containing the embedding in .obsm['X_{basis}'].
    basis
        Key for the embedding in .obsm, e.g., 'umap', 'tsne', 'pca', etc.
    color
        Keys for annotations of observations/cells or variables/genes.
    components
        Components to plot, e.g. (0, 1) means x=component 0, y=component 1.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    color_map
        Dictionary mapping categories to colors for categorical annotations.
    opacity
        Opacity for the plot.
    size
        Size for the plot.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    **kwargs
        Additional arguments to pass to Altair Chart.

    Returns
    -------
    Altair Chart object if show is False.
    """
    # Extract embedding data
    basis_key = f"X_{basis}"
    if basis_key not in data.obsm:
        raise ValueError(f"{basis_key} not found in .obsm")

    # Prepare the data
    embedding = data.obsm[basis_key]
    if issparse(embedding):
        embedding = embedding.toarray()

    comp1, comp2 = components
    df = pd.DataFrame({
        f"{basis}_1": embedding[:, comp1],
        f"{basis}_2": embedding[:, comp2],
    }, index=data.obs_names)

    # Add color data if provided
    if color is not None:
        if isinstance(color, str):
            color = [color]

        # Process each color key
        for c in color:
            if c in data.obs.columns:
                df[c] = data.obs[c].values
            elif c in data.var_names:
                if data.raw is not None:
                    x = data.raw[:, c].X
                else:
                    x = data[:, c].X
                x = x.toarray() if issparse(x) else x
                df[c] = x.flatten()

    # Reset index to get a clean dataframe for Altair
    df = df.reset_index()

    # Create basic chart
    chart = alt.Chart(df).mark_point(opacity=opacity, size=size)

    # Create multiple charts if coloring by multiple variables
    charts = []
    if color is not None:
        for c in color:
            # Determine if variable is categorical or continuous
            if pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
                # Categorical coloring
                color_encoding = alt.Color(c, type='nominal', title=c)
                if color_map is not None and c in color_map:
                    color_encoding = alt.Color(c, type='nominal', scale=alt.Scale(domain=list(color_map[c].keys()),
                                                                               range=list(color_map[c].values())),
                                            title=c)
            else:
                # Continuous coloring
                color_encoding = alt.Color(c, type='quantitative', title=c)

            # Set up properties with title handling
            props = {'width': width, 'height': height}
            if title is not None or c is not None:
                # Only add title if either title or c is not None
                chart_title = ""
                if title is not None:
                    chart_title = title
                if c is not None:
                    if chart_title:
                        chart_title += f" {c}"
                    else:
                        chart_title = c
                props['title'] = chart_title

            c_chart = chart.encode(
                x=alt.X(f"{basis}_1", title=f"{basis.upper()} {comp1+1}"),
                y=alt.Y(f"{basis}_2", title=f"{basis.upper()} {comp2+1}"),
                color=color_encoding,
                tooltip=['index', f"{basis}_1", f"{basis}_2", c]
            ).properties(**props)

            charts.append(c_chart)

        # Combine charts
        if len(charts) > 1:
            full_chart = alt.hconcat(*charts)
        else:
            full_chart = charts[0]
    else:
        # No color, simple chart
        # Set up properties with title handling
        props = {'width': width, 'height': height}
        if title is not None:
            props['title'] = title

        full_chart = chart.encode(
            x=alt.X(f"{basis}_1", title=f"{basis.upper()} {comp1+1}"),
            y=alt.Y(f"{basis}_2", title=f"{basis.upper()} {comp2+1}"),
            tooltip=['index', f"{basis}_1", f"{basis}_2"]
        ).properties(**props)

    if save:
        if isinstance(save, str):
            filename = f"{basis}_{save}"
        else:
            filename = f"{basis}"
        savefig_or_show(filename, show=False, save=filename)

    # Return the chart or display it
    if show is None or show:
        return full_chart.display()
    else:
        return full_chart

def pca(
    data: AnnData,
    components: list[int] | tuple[int, int] = (0, 1),
    color: str | Sequence[str] | None = None,
    width: int = 400,
    height: int = 400,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs
) -> alt.Chart:
    """
    Plot PCA embedding using Altair.

    Parameters
    ----------
    data
        AnnData object containing the PCA embedding in .obsm['X_pca'].
    components
        Components to plot, e.g. (0, 1) means x=component 1, y=component 2.
    color
        Keys for annotations of observations/cells or variables/genes.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    **kwargs
        Additional arguments to pass to Altair Chart.

    Returns
    -------
    Altair Chart object if show is False.
    """
    return embeddings(
        data=data,
        basis="pca",
        components=components,
        color=color,
        title=title,
        width=width,
        height=height,
        show=show,
        save=save,
        **kwargs
    )

def umap(
    data: AnnData,
    color: str | Sequence[str] | None = None,
    width: int = 400,
    height: int = 400,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs
) -> alt.Chart:
    """
    Plot UMAP embedding using Altair.

    Parameters
    ----------
    data
        AnnData object containing the UMAP embedding in .obsm['X_umap'].
    color
        Keys for annotations of observations/cells or variables/genes.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    **kwargs
        Additional arguments to pass to Altair Chart.

    Returns
    -------
    Altair Chart object if show is False.
    """
    return embeddings(
        data=data,
        basis="umap",
        components=(0, 1),
        color=color,
        title=title,
        width=width,
        height=height,
        show=show,
        save=save,
        **kwargs
    )

def tsne(
    data: AnnData,
    color: str | Sequence[str] | None = None,
    width: int = 400,
    height: int = 400,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs
) -> alt.Chart:
    """
    Plot t-SNE embedding using Altair.

    Parameters
    ----------
    data
        AnnData object containing the t-SNE embedding in .obsm['X_tsne'].
    color
        Keys for annotations of observations/cells or variables/genes.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    **kwargs
        Additional arguments to pass to Altair Chart.

    Returns
    -------
    Altair Chart object if show is False.
    """
    return embeddings(
        data=data,
        basis="tsne",
        components=(0, 1),
        color=color,
        title=title,
        width=width,
        height=height,
        show=show,
        save=save,
        **kwargs
    )

def lsi(
    data: AnnData,
    components: list[int] | tuple[int, int] = (1, 2),
    color: str | Sequence[str] | None = None,
    width: int = 400,
    height: int = 400,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs
) -> alt.Chart:
    """
    Plot LSI embedding using Altair.

    Parameters
    ----------
    data
        AnnData object containing the LSI embedding in .obsm['X_lsi'].
    components
        Components to plot, e.g. (1, 2) means x=component 1, y=component 2.
        LSI component indexing is 1-based by convention.
    color
        Keys for annotations of observations/cells or variables/genes.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    **kwargs
        Additional arguments to pass to Altair Chart.

    Returns
    -------
    Altair Chart object if show is False.
    """
    # Adjust component indices (LSI components are 1-based for users)
    dims = []
    for i in range(len(components)):
        # Convert 1-based to 0-based indexing
        if components[i] > 0:
            dims.append(components[i] - 1)
        else:
            dims.append(components[i])

    return embeddings(
        data=data,
        basis="lsi",
        components=dims,
        color=color,
        title=title,
        width=width,
        height=height,
        show=show,
        save=save,
        **kwargs
    )

def svd(
    data: AnnData,
    components: list[int] | tuple[int, int] = (0, 1),
    color: str | Sequence[str] | None = None,
    width: int = 400,
    height: int = 400,
    title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs
) -> alt.Chart:
    """
    Plot SVD embedding using Altair.

    Parameters
    ----------
    data
        AnnData object containing the SVD embedding in .obsm['X_svd'].
    components
        Components to plot, e.g. (0, 1) means x=component 1, y=component 2.
    color
        Keys for annotations of observations/cells or variables/genes.
    width
        Width of the plot in pixels.
    height
        Height of the plot in pixels.
    title
        Title for the plot.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    **kwargs
        Additional arguments to pass to Altair Chart.

    Returns
    -------
    Altair Chart object if show is False.
    """
    return embeddings(
        data=data,
        basis="svd",
        components=components,
        color=color,
        title=title,
        width=width,
        height=height,
        show=show,
        save=save,
        **kwargs
    )
