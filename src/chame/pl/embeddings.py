import numpy as np
import scanpy as sc
from anndata import AnnData


def lsi(
    data: AnnData,
    components=[1, 2],
    color=None,
    title=None,
    show=True,
    save=None,
    **kwargs,
):
    """
    Plot LSI embeddings.

    PARAMETERS
    ----------
    data:
        AnnData object
    components: list of int (default: [1,2])
        Which LSI components to plot
    color: str or list of str (default: None)
        Keys for annotations to color the plot
    title: str (default: None)
        Title for the plot
    show: bool (default: True)
        Show the plot
    save: str (default: None)
        Path to save the plot
    **kwargs:
        Additional arguments passed to scanpy.pl.embedding()
    """
    if not isinstance(data, AnnData):
        raise TypeError("Expected AnnData")

    if "X_lsi" not in data.obsm:
        raise ValueError("LSI embeddings not found. Run tl.lsi() first.")

    # Adjust component indices (LSI components are 1-based for users)
    dims = []
    for i in range(len(components)):
        # Convert 1-based to 0-based indexing
        if components[i] > 0:
            dims.append(components[i] - 1)
        else:
            dims.append(components[i])

    # Check that component indices are valid
    if np.max(dims) >= data.obsm["X_lsi"].shape[1]:
        raise ValueError(f"Selected components must be between 1 and {data.obsm['X_lsi'].shape[1]}")

    # Convert to the format expected by scanpy.pl.embedding
    dim_tuple = tuple(dims)

    return sc.pl.embedding(
        data,
        basis="lsi",
        color=color,
        dimensions=[dim_tuple],
        title=title,
        show=show,
        save=save,
        **kwargs
    )

def svd(
    data: AnnData,
    components=[1, 2],
    color=None,
    title=None,
    show=True,
    save=None,
    **kwargs,
):
    """
    Plot SVD embeddings.
    """
    if not isinstance(data, AnnData):
        raise TypeError("Expected AnnData")

    if "X_svd" not in data.obsm:
        raise ValueError("SVD embeddings not found. Run tl.svd() first.")

    # Adjust component indices (SVD components are 1-based for users)
    dims = []
    for i in range(len(components)):
        # Convert 1-based to 0-based indexing
        if components[i] > 0:
            dims.append(components[i] - 1)
        else:
            dims.append(components[i])

    # Check that component indices are valid
    if np.max(dims) >= data.obsm["X_svd"].shape[1]:
        raise ValueError(f"Selected components must be between 1 and {data.obsm['X_svd'].shape[1]}")

    # Convert to the format expected by scanpy.pl.embedding
    dim_tuple = tuple(dims)

    return sc.pl.embedding(
        data,
        basis="svd",
        color=color,
        dimensions=[dim_tuple],
        title=title,
        show=show,
        save=save,
        **kwargs
    )
