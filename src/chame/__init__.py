"""Chromatin analysis module"""

try:  # See https://github.com/maresb/hatch-vcs-footgun-example
    from setuptools_scm import get_version

    __version__ = get_version(root="../..", relative_to=__file__)
except (ImportError, LookupError):
    try:
        from ._version import __version__
    except ModuleNotFoundError:
        raise RuntimeError("chame is not correctly installed. Please install it, e.g. with pip.")


from . import fragments, io, pl, pp, ranges, tl, utils
from .io.read_10x import read_10x, read_10x_h5, read_10x_mtx

__all__ = ["io", "pl", "pp", "tl", "utils", "read_10x", "fragments", "ranges"]
