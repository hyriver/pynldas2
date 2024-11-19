"""Top-level package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pynldas2 import exceptions
from pynldas2.print_versions import show_versions
from pynldas2.pynldas2 import get_bycoords, get_bygeom, get_grid_mask

try:
    __version__ = version("pynldas2")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "__version__",
    "exceptions",
    "get_bycoords",
    "get_bygeom",
    "get_grid_mask",
    "show_versions",
]
