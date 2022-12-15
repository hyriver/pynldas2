"""Top-level package."""
from importlib.metadata import PackageNotFoundError, version

from .pynldas2 import get_bycoords, get_bygeom
from .exceptions import InputTypeError, InputValueError, InputRangeError, NLDASServiceError
from .print_versions import show_versions

try:
    __version__ = version("pynldas2")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "get_bycoords",
    "get_bygeom",
    "InputTypeError",
    "InputValueError",
    "InputRangeError",
    "show_versions",
    "InputTypeError",
    "InputValueError",
    "InputRangeError",
    "NLDASServiceError",
]
