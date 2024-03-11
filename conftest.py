"""Configuration for pytest."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _add_standard_imports(doctest_namespace):
    """Add async_retriever namespace for doctest."""
    import pynldas2 as nldas

    doctest_namespace["nldas"] = nldas
