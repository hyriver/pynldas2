"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add async_retriever namespace for doctest."""
    import pynldas2 as nldas

    doctest_namespace["nldas"] = nldas
