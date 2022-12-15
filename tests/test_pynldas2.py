"""Tests for the package PyNLDAS2."""
import io

import pynldas2 as nldas


def test_show_versions():
    f = io.StringIO()
    nldas.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
