"""Tests for the package PyNLDAS2."""
import io

import numpy as np
from shapely.geometry import Polygon

import pynldas2 as nldas

GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
VAR = ["prcp", "pet"]
DEF_CRS = 4326
ALT_CRS = 3542
COORDS = (-1431147.7928, 318483.4618)
START = "2000-01-01"
END = "2000-01-12"


def assert_close(a: float, b: float, rtol: float = 1e-2) -> bool:
    assert np.isclose(a, b, rtol=rtol).all()


def test_coords():
    clm = nldas.get_bycoords(COORDS, START, END, ALT_CRS, VAR)
    assert_close(clm.prcp.mean(), 0.0051)
    assert_close(clm.pet.mean(), 0.1346)


def test_coords_xr():
    clm = nldas.get_bycoords(COORDS, START, END, ALT_CRS, VAR, True)
    assert_close(clm.prcp.mean(), 0.0051)
    assert_close(clm.pet.mean(), 0.1346)


def test_geom():
    clm = nldas.get_bygeom(GEOM, START, END, DEF_CRS, VAR)
    assert_close(clm.prcp.mean(), 0.1534)
    assert_close(clm.pet.mean(), 0.0418)


def test_show_versions():
    f = io.StringIO()
    nldas.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
