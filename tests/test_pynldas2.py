"""Tests for the package PyNLDAS2."""
import io
import os

import numpy as np
import pytest
from shapely import Point, Polygon

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
CONN = 1 if int(os.environ.get("GH_CI", 0)) else 4


def assert_close(a: float, b: float, rtol: float = 1e-2) -> bool:
    assert np.isclose(a, b, rtol=rtol).all()


def test_coords():
    clm = nldas.get_bycoords(COORDS, START, END, crs=ALT_CRS, variables=VAR, n_conn=CONN)
    assert_close(clm.prcp.mean(), 0.0051)
    assert_close(clm.pet.mean(), 0.1346)


def test_coords_xr():
    clm = nldas.get_bycoords(
        COORDS, START, END, crs=ALT_CRS, variables="prcp", to_xarray=True, n_conn=CONN
    )
    assert_close(clm.prcp.mean(), 0.0051)


def test_geom():
    clm = nldas.get_bygeom(GEOM, START, END, DEF_CRS, VAR, n_conn=CONN)
    assert_close(clm.prcp.mean(), 0.1534)
    assert_close(clm.pet.mean(), 0.0418)


def test_geom_box():
    clm = nldas.get_bygeom(GEOM.bounds, START, END, DEF_CRS, "prcp", n_conn=CONN)
    assert_close(clm.prcp.mean(), 0.1534)


@pytest.mark.speedup
def test_snow():
    clm = nldas.get_bycoords(
        (-89.6, 48.3), "2000-01-01", "2000-01-02", crs=DEF_CRS, variables="prcp", snow=True
    )
    assert_close(clm.snow.mean(), 0.0017)
    clm = nldas.get_bygeom(
        Point(-89.6, 48.3).buffer(0.05),
        "2000-01-01",
        "2000-01-02",
        DEF_CRS,
        "prcp",
        snow=True,
    )
    assert_close(clm.snow.mean().compute().item(), 0.00163)


def test_show_versions():
    f = io.StringIO()
    nldas.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()
