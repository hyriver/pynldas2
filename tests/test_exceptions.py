"""Tests for exceptions and requests"""
import pytest
from shapely.geometry import Polygon

import pynldas2 as nldas
from pynldas2 import InputRangeError, InputTypeError, InputValueError

GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
VAR = ["prcp", "pet"]
DEF_CRS = 4326
ALT_CRS = 3542
COORDS = (-1431147.7928, 318483.4618)
START = "2000-01-01"
END = "2000-01-12"


def test_coords_range_coords():
    with pytest.raises(InputRangeError) as ex:
        _ = nldas.get_bycoords(COORDS, START, END, crs=DEF_CRS, variables=VAR)
    assert "range" in str(ex.value)


def test_coords_range_start():
    with pytest.raises(InputRangeError) as ex:
        _ = nldas.get_bycoords(COORDS, "1970-01-01", END, crs=ALT_CRS, variables=VAR)
    assert "start_date" in str(ex.value)


def test_coords_range_end():
    with pytest.raises(InputRangeError) as ex:
        _ = nldas.get_bycoords(COORDS, START, "1970-01-01", crs=ALT_CRS, variables=VAR)
    assert "end_date" in str(ex.value)


def test_coords_range_end_before_start():
    with pytest.raises(InputRangeError) as ex:
        _ = nldas.get_bycoords(COORDS, START, "1980-01-01", crs=ALT_CRS, variables=VAR)
    assert "after start_date" in str(ex.value)


def test_coords_type():
    with pytest.raises(InputTypeError) as ex:
        _ = nldas.get_bycoords(COORDS[0], START, END, crs=ALT_CRS, variables=VAR)
    assert "tuple" in str(ex.value)


def test_coords_type_len():
    with pytest.raises(InputTypeError) as ex:
        _ = nldas.get_bycoords(COORDS + COORDS, START, END, crs=ALT_CRS, variables=VAR)
    assert "length 2" in str(ex.value)


def test_coords_var():
    with pytest.raises(InputValueError) as ex:
        _ = nldas.get_bycoords(COORDS, START, END, crs=ALT_CRS, variables="tmin")
    assert "prcp" in str(ex.value)
