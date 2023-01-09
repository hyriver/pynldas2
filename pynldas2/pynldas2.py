"""Get hourly NLDAS2 forcing data."""
from __future__ import annotations

import functools
import itertools
import re
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Union

import async_retriever as ar
import pandas as pd
import pygeoutils as hgu
import pyproj
import rioxarray  # pyright: reportUnusedImport=false
import xarray as xr
from numpy.core._exceptions import UFuncTypeError
from pandas.errors import EmptyDataError

from .exceptions import InputRangeError, InputTypeError, InputValueError, NLDASServiceError

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

CRSTYPE = Union[int, str, pyproj.CRS]
URL = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"
NLDAS_VARS = {
    "prcp": {"nldas_name": "APCPsfc", "long_name": "Precipitation hourly total", "units": "mm"},
    "pet": {"nldas_name": "PEVAPsfc", "long_name": "Potential evaporation", "units": "mm"},
    "temp": {"nldas_name": "TMP2m", "long_name": "2-m above ground temperature", "units": "K"},
    "wind_u": {
        "nldas_name": "UGRD10m",
        "long_name": "10-m above ground zonal wind",
        "units": "m/s",
    },
    "wind_v": {
        "nldas_name": "VGRD10m",
        "long_name": "10-m above ground meridional wind",
        "units": "m/s",
    },
    "rlds": {
        "nldas_name": "DLWRFsfc",
        "long_name": "Surface DW longwave radiation flux",
        "units": "W/m^2",
    },
    "rsds": {
        "nldas_name": "DSWRFsfc",
        "long_name": "Surface DW shortwave radiation flux",
        "units": "W/m^2",
    },
    "humidity": {
        "nldas_name": "SPFH2m",
        "long_name": "2-m above ground specific humidity",
        "units": "kg/kg",
    },
}
DATE_COL = "Date&Time"
DATE_FMT = "%Y-%m-%dT%H"
__all__ = ["get_bycoords", "get_grid_mask", "get_bygeom"]


def _txt2df(txt: str, resp_id: int, kwds: list[dict[str, dict[str, str]]]) -> pd.Series:
    """Convert text to dataframe."""
    try:
        data = pd.read_csv(StringIO(txt), skiprows=39, delim_whitespace=True).dropna()
        data.index = pd.to_datetime(data.index + " " + data[DATE_COL])
    except EmptyDataError:
        return pd.Series(name=kwds[resp_id]["params"]["variable"].split(":")[-1])
    except UFuncTypeError as ex:
        msg = "".join(re.findall("<strong>(.*?)</strong>", txt, re.DOTALL)).strip()
        raise NLDASServiceError(msg) from ex

    data = data.drop(columns=DATE_COL)
    data.index.freq = data.index.inferred_freq
    data = data["Data"]
    data.name = kwds[resp_id]["params"]["variable"].split(":")[-1]
    return data


def _check_inputs(
    start_date: str,
    end_date: str,
    variables: str | list[str] | None = None,
) -> tuple[list[pd.Timestamp], list[str]]:
    """Check inputs."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta("1D")
    if start < pd.to_datetime("1979-01-01T13"):
        raise InputRangeError("start_date", "1979-01-01 to yesterday")
    if end > pd.Timestamp.now() - pd.Timedelta("1D"):
        raise InputRangeError("end_date", "1979-01-01 to yesterday")
    if end <= start:
        raise InputRangeError("end_date", "after start_date")

    dates = pd.date_range(start, end, freq="10000D").tolist()
    dates = dates + [end] if dates[-1] < end else dates

    if variables is None:
        clm_vars = [f"NLDAS:NLDAS_FORA0125_H.002:{d['nldas_name']}" for d in NLDAS_VARS.values()]
    else:
        clm_vars = [variables] if isinstance(variables, str) else list(variables)
        if any(v not in NLDAS_VARS for v in clm_vars):
            raise InputValueError("variables", list(NLDAS_VARS))
        clm_vars = [f"NLDAS:NLDAS_FORA0125_H.002:{NLDAS_VARS[v]['nldas_name']}" for v in clm_vars]

    return dates, clm_vars


def get_byloc(
    lon: float,
    lat: float,
    start_date: str,
    end_date: str,
    variables: str | list[str] | None = None,
    n_conn: int = 4,
) -> pd.DataFrame:
    """Get NLDAS climate forcing data for a single location.

    Parameters
    ----------
    lon : float
        Longitude of the location.
    lat : float
        Latitude of the location.
    start_date : str
        Start date of the data.
    end_date : str
        End date of the data.
    variables : str or list of str, optional
        Variables to download. If None, all variables are downloaded.
        Valid variables are: ``prcp``, ``pet``, ``temp``, ``wind_u``, ``wind_v``,
        ``rlds``, ``rsds``, and ``humidity``.
    n_conn : int, optional
        Number of parallel connections to use for retrieving data, defaults to 4.
        The maximum number of connections is 4, if more than 4 are requested, 4
        connections will be used.

    Returns
    -------
    pandas.DataFrame
        The requested data as a dataframe.
    """
    dates, clm_vars = _check_inputs(start_date, end_date, variables)
    kwds = [
        {
            "params": {
                "type": "asc2",
                "location": f"GEOM:POINT({lon}, {lat})",
                "variable": v,
                "startDate": s.strftime(DATE_FMT),
                "endDate": e.strftime(DATE_FMT),
            }
        }
        for (s, e), v in itertools.product(zip(dates[:-1], dates[1:]), clm_vars)
    ]

    n_conn = min(n_conn, 4)
    resp = ar.retrieve_text([URL] * len(kwds), kwds, max_workers=n_conn)

    clm_list = (_txt2df(txt, i, kwds) for i, txt in enumerate(resp))
    clm_merged = (
        pd.concat(df)
        for _, df in itertools.groupby(
            sorted(clm_list, key=lambda x: x.name), lambda x: x.name  # type: ignore[no-any-return]
        )
    )
    clm = pd.concat(clm_merged, axis=1)
    clm = clm.rename(columns={d["nldas_name"]: n for n, d in NLDAS_VARS.items()})
    return clm.loc[start_date:end_date]  # type: ignore[misc]


def _get_lon_lat(
    coords: list[tuple[float, float]] | tuple[float, float],
    crs: CRSTYPE = 4326,
) -> tuple[list[float], list[float]]:
    """Get longitude and latitude from a list of coordinates."""
    if not isinstance(coords, list) and not (isinstance(coords, tuple) and len(coords) == 2):
        raise InputTypeError("coords", "tuple of length 2 or a list of them")

    coords_list = coords if isinstance(coords, list) else [coords]
    if any(not (isinstance(c, tuple) and len(c) == 2) for c in coords_list):
        raise InputTypeError("coords", "tuple of length 2 or a list of them")

    xx, yy = zip(*coords_list)
    if pyproj.CRS(crs) == pyproj.CRS(4326):
        return list(xx), list(yy)

    project = pyproj.Transformer.from_crs(crs, 4326, always_xy=True).transform
    lons, lats = project(xx, yy)
    return list(lons), list(lats)


def get_bycoords(
    coords: list[tuple[float, float]],
    start_date: str,
    end_date: str,
    crs: CRSTYPE = 4326,
    variables: str | list[str] | None = None,
    to_xarray: bool = False,
    n_conn: int = 4,
) -> pd.DataFrame | xr.Dataset:
    """Get NLDAS climate forcing data for a list of coordinates.

    Parameters
    ----------
    coords : list of tuples
        List of (lon, lat) coordinates.
    start_date : str
        Start date of the data.
    end_date : str
        End date of the data.
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input coordinates, defaults to ``EPSG:4326``.
    variables : str or list of str, optional
        Variables to download. If None, all variables are downloaded.
        Valid variables are: ``prcp``, ``pet``, ``temp``, ``wind_u``, ``wind_v``,
        ``rlds``, ``rsds``, and ``humidity``.
    to_xarray : bool, optional
        If True, the data is returned as an xarray dataset.
    n_conn : int, optional
        Number of parallel connections to use for retrieving data, defaults to 4.
        The maximum number of connections is 4, if more than 4 are requested, 4
        connections will be used.

    Returns
    -------
    pandas.DataFrame
        The requested data as a dataframe.
    """
    lons, lats = _get_lon_lat(coords, crs)

    bounds = (-125.0, 25.0, -67.0, 53.0)
    points = hgu.Coordinates(lons, lats, bounds).points
    if len(points) == 0:
        raise InputRangeError("coords", f"{bounds}")

    coords_val = list(zip(points.x, points.y))
    nldas = functools.partial(
        get_byloc, variables=variables, start_date=start_date, end_date=end_date, n_conn=n_conn
    )
    clm = pd.concat(
        (nldas(lon=lon, lat=lat) for lon, lat in coords_val),
        keys=coords_val,
    )
    clm.index = clm.index.set_names(["lon", "lat", "time"])
    if to_xarray:
        clm = clm.reset_index()
        clm["time"] = clm["time"].dt.tz_localize(None)
        clm_ds = clm.set_index(["lon", "lat", "time"]).to_xarray()
        clm_ds.attrs["tz"] = "UTC"
        clm_ds = clm_ds.rio.write_crs(4326)
        for v in clm_ds.data_vars:
            clm_ds[v].attrs = NLDAS_VARS[v]
        return clm_ds
    return clm


def get_grid_mask():
    """Get the NLDAS2 grid that contains the land/water/soil/vegetation mask.

    Returns
    -------
    xarray.Dataset
        The grid mask.
    """
    url = "/".join(
        (
            "https://ldas.gsfc.nasa.gov/sites/default",
            "files/ldas/nldas/NLDAS_masks-veg-soil.nc4",
        )
    )
    resp = ar.retrieve_binary([url])
    grid = xr.open_dataset(BytesIO(resp[0]), engine="h5netcdf")
    grid = grid.rio.write_transform()
    grid = grid.rio.write_crs(4326)
    grid = grid.rio.write_coordinate_system()
    return grid


def _txt2da(txt: str, resp_id: int, kwds: list[dict[str, dict[str, str]]]) -> xr.DataArray:
    """Convert text to dataarray."""
    try:
        data = pd.read_csv(StringIO(txt), skiprows=39, delim_whitespace=True).dropna()
        data.index = pd.to_datetime(data.index + " " + data[DATE_COL])
    except EmptyDataError:
        return xr.DataArray(name=kwds[resp_id]["params"]["variable"].split(":")[-1])
    except UFuncTypeError as ex:
        msg = "".join(re.findall("<strong>(.*?)</strong>", txt, re.DOTALL)).strip()
        raise NLDASServiceError(msg) from ex

    data = data["Data"]
    data.name = kwds[resp_id]["params"]["variable"].split(":")[-1]
    data.index.name = "time"
    data.index = data.index.tz_localize(None)
    da = data.to_xarray()
    lon, lat = kwds[resp_id]["params"]["location"].split("(")[-1].strip(")").split(",")
    da = da.assign_coords(x=float(lon), y=float(lat))
    da = da.expand_dims("y").expand_dims("x")
    return da


def get_bygeom(
    geometry: Polygon | MultiPolygon | tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    geo_crs: CRSTYPE,
    variables: str | list[str] | None = None,
    n_conn: int = 4,
) -> xr.Dataset:
    """Get hourly NLDAS climate forcing within a geometry at 0.125 resolution.

    Parameters
    ----------
    geometry : shapely.Polygon, shapely.MultiPolygon, or tuple of length 4
        Input polygon or a bounding box like so (xmin, ymin, xmax, ymax).
    start_date : str
        Start date of the data.
    end_date : str
        End date of the data.
    geo_crs : int, str, or pyproj.CRS
        CRS of the input geometry
    variables : str or list of str, optional
        Variables to download. If None, all variables are downloaded.
        Valid variables are: ``prcp``, ``pet``, ``temp``, ``wind_u``, ``wind_v``,
        ``rlds``, ``rsds``, and ``humidity``.
    n_conn : int, optional
        Number of parallel connections to use for retrieving data, defaults to 4.
        It should be less than 4.

    Returns
    -------
    xarray.Dataset
        The requested forcing data.
    """
    dates, clm_vars = _check_inputs(start_date, end_date, variables)

    nldas_grid = get_grid_mask()
    geom = hgu.geo2polygon(geometry, geo_crs, nldas_grid.rio.crs)
    msk = nldas_grid.CONUS_mask.rio.clip([geom], all_touched=True)
    coords = itertools.product(msk.get_index("lon"), msk.get_index("lat"))
    kwds = [
        {
            "params": {
                "type": "asc2",
                "location": f"GEOM:POINT({lon}, {lat})",
                "variable": v,
                "startDate": s.strftime(DATE_FMT),
                "endDate": e.strftime(DATE_FMT),
            }
        }
        for (lon, lat), (s, e), v in itertools.product(coords, zip(dates[:-1], dates[1:]), clm_vars)
    ]

    n_conn = min(n_conn, 4)
    resp = ar.retrieve_text([URL] * len(kwds), kwds, max_workers=n_conn)

    clm = xr.merge(_txt2da(txt, i, kwds) for i, txt in enumerate(resp))
    clm = clm.rename({d["nldas_name"]: n for n, d in NLDAS_VARS.items() if d["nldas_name"] in clm})
    clm = clm.sel(time=slice(start_date, end_date))
    clm.attrs["tz"] = "UTC"
    clm = clm.transpose("time", "y", "x")
    for v in clm:
        clm[v].attrs = NLDAS_VARS[str(v)]
    clm = clm.rio.write_transform()
    clm = clm.rio.write_crs(4326)
    clm = clm.rio.write_coordinate_system()
    if isinstance(geometry, (list, tuple)):
        return clm
    return hgu.xarray_geomask(clm, geometry, geo_crs, all_touched=True)
