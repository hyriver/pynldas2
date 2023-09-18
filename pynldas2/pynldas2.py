"""Get hourly NLDAS2 forcing data."""
from __future__ import annotations

import functools
import itertools
import re
import warnings
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Sequence, TypeVar, Union

import async_retriever as ar
import numpy as np
import numpy.typing as npt
import pandas as pd
import pygeoutils as hgu
import pyproj
import xarray as xr
from numpy.core._exceptions import UFuncTypeError
from pandas.errors import EmptyDataError

from pynldas2.exceptions import InputRangeError, InputTypeError, InputValueError, NLDASServiceError

try:
    from numba import config as numba_config
    from numba import njit, prange

    ngjit = functools.partial(njit, cache=True, nogil=True)
    numba_config.THREADING_LAYER = "workqueue"  # pyright: ignore[reportGeneralTypeIssues]
    has_numba = True
except ImportError:
    has_numba = False
    prange = range
    numba_config = None
    njit = None

    def ngjit(ntypes, parallel=None):  # type: ignore
        def decorator_njit(func):  # type: ignore
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

    DF = TypeVar("DF", pd.DataFrame, xr.Dataset)

# Default snow params from https://doi.org/10.5194/gmd-11-1077-2018
T_RAIN = 2.5  # degC
T_SNOW = 0.6  # degC
CRSTYPE = Union[int, str, pyproj.CRS]
URL = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"

NLDAS_VARS_GRIB = {
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

NLDAS_VARS_NETCDF = {
    "prcp": {"nldas_name": "Rainf", "long_name": "Total precipitation", "units": "kg/m^2"},
    "rlds": {
        "nldas_name": "LWdown",
        "long_name": "Surface downward longwave radiation",
        "units": "W/m^2",
    },
    "rsds": {
        "nldas_name": "SWdown",
        "long_name": "Surface downward shortwave radiation",
        "units": "W/m^2",
    },
    "pet": {"nldas_name": "PotEvap", "long_name": "Potential evaporation", "units": "kg/m^2"},
    "psurf": {"nldas_name": "PSurf", "long_name": "Surface pressure", "units": "Pa"},
    "humidity": {
        "nldas_name": "Qair",
        "long_name": "2-m above ground specific humidity",
        "units": "kg/kg",
    },
    "temp": {
        "nldas_name": "Tair",
        "long_name": "2-m above ground temperature",
        "units": "K",
    },
    "wind_u": {
        "nldas_name": "Wind_E",
        "long_name": "U wind component at 10-m above the surface",
        "units": "m/s",
    },
    "wind_v": {
        "nldas_name": "Wind_N",
        "long_name": "V wind component at 10-m above the surface",
        "units": "m/s",
    },
}

DATE_COL = "Date&Time"
DATE_FMT = "%Y-%m-%dT%H"
__all__ = ["get_bycoords", "get_grid_mask", "get_bygeom"]


@ngjit("f8[::1](f8[::1], f8[::1], f8, f8)")
def _separate_snow(
    prcp: npt.NDArray[np.float64],
    temp: npt.NDArray[np.float64],
    t_rain: np.float64,
    t_snow: np.float64,
) -> npt.NDArray[np.float64]:
    """Separate snow in precipitation."""
    t_rng = t_rain - t_snow
    snow = np.zeros_like(prcp)

    for t in prange(prcp.shape[0]):
        if temp[t] > t_rain:
            snow[t] = 0.0
        elif temp[t] < t_snow:
            snow[t] = prcp[t]
        else:
            snow[t] = prcp[t] * (t_rain - temp[t]) / t_rng
    return snow


def _snow_point(climate: pd.DataFrame, t_rain: float, t_snow: float) -> pd.DataFrame:
    """Separate snow from precipitation."""
    clm = climate.copy()
    clm["snow"] = _separate_snow(
        clm["prcp"].to_numpy("f8"),
        clm["temp"].to_numpy("f8"),
        np.float64(t_rain),
        np.float64(t_snow),
    )
    return clm


def _snow_gridded(climate: xr.Dataset, t_rain: float, t_snow: float) -> xr.Dataset:
    """Separate snow from precipitation."""
    clm = climate.copy()

    def snow_func(
        prcp: npt.NDArray[np.float64],
        temp: npt.NDArray[np.float64],
        t_rain: float,
        t_snow: float,
    ) -> npt.NDArray[np.float64]:
        """Separate snow based on Martinez and Gupta (2010)."""
        return _separate_snow(
            prcp.astype("f8"),
            temp.astype("f8"),
            np.float64(t_rain),
            np.float64(t_snow),
        )

    clm["snow"] = xr.apply_ufunc(
        snow_func,
        clm["prcp"],
        clm["temp"],
        t_rain,
        t_snow,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[clm["prcp"].dtype],
    ).transpose("time", "y", "x")
    clm["snow"].attrs["units"] = "mm"
    clm["snow"].attrs["long_name"] = "Snowfall hourly total"
    return clm


def separate_snow(clm: DF, t_rain: float = T_RAIN, t_snow: float = T_SNOW) -> DF:
    """Separate snow based on :footcite:t:`Martinez_2010`.

    Parameters
    ----------
    clm : pandas.DataFrame or xarray.Dataset
        Climate data that should include ``prcp`` and ``temp``.
    t_rain : float, optional
        Threshold for temperature in deg C for considering rain, defaults to 2.5 degrees C.
    t_snow : float, optional
        Threshold for temperature in deg C for considering snow, defaults to 0.6 degrees C.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Input data with ``snow`` column if input is a ``pandas.DataFrame``,
        or ``snow`` variable if input is an ``xarray.Dataset``.

    References
    ----------
    .. footbibliography::
    """
    if not has_numba:
        warnings.warn(
            "Numba not installed. Using slow pure python version.", UserWarning, stacklevel=2
        )

    if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
        raise InputTypeError("clm", "pandas.DataFrame or xarray.Dataset")

    if isinstance(clm, xr.Dataset):
        return _snow_gridded(clm, t_rain + 273.15, t_snow + 273.15)  # type: ignore
    return _snow_point(clm, t_rain + 273.15, t_snow + 273.15)


def _txt2df(
    txt: str,
    resp_id: int,
    kwds: list[dict[str, dict[str, str]]],
    source: str = "grib",
) -> pd.Series:
    """Convert text to dataframe."""
    try:
        if source == "grib":
            data = pd.read_csv(StringIO(txt), skiprows=39, delim_whitespace=True).dropna()
            data.index = pd.to_datetime(data.index + " " + data[DATE_COL], utc=True)
        else:
            data = pd.read_csv(StringIO(txt), skiprows=12, delim_whitespace=True).dropna()
            data.index = pd.to_datetime(data[DATE_COL], utc=True)
    except EmptyDataError:
        return pd.Series(name=kwds[resp_id]["params"]["variable"].split(":")[-1])
    except UFuncTypeError as ex:
        msg = "".join(re.findall("<strong>(.*?)</strong>", txt, re.DOTALL)).strip()
        raise NLDASServiceError(msg) from ex

    data = data.drop(columns=DATE_COL)["Data"]
    data.name = kwds[resp_id]["params"]["variable"].split(":")[-1]
    return data


def _get_variables(
    variables: str | list[str] | None = None,
    snow: bool = False,
    source: str = "grib",
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Get variables."""
    if source == "grib":
        source_tag = "NLDAS:NLDAS_FORA0125_H.002"
        nldas_vars = NLDAS_VARS_GRIB
    elif source == "netcdf":
        source_tag = "NLDAS2:NLDAS_FORA0125_H_v2.0"
        nldas_vars = NLDAS_VARS_NETCDF
    else:
        raise InputValueError("source", ["grib", "netcdf"])

    if variables is None:
        clm_vars = [f"{source_tag}:{d['nldas_name']}" for d in nldas_vars.values()]
    else:
        clm_vars = [variables] if isinstance(variables, str) else list(variables)
        if snow:
            required_vars = ["temp", "prcp"]
            if not all(v in clm_vars for v in required_vars):
                clm_vars = list(set(clm_vars).union(required_vars))
        if any(v not in nldas_vars for v in clm_vars):
            raise InputValueError("variables", list(nldas_vars))
        clm_vars = [f"{source_tag}:{nldas_vars[v]['nldas_name']}" for v in clm_vars]
    return clm_vars, nldas_vars


def _get_dates(
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    """Get dates."""
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
    return dates


def _byloc(
    lon: float,
    lat: float,
    start_date: str,
    end_date: str,
    variables: str | list[str] | None = None,
    n_conn: int = 4,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    source: str = "grib",
) -> pd.DataFrame:
    """Get NLDAS climate forcing data for a single location."""
    dates = _get_dates(start_date, end_date)
    clm_vars, nldas_vars = _get_variables(variables, snow, source)
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

    clm_list = (_txt2df(txt, i, kwds, source=source) for i, txt in enumerate(resp))

    clm_merged = (
        pd.concat(df)
        for _, df in itertools.groupby(
            sorted(clm_list, key=lambda x: str(x.name)), lambda x: str(x.name)
        )
    )
    clm = pd.concat(clm_merged, axis=1)
    clm = clm.rename(columns={d["nldas_name"]: n for n, d in nldas_vars.items()})

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)
    clm.index.name = "time"
    clm.index = pd.to_datetime(clm.index).tz_localize(None)
    return clm.loc[start_date:end_date]


def _get_lon_lat(
    coords: list[tuple[float, float]] | tuple[float, float],
    crs: CRSTYPE = 4326,
) -> tuple[list[float], list[float]]:
    """Get longitude and latitude from a list of coordinates."""
    try:
        coords_list = hgu.coords_list(coords)
    except hgu.InputTypeError as ex:
        raise InputTypeError("coords", "tuple of length 2 or list of tuples") from ex

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
    coords_id: Sequence[str | int] | None = None,
    crs: CRSTYPE = 4326,
    variables: str | list[str] | None = None,
    to_xarray: bool = False,
    n_conn: int = 4,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    source: str = "grib",
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
        ``rlds``, ``rsds``, and ``humidity`` (and ``psurf`` if ``source=netcdf``)
    to_xarray : bool, optional
        If True, the data is returned as an xarray dataset.
    n_conn : int, optional
        Number of parallel connections to use for retrieving data, defaults to 4.
        The maximum number of connections is 4, if more than 4 are requested, 4
        connections will be used.
    snow : bool, optional
        Compute snowfall from precipitation and temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    source: str, optional
        Source to pull data rods from. Valid sources are: ``grib`` and ``netcdf``

    Returns
    -------
    pandas.DataFrame
        The requested data as a dataframe.
    """
    lons, lats = _get_lon_lat(coords, crs)

    bounds = (-125.0, 25.0, -67.0, 53.0)
    points = hgu.Coordinates(lons, lats, bounds).points
    n_pts = len(points)
    if n_pts == 0 or n_pts != len(lons):
        raise InputRangeError("coords", f"{bounds}")

    idx = list(coords_id) if coords_id is not None else [f"P{i}" for i in range(n_pts)]
    nldas = functools.partial(
        _byloc,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        n_conn=n_conn,
        snow=snow,
        snow_params=snow_params,
        source=source,
    )

    _, nldas_vars = _get_variables(variables, snow, source)

    clm_list = itertools.starmap(nldas, zip(points.x, points.y))
    if to_xarray:
        clm_ds = xr.concat(
            (xr.Dataset.from_dataframe(clm) for clm in clm_list), dim=pd.Index(idx, name="id")
        )
        clm_ds.attrs["tz"] = "UTC"
        for v in clm_ds.data_vars:
            clm_ds[v].attrs = nldas_vars[str(v)]
        return clm_ds

    if n_pts == 1:
        clm = next(iter(clm_list), pd.DataFrame())
    else:
        clm = pd.concat(clm_list, keys=idx, axis=1)
        clm.columns = clm.columns.set_names(["id", "variable"])
    clm.index = pd.DatetimeIndex(clm.index, tz="UTC")
    clm.index.name = "time"
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
    grid = hgu.xd_write_crs(grid, 4326)
    return grid


def _txt2da(
    txt: str,
    resp_id: int,
    kwds: list[dict[str, dict[str, str]]],
    source: str = "grib",
) -> xr.DataArray:
    """Convert text to dataarray."""
    try:
        if source == "grib":
            data = pd.read_csv(StringIO(txt), skiprows=39, delim_whitespace=True).dropna()
            data.index = pd.to_datetime(data.index + " " + data[DATE_COL], utc=True)
        else:
            data = pd.read_csv(StringIO(txt), skiprows=12, delim_whitespace=True).dropna()
            data.index = pd.to_datetime(data[DATE_COL], utc=True)
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
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    source: str = "grib",
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
        ``rlds``, ``rsds``, and ``humidity`` (and ``psurf`` if ``source=netcdf``)
    n_conn : int, optional
        Number of parallel connections to use for retrieving data, defaults to 4.
        It should be less than 4.
    snow : bool, optional
        Compute snowfall from precipitation and temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    source: str, optional
        Source to pull data rods from. Valid sources are: ``grib`` and ``netcdf``.

    Returns
    -------
    xarray.Dataset
        The requested forcing data.
    """
    dates = _get_dates(start_date, end_date)
    clm_vars, nldas_vars = _get_variables(variables, snow, source)

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

    clm = xr.merge(_txt2da(txt, i, kwds, source=source) for i, txt in enumerate(resp))
    clm = clm.rename({d["nldas_name"]: n for n, d in nldas_vars.items() if d["nldas_name"] in clm})
    clm = clm.sel(time=slice(start_date, end_date))
    clm.attrs["tz"] = "UTC"
    clm = clm.transpose("time", "y", "x")
    for v in clm:
        clm[v].attrs = nldas_vars[str(v)]
    clm = hgu.xd_write_crs(clm, 4326)
    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)
    if isinstance(geometry, (list, tuple)):
        return clm
    clm = hgu.xarray_geomask(clm, geometry, geo_crs, all_touched=True)
    return clm
