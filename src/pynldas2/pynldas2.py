"""Get hourly NLDAS2 forcing data."""

from __future__ import annotations

import functools
import hashlib
import itertools
import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from urllib.parse import urlencode

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pandas.errors import EmptyDataError

import pynldas2._utils as utils
import tiny_retriever as terry
from pynldas2.exceptions import InputRangeError, InputTypeError, InputValueError, NLDASServiceError

try:
    from numpy._core._exceptions import UFuncTypeError
except ImportError:
    UFuncTypeError = TypeError

try:
    from numba import config as numba_config
    from numba import njit, prange

    ngjit = functools.partial(njit, nogil=True)
    numba_config.THREADING_LAYER = "workqueue"
    has_numba = True
except ImportError:
    has_numba = False
    prange = range

    T = TypeVar("T")
    Func = Callable[..., T]

    def ngjit(_: str | Func[T]) -> Callable[[Func[T]], Func[T]]:
        def decorator_njit(func: Func[T]) -> Func[T]:
            @functools.wraps(func)
            def wrapper_decorator(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> T:
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pyproj import CRS
    from shapely import Polygon

    CRSType = int | str | CRS
    Dataset = TypeVar("Dataset", pd.DataFrame, xr.Dataset)

# Default snow params from https://doi.org/10.5194/gmd-11-1077-2018
T_RAIN = 2.5  # degC
T_SNOW = 0.6  # degC
URL = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"

NLDAS2_VARS = {
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

__all__ = ["get_bycoords", "get_bygeom"]


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


def separate_snow(clm: Dataset, t_rain: float = T_RAIN, t_snow: float = T_SNOW) -> Dataset:
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
        return _snow_gridded(clm, t_rain + 273.15, t_snow + 273.15)
    return _snow_point(clm, t_rain + 273.15, t_snow + 273.15)


def _txt2df(name: str, txt_file: Path) -> pd.Series:
    """Convert text to dataframe."""
    try:
        data = pd.read_csv(txt_file, skiprows=12, sep=r"\s+").dropna()
        data.index = pd.to_datetime(data[DATE_COL], utc=True)
    except EmptyDataError:
        return pd.Series(name=name)
    except UFuncTypeError as ex:
        msg = "".join(re.findall("<strong>(.*?)</strong>", txt_file.read_text(), re.DOTALL)).strip()
        raise NLDASServiceError(msg) from ex

    data = data["Data"]
    data.name = name
    data.index.name = "time"
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def _get_variables(
    variables: str | list[str] | None, snow: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Get variables."""
    source_tag = "NLDAS2:NLDAS_FORA0125_H_v2.0"
    if variables is None:
        url_vars = tuple(f"{source_tag}:{d['nldas_name']}" for d in NLDAS2_VARS.values())
        clm_vars = tuple(NLDAS2_VARS)
    else:
        url_vars = [variables] if isinstance(variables, str) else list(variables)
        if snow:
            url_vars = list(set(url_vars).union({"temp", "prcp"}))
        if any(v not in NLDAS2_VARS for v in url_vars):
            raise InputValueError("variables", list(NLDAS2_VARS))
        url_vars, clm_vars = zip(
            *((f"{source_tag}:{NLDAS2_VARS[v]['nldas_name']}", v) for v in url_vars)
        )
    return url_vars, clm_vars


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
    dates = [*dates, end] if dates[-1] < end else dates
    return dates


def _download_files(
    lons: Iterable[float],
    lats: Iterable[float],
    variables: str | list[str] | None,
    snow: bool,
    start_date: str,
    end_date: str,
    validate_filesize: bool,
    timeout: int,
) -> dict[tuple[float, float], dict[str, list[Path]]]:
    """Download NLDAS data and return a dictionary of files grouped by location and variable."""
    dates = _get_dates(start_date, end_date)
    url_vars, clm_vars = _get_variables(variables, snow)
    meta, urls = zip(
        *(
            (
                (float(lon), float(lat), v),
                f"{URL}?"
                + urlencode(
                    {
                        "variable": url_v,
                        "location": f"GEOM:POINT({lon}, {lat})",
                        "startDate": s.strftime(DATE_FMT),
                        "endDate": e.strftime(DATE_FMT),
                        "type": "asc2",
                    }
                ),
            )
            for lon, lat in zip(lons, lats)
            for (url_v, v), (s, e) in itertools.product(
                zip(url_vars, clm_vars), zip(dates[:-1], dates[1:])
            )
        )
    )

    hr_cache = os.getenv("HYRIVER_CACHE_NAME")
    cache_dir = Path(hr_cache).parent if hr_cache else Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    file_paths = [
        cache_dir / f"{x}_{y}_{v}_{hashlib.sha256(url.encode()).hexdigest()}.txt"
        for (x, y, v), url in zip(meta, urls)
    ]
    if not validate_filesize and all(f.exists() and f.stat().st_size > 0 for f in file_paths):
        pass
    else:
        terry.download(urls, file_paths, timeout=timeout)
    # group based on lon, lat, and variable, i.e, dict of dict of list
    grouped_files = {}
    for (x, y, v), f in zip(meta, file_paths):
        if (x, y) not in grouped_files:
            grouped_files[(x, y)] = {}
        if v not in grouped_files[(x, y)]:
            grouped_files[(x, y)][v] = []

        grouped_files[(x, y)][v].append(f)
    return grouped_files


def _by_coord(
    txt_files: dict[str, list[Path]],
    start_date: str,
    end_date: str,
    snow: bool,
    snow_params: dict[str, float] | None,
) -> pd.DataFrame:
    """Get NLDAS climate forcing data for a single location."""
    clm = (
        pd.concat((pd.concat(_txt2df(v, f) for f in resp) for v, resp in txt_files.items()), axis=1)
        .loc[start_date:end_date]
        .rename(columns={d["nldas_name"]: n for n, d in NLDAS2_VARS.items()})
    )
    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)
    return clm


def _get_lon_lat(
    coords: list[tuple[float, float]] | tuple[float, float],
    bounds: tuple[float, float, float, float],
    coords_id: Sequence[str | int] | None,
    crs: CRSType,
    to_xarray: bool,
) -> tuple[list[float], list[float]]:
    """Get longitude and latitude from a list of coordinates."""
    coords_list = utils.transform_coords(coords, crs, 4326)

    if to_xarray and coords_id is not None and len(coords_id) != len(coords_list):
        raise InputTypeError("coords_id", "list with the same length as of coords")

    lon, lat = utils.validate_coords(coords_list, bounds).T
    return lon.tolist(), lat.tolist()


def get_bycoords(
    coords: list[tuple[float, float]],
    start_date: str,
    end_date: str,
    coords_id: Sequence[str | int] | None = None,
    crs: CRSType = 4326,
    variables: str | list[str] | None = None,
    to_xarray: bool = False,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> pd.DataFrame | xr.Dataset:
    """Get NLDAS-2 climate forcing data for a list of coordinates.

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
        ``rlds``, ``rsds``, and ``humidity`` and ``psurf``.
    to_xarray : bool, optional
        If True, the data is returned as an xarray dataset.
    snow : bool, optional
        Compute snowfall from precipitation and temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

    Returns
    -------
    pandas.DataFrame
        The requested data as a dataframe.
    """
    bounds = (-125.0, 25.0, -67.0, 53.0)
    lons, lats = _get_lon_lat(coords, bounds, coords_id, crs, to_xarray)
    n_pts = len(lons)
    grouped_files = _download_files(
        lons, lats, variables, snow, start_date, end_date, validate_filesize, conn_timeout
    )

    idx = list(coords_id) if coords_id is not None else list(range(n_pts))
    idx = dict(zip(zip(lons, lats), idx))
    clm_dict = {
        idx[c]: _by_coord(file, start_date, end_date, snow, snow_params)
        for c, file in grouped_files.items()
    }
    if to_xarray:
        clm_ds = xr.concat(
            (xr.Dataset.from_dataframe(clm) for clm in clm_dict.values()),
            dim=pd.Index(list(clm_dict), name="id"),
        )
        clm_ds.attrs["tz"] = "UTC"
        return clm_ds

    if n_pts == 1:
        clm = next(iter(clm_dict.values()), pd.DataFrame())
    else:
        clm = pd.concat(clm_dict.values(), keys=list(clm_dict), axis=1)
        clm.columns = clm.columns.set_names(["id", "variable"])
    clm.index = pd.DatetimeIndex(clm.index, tz="UTC")
    clm.index.name = "time"
    return clm


def _txt2da(
    lon: float,
    lat: float,
    name: str,
    txt_file: Path,
) -> xr.DataArray:
    """Convert text to dataarray."""
    data = _txt2df(name, txt_file)
    da = data.to_xarray()
    return da.assign_coords(x=lon, y=lat).expand_dims("y").expand_dims("x")


def get_bygeom(
    geometry: Polygon | tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    geo_crs: CRSType = 4326,
    variables: str | list[str] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> xr.Dataset:
    """Get hourly NLDAS-2 climate forcing within a geometry at 0.125 resolution.

    Parameters
    ----------
    geometry : Polygon or tuple
        The geometry of the region of interest. It can be a shapely Polygon or a tuple
        of length 4 representing the bounding box (minx, miny, maxx, maxy).
    start_date : str
        Start date of the data.
    end_date : str
        End date of the data.
    geo_crs : int, str, or pyproj.CRS
        CRS of the input geometry
    variables : str or list of str, optional
        Variables to download. If None, all variables are downloaded.
        Valid variables are: ``prcp``, ``pet``, ``temp``, ``wind_u``, ``wind_v``,
        ``rlds``, ``rsds``, and ``humidity`` and ``psurf``.
    snow : bool, optional
        Compute snowfall from precipitation and temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

    Returns
    -------
    xarray.Dataset
        The requested forcing data.
    """
    nldas_grid = utils.get_grid_mask()
    geom = utils.to_geometry(geometry, geo_crs, nldas_grid.rio.crs)
    msk = nldas_grid.CONUS_mask.rio.clip([geom], all_touched=True)
    lons, lats = zip(*itertools.product(msk.get_index("lon"), msk.get_index("lat")))
    grouped_files = _download_files(
        lons, lats, variables, snow, start_date, end_date, validate_filesize, conn_timeout
    )

    clm = xr.merge(
        _txt2da(x, y, f, r)
        for (x, y), v_files in grouped_files.items()
        for f, resp in v_files.items()
        for r in resp
    )
    clm = (
        clm.rename(
            {d["nldas_name"]: n for n, d in NLDAS2_VARS.items() if d["nldas_name"] in clm.data_vars}
        )
        .sel(time=slice(start_date, end_date))
        .transpose("time", "y", "x")
    )
    clm.attrs["tz"] = "UTC"
    for v in clm.data_vars:
        clm[v].attrs = NLDAS2_VARS[str(v)]
    clm = clm.rio.write_crs(nldas_grid.rio.crs)
    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)
    return clm
