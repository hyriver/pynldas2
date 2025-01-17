"""Some utilities for PyDaymet."""

from __future__ import annotations

import atexit
import hashlib
import re
import os
from collections.abc import Generator, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyproj
import shapely
import urllib3
from pyproj import Transformer
from pyproj.exceptions import CRSError as ProjCRSError
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import rowcol
import xarray as xr
from rasterio.windows import Window
from rioxarray.exceptions import OneDimensionalRaster
from shapely import MultiPolygon, Polygon, STRtree, ops
from shapely.geometry import shape
from urllib3.exceptions import HTTPError

from pynldas2.exceptions import DownloadError, InputRangeError, InputTypeError

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry

    CRSTYPE = int | str | pyproj.CRS
    POLYTYPE = Polygon | MultiPolygon | tuple[float, float, float, float]
    NUMBER = int | float | np.number

__all__ = [
    "clip_dataset",
    "get_grid_mask",
    "download_files",
    "to_geometry",
    "transform_coords",
    "validate_coords",
    "validate_crs",
    "extract_info",
]

TransformerFromCRS = lru_cache(Transformer.from_crs)
HTTPSPool = urllib3.HTTPSConnectionPool(
    "hydro1.gesdisc.eosdis.nasa.gov",
    maxsize=10,
    block=True,
    retries=urllib3.Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 504],
        allowed_methods=["HEAD", "GET"],
    ),
)


def _cleanup_https_pool():
    """Cleanup the HTTPS connection pool."""
    HTTPSPool.close()


atexit.register(_cleanup_https_pool)


def validate_crs(crs: CRSTYPE) -> str:
    """Validate a CRS.

    Parameters
    ----------
    crs : str, int, or pyproj.CRS
        Input CRS.

    Returns
    -------
    str
        Validated CRS as a string.
    """
    try:
        return pyproj.CRS(crs).to_string()
    except ProjCRSError as ex:
        raise InputTypeError("crs", "a valid CRS") from ex


def transform_coords(
    coords: Sequence[tuple[float, float]], in_crs: CRSTYPE, out_crs: CRSTYPE
) -> list[tuple[float, float]]:
    """Transform coordinates from one CRS to another."""
    try:
        pts = shapely.points(np.atleast_2d(coords))
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    x, y = shapely.get_coordinates(pts).T
    x_proj, y_proj = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform(x, y)
    return list(zip(x_proj, y_proj))


def _geo_transform(geom: BaseGeometry, in_crs: CRSTYPE, out_crs: CRSTYPE) -> BaseGeometry:
    """Transform a geometry from one CRS to another."""
    project = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform
    return ops.transform(project, geom)


def validate_coords(
    coords: Iterable[tuple[float, float]], bounds: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    """Validate coordinates within a bounding box."""
    try:
        pts = shapely.points(list(coords))
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    if shapely.contains(shapely.box(*bounds), pts).all():
        return shapely.get_coordinates(pts).round(6)
    raise InputRangeError("coords", f"within {bounds}")


def to_geometry(
    geometry: BaseGeometry | tuple[float, float, float, float],
    geo_crs: CRSTYPE | None = None,
    crs: CRSTYPE | None = None,
) -> BaseGeometry:
    """Return a Shapely geometry and optionally transformed to a new CRS.

    Parameters
    ----------
    geometry : shaple.Geometry or tuple of length 4
        Any shapely geometry object or a bounding box (minx, miny, maxx, maxy).
    geo_crs : int, str, or pyproj.CRS, optional
        Spatial reference of the input geometry, defaults to ``None``.
    crs : int, str, or pyproj.CRS
        Target spatial reference, defaults to ``None``.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        A shapely geometry object.
    """
    is_geom = np.atleast_1d(shapely.is_geometry(geometry))
    if is_geom.all() and len(is_geom) == 1:
        geom = geometry
    elif isinstance(geometry, Iterable) and len(geometry) == 4 and np.isfinite(geometry).all():
        geom = shapely.box(*geometry)
    else:
        raise InputTypeError("geometry", "a shapley geometry or tuple of length 4")

    if geo_crs is not None and crs is not None:
        return _geo_transform(geom, geo_crs, crs)
    elif geo_crs is None and crs is not None:
        return geom
    raise InputTypeError("geo_crs/crs", "either both None or both valid CRS")


def _download(url: str, fname: Path) -> None:
    """Download a file from a URL."""
    parsed_url = urlparse(url)
    path = f"{parsed_url.path}?{parsed_url.query}"
    try:
        head = HTTPSPool.request("HEAD", path)
    except urllib3.exceptions.HTTPError as e:
        raise DownloadError(url, f"Failed HEAD request: {e}") from e
    fsize = int(head.headers.get("Content-Length", -1))
    if fname.exists() and fname.stat().st_size == fsize:
        return
    fname.unlink(missing_ok=True)
    fname.write_text(HTTPSPool.request("GET", path).data.decode())


def _get_loc(query: str) -> tuple[float, float]:
    """Extract longitude and latitude."""
    match = re.match(r"GEOM:POINT\(([-\d.]+), ([-\d.]+)\)", query)
    if match:
        lon, lat = map(float, match.groups())
        return lon, lat
    raise ValueError("Input string is not in the expected format")


def _get_prefix(url: str) -> str:
    """Get the file prefix for creating a unique filename from a URL."""
    query = urlparse(url).query
    var = parse_qs(query).get("variable", ["var"])[0].split(":")[-1]
    loc = parse_qs(query).get("location", ["GEOM:POINT"])[0]
    lon, lat = _get_loc(loc)
    return f"{lon}_{lat}_{var}"


def download_files(url_list: list[str], rewrite: bool = False) -> list[Path]:
    """Download multiple files concurrently."""
    hr_cache = os.getenv("HYRIVER_CACHE_NAME")
    cache_dir = Path(hr_cache).parent if hr_cache else Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)

    file_list = [
        Path(cache_dir, f"{_get_prefix(url)}_{hashlib.sha256(url.encode()).hexdigest()}.txt")
        for url in url_list
    ]

    if rewrite:
        _ = [f.unlink(missing_ok=True) for f in file_list]
    max_workers = min(4, os.cpu_count() or 1, len(url_list))
    if max_workers == 1:
        _ = [_download(url, path) for url, path in zip(url_list, file_list)]
        return file_list

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(_download, url, path): url for url, path in zip(url_list, file_list)
        }
        for future in as_completed(future_to_url):
            try:
                future.result()
            except Exception as e:  # noqa: PERF203
                raise DownloadError(future_to_url[future], e) from e
    return file_list


def clip_dataset(
    ds: xr.Dataset,
    geometry: Polygon | MultiPolygon | tuple[float, float, float, float],
    crs: CRSTYPE,
) -> xr.Dataset:
    """Mask a ``xarray.Dataset`` based on a geometry."""
    attrs = {v: ds[v].attrs for v in ds}

    geom = to_geometry(geometry, crs, ds.rio.crs)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            ds = ds.rio.clip([geom], all_touched=True)
    except OneDimensionalRaster:
        ds = ds.rio.clip([geom], all_touched=True)

    _ = [ds[v].rio.update_attrs(attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds


def get_grid_mask(save_dir: str | Path | None = None) -> xr.Dataset:
    """Get the NLDAS-2 grid that contains the land/water/soil/vegetation mask.

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
    if save_dir is None:
        hr_cache = os.getenv("HYRIVER_CACHE_NAME")
        save_dir = Path(hr_cache).parent if hr_cache else Path("cache")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    nc_path = save_dir / Path(url).name
    if nc_path.exists():
        return xr.open_dataset(nc_path, decode_coords="all")

    try:
        resp = urllib3.request(
            "GET",
            url,
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 504],
                allowed_methods=["GET"],
            ),
        )
    except HTTPError as e:
        raise DownloadError(url, e) from e
    grid = xr.open_dataset(BytesIO(resp.data)).rio.write_crs(4326)
    grid.to_netcdf(nc_path)
    return grid


def extract_info(file_path: Path) -> tuple[float, float, str]:
    """Extract variable names from file stems."""
    pattern = re.compile(r"[-\d.]+_[\d.]+_([A-Za-z0-9_]+)_")
    stem = file_path.stem
    match = pattern.search(stem)
    lon, lat = map(float, stem.split("_")[:2])
    if match:
        return lon, lat, match.group(1)
    raise ValueError(f"Invalid file format: {file_path}")