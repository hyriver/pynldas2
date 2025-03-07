"""Some utilities for PyDaymet."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import shapely
import xarray as xr
from pyproj import Transformer
from pyproj.exceptions import CRSError as ProjCRSError
from rioxarray.exceptions import OneDimensionalRaster
from shapely import Polygon, ops

import tiny_retriever as terry
from pynldas2.exceptions import InputRangeError, InputTypeError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    CRSType = int | str | pyproj.CRS
    PolyType = Polygon | tuple[float, float, float, float]
    Number = int | float | np.number[Any]

__all__ = [
    "clip_dataset",
    "get_grid_mask",
    "to_geometry",
    "transform_coords",
    "validate_coords",
    "validate_crs",
]

TransformerFromCRS = lru_cache(Transformer.from_crs)


def validate_crs(crs: CRSType) -> str:
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
    coords: Sequence[tuple[float, float]], in_crs: CRSType, out_crs: CRSType
) -> list[tuple[float, float]]:
    """Transform coordinates from one CRS to another."""
    try:
        pts = shapely.points(np.atleast_2d(coords))
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    x, y = shapely.get_coordinates(pts).T
    x_proj, y_proj = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform(x, y)
    return list(zip(x_proj, y_proj))


def _geo_transform(geom: Polygon, in_crs: CRSType, out_crs: CRSType) -> Polygon:
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
    geometry: Polygon | tuple[float, float, float, float],
    geo_crs: CRSType | None = None,
    crs: CRSType | None = None,
) -> Polygon:
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
    shapely.Polygon
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
        return _geo_transform(geom, geo_crs, crs)  # pyright: ignore[reportArgumentType]
    elif geo_crs is None and crs is not None:
        return geom  # pyright: ignore[reportArgumentType]
    raise InputTypeError("geo_crs/crs", "either both None or both valid CRS")


def clip_dataset(
    ds: xr.Dataset,
    geometry: Polygon | tuple[float, float, float, float],
    crs: CRSType,
) -> xr.Dataset:
    """Mask a ``xarray.Dataset`` based on a geometry."""
    attrs = {v: ds[v].attrs for v in ds}

    geom = to_geometry(geometry, crs, ds.rio.crs)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, Polygon):
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
    terry.download(url, nc_path)
    with xr.open_dataset(nc_path) as ds:
        grid = ds.rio.write_crs(4326).load()
    nc_path.unlink()
    grid.to_netcdf(nc_path)
    return grid
