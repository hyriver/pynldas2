=======
History
=======

0.17.1 (2024-09-14)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.8 since its end-of-life date is October 2024.

0.17.0 (2024-05-07)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add the ``exceptions`` module to the high-level API to declutter
  the main module. In the future, all exceptions will be raised from
  this module and not from the main module. For now, the exceptions
  are raised from both modules for backward compatibility.
- Switch to using the ``src`` layout instead of the ``flat`` layout
  for the package structure. This is to make the package more
  maintainable and to avoid any potential conflicts with other
  packages.
- Add artifact attestations to the release workflow.

0.16.0 (2024-01-03)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Drop support for ``shapely<2``.

0.15.2 (2023-09-22)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Remove dependency on ``dask``.
- Reduce complexity of the code by breaking down the ``_check_inputs``
  function into ``_get_variables`` and ``_get_dates`` functions.

0.15.1 (2023-07-10)
-------------------

Bug Fixes
~~~~~~~~~
- Fix a bug in computing snow where the ``t_snow`` argument was not
  being converted to Kelvin.

New Features
~~~~~~~~~~~~
- If ``snow=True`` is passed to both ``get_bygeom`` and ``get_bycoords``
  functions, the ``variables`` argument will be checked to see if it
  contains ``prcp`` and ``temp``, if not, they will be added to the
  list of variables to be retrieved. This is to ensure that the
  ``snow`` argument works as expected.

0.15.0 (2023-05-07)
-------------------
From release 0.15 onward, all minor versions of HyRiver packages
will be pinned. This ensures that previous minor versions of HyRiver
packages cannot be installed with later minor releases. For example,
if you have ``py3dep==0.14.x`` installed, you cannot install
``pydaymet==0.15.x``. This is to ensure that the API is
consistent across all minor versions.

New Features
~~~~~~~~~~~~
- Add ``source`` argument to both ``get_bygeom`` and ``get_bycoords`` functions.
  Valid values for source are ``grib`` (default) and ``netcdf``.
  Both return the same values, the latter also offers additional variable ``psurf``
  for surface pressure.
  Valid variable names for ``netcdf`` are:
  ``prcp``, ``pet``, ``wind_u``, ``wind_v``, ``humidity``,
  ``temp``, ``rsds``, ``rlds``, ``psurf``
  Valid variable names for ``grib`` source are unchanged as to not
  introduce breaking changes. By `Luc Rébillout <https://github.com/LucRSquared>`__.
- For now, retain compatibility with ``shapely<2`` while supporting
  ``shapley>=2``.

0.14.0 (2023-03-05)
-------------------

New Features
~~~~~~~~~~~~
- Add ``snow`` and ``snow_params`` arguments to both ``get_bygeom``
  and ``get_bycoords`` functions for computing snow from ``prcp``
  and ``temp``.
- Rewrite ``by_coords`` functions to improve performance and
  reduce memory usage. Also, its ``to_xarray`` argument now returns
  a much better structured ``xarray.Dataset``. Moreover, the function
  has a new argument called ``coords_id`` which allows the user to
  specify IDs for the input coordinates. This is useful for cases
  where the coordinates belong to some specific features, such as
  station location, that have their own IDs. These IDs will be used
  for both cases where the data is returned as ``pandas.DataFrame``
  or ``xarray.Dataset``.

Internal Changes
~~~~~~~~~~~~~~~~
- Sync all minor versions of HyRiver packages to 0.14.0.

0.1.12 (2023-02-10)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Fully migrate ``setup.cfg`` and ``setup.py`` to ``pyproject.toml``.
- Convert relative imports to absolute with ``absolufy-imports``.
- Sync all patch versions of HyRiver packages to x.x.12.

0.1.2 (2023-01-08)
------------------

New Features
~~~~~~~~~~~~
- Refactor the ``show_versions`` function to improve performance and
  print the output in a nicer table-like format.

0.1.1 (2022-12-16)
------------------

Bug Fixes
~~~~~~~~~
- Fix an issue where for single variable, i.e., not a list, could not
  be detected correctly.
- Fix an issue in converting the response from the service to a dataframe
  or dataset when service fails and throws an error.

0.1.0 (2022-12-15)
------------------

- Initial release.
