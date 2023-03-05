=======
History
=======

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
