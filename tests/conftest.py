import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer

# A small projected grid (UTM zone 32N) with CF-complete metadata, used by the
# coordinate attribute and georeferencing tests.
TEST_CRS = CRS.from_epsg(32632)


def make_projected_dataset(
    x_units: str = "m",
    n_x: int = 5,
    n_y: int = 4,
    crs_attr_overrides: dict | None = None,
    drop_crs_attrs: tuple = (),
    coord_attr_overrides: dict | None = None,
) -> xr.Dataset:
    """
    Build a dataset with projected x/y, 2-D lat/lon computed from TEST_CRS, a
    crs variable carrying WKT + CF grid mapping attributes and one data
    variable. ``x_units`` may be "m" or "km" (values are converted).
    """
    scale = 1000.0 if x_units == "km" else 1.0
    x_m = 500000.0 + np.arange(n_x) * 1000.0
    y_m = 5.0e6 - np.arange(n_y) * 1000.0
    xx, yy = np.meshgrid(x_m, y_m)
    lon, lat = Transformer.from_crs(TEST_CRS, "EPSG:4326", always_xy=True).transform(
        xx, yy
    )

    crs_attrs = TEST_CRS.to_cf()
    crs_attrs["spatial_ref"] = crs_attrs["crs_wkt"]
    crs_attrs.update(crs_attr_overrides or {})
    for key in drop_crs_attrs:
        crs_attrs.pop(key, None)

    coord_attrs = {
        "x": {
            "standard_name": "projection_x_coordinate",
            "units": x_units,
            "axis": "X",
        },
        "y": {
            "standard_name": "projection_y_coordinate",
            "units": x_units,
            "axis": "Y",
        },
        "lat": {"standard_name": "latitude", "units": "degrees_north"},
        "lon": {"standard_name": "longitude", "units": "degrees_east"},
    }
    for name, attrs in (coord_attr_overrides or {}).items():
        coord_attrs[name] = attrs

    time = pd.date_range("2020-01-01", periods=2, freq="h")
    ds = xr.Dataset(
        {
            "rr": (
                ("time", "y", "x"),
                np.zeros((2, n_y, n_x), dtype="float32"),
                {"grid_mapping": "crs"},
            ),
            "crs": ((), np.array(0, dtype="int32"), crs_attrs),
        },
        coords={
            "time": time,
            "x": ("x", x_m / scale, coord_attrs["x"]),
            "y": ("y", y_m / scale, coord_attrs["y"]),
            "lat": (("y", "x"), lat, coord_attrs["lat"]),
            "lon": (("y", "x"), lon, coord_attrs["lon"]),
        },
    )
    return ds


@pytest.fixture
def projected_dataset() -> xr.Dataset:
    return make_projected_dataset()
