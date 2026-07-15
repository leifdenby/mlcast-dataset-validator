"""Tests for coordinate names checks."""

import numpy as np
import xarray as xr

from mlcast_dataset_validator.checks.coords.names import check_coordinate_names


def _make_ds(x_attrs=None, y_attrs=None):
    """Helper: create a minimal dataset with configurable x/y attrs."""
    x_attrs = x_attrs or {"units": "m"}
    y_attrs = y_attrs or {"units": "m"}
    return xr.Dataset(
        coords={
            "time": [0, 1],
            "y": ("y", [0, 1], y_attrs),
            "x": ("x", [0, 1], x_attrs),
            "lat": (
                ("y", "x"),
                np.zeros((2, 2)),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("y", "x"),
                np.zeros((2, 2)),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )


class TestCheckCoordinateNames:
    def test_legacy_name_fallback_passes_non_strict(self):
        """Name-only match should pass when strict=False (backward compat)."""
        ds = _make_ds(x_attrs={"units": "m"}, y_attrs={"units": "m"})
        report = check_coordinate_names(
            ds,
            require_time_coord=True,
            require_projected_coords=True,
            require_latlon_coords=True,
            strict=False,
        )
        assert report.has_fails() is False
        assert any(
            "WARNING" in str(r) or "matched by name" in str(r) for r in report.results
        )

    def test_name_fallback_fails_strict(self):
        """Name-only match should fail when strict=True."""
        ds = _make_ds(x_attrs={"units": "m"}, y_attrs={"units": "m"})
        report = check_coordinate_names(
            ds,
            require_time_coord=True,
            require_projected_coords=True,
            require_latlon_coords=True,
            strict=True,
        )
        assert report.has_fails() is True

    def test_standard_name_passes_strict(self):
        """With standard_name, should pass even in strict mode."""
        ds = _make_ds(
            x_attrs={"standard_name": "projection_x_coordinate", "units": "m"},
            y_attrs={"standard_name": "projection_y_coordinate", "units": "m"},
        )
        report = check_coordinate_names(
            ds,
            require_time_coord=True,
            require_projected_coords=True,
            require_latlon_coords=True,
            strict=True,
        )
        assert report.has_fails() is False

    def test_axis_attribute_passes_strict(self):
        """With axis=X/Y, should pass even in strict mode."""
        ds = _make_ds(
            x_attrs={"axis": "X", "units": "m"},
            y_attrs={"axis": "Y", "units": "m"},
        )
        report = check_coordinate_names(
            ds,
            require_time_coord=True,
            require_projected_coords=True,
            require_latlon_coords=True,
            strict=True,
        )
        assert report.has_fails() is False
