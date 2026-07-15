"""Tests for CF grid mapping validation."""

import numpy as np
import xarray as xr

from mlcast_dataset_validator.checks.data_vars.georeferencing import (
    check_georeferencing,
)

POLAR_STEREO_WKT = (
    'PROJCS["unknown",'
    'GEOGCS["unknown",'
    'DATUM["unknown",SPHEROID["unknown",6378137,298.252840776245]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",45],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)


def _make_ds(cf_compliant=False):
    """Create a minimal dataset with configurable CRS."""
    ds = xr.Dataset(
        coords={
            "time": [0, 1],
            "y": [0, 1],
            "x": [0, 1],
            "lat": (("y", "x"), np.zeros((2, 2))),
            "lon": (("y", "x"), np.zeros((2, 2))),
        },
        data_vars={
            "prate": xr.DataArray(
                np.zeros((2, 2, 2), dtype=np.float32),
                dims=("time", "y", "x"),
                attrs={"grid_mapping": "crs"},
            ),
        },
    )
    ds["crs"] = xr.DataArray(np.float64(0.0))
    if cf_compliant:
        import pyproj

        crs = pyproj.CRS.from_wkt(POLAR_STEREO_WKT)
        ds.crs.attrs = crs.to_cf()
        ds.crs.attrs["crs_wkt"] = POLAR_STEREO_WKT
    else:
        ds.crs.attrs = {"spatial_ref": POLAR_STEREO_WKT, "crs_wkt": POLAR_STEREO_WKT}
    return ds


class TestCheckGeoreferencing:
    def test_non_cf_crs_passes_without_cf_check(self):
        """Without cf_grid_mapping, legacy attrs should still pass."""
        ds = _make_ds(cf_compliant=False)
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            require_grid_mapping=True,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=False,
        )
        assert not any("FAIL" in str(r) for r in report.results)

    def test_non_cf_crs_fails_with_cf_check(self):
        """Missing grid_mapping_name should fail when check is enabled."""
        ds = _make_ds(cf_compliant=False)
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            require_grid_mapping=True,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any(
            "grid_mapping_name" in str(r) and "FAIL" in str(r) for r in report.results
        )

    def test_cf_compliant_crs_passes(self):
        """CRS with all CF grid mapping attrs should pass."""
        ds = _make_ds(cf_compliant=True)
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            require_grid_mapping=True,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any("PASS" in str(r) and "has all CF" in str(r) for r in report.results)

    def test_missing_crs_wkt_fails(self):
        """Without crs_wkt, CF attrs can't be derived."""
        ds = _make_ds(cf_compliant=False)
        del ds.crs.attrs["crs_wkt"]
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            require_grid_mapping=True,
            crs_attrs=[],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any("no 'crs_wkt'" in str(r).lower() for r in report.results)

    def test_missing_one_projection_param_fails(self):
        """Missing a single projection param should be detected."""
        ds = _make_ds(cf_compliant=True)
        del ds.crs.attrs["standard_parallel"]
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            require_grid_mapping=True,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any(
            "standard_parallel" in str(r) and "FAIL" in str(r) for r in report.results
        )
