"""Tests for georeferencing checks."""

import numpy as np
import xarray as xr

from mlcast_dataset_validator.checks.data_vars.georeferencing import (
    check_georeferencing,
)


def _make_minimal_crs_dataset(cf_compliant=False):
    """Create a minimal dataset with a CRS variable for testing."""
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
    crs_wkt = (
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
    ds["crs"] = xr.DataArray(np.float64(0.0))
    if cf_compliant:
        import pyproj

        crs = pyproj.CRS.from_wkt(crs_wkt)
        ds.crs.attrs = crs.to_cf()
        ds.crs.attrs["crs_wkt"] = crs_wkt
    else:
        ds.crs.attrs = {"spatial_ref": crs_wkt, "crs_wkt": crs_wkt}
    return ds


class TestCheckGeoreferencing:
    def test_requires_grid_mapping_attribute(self):
        ds = _make_minimal_crs_dataset()
        del ds["prate"].attrs["grid_mapping"]
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any(
            "grid_mapping" in str(entry) and "FAIL" in str(entry)
            for entry in report.results
        )

    def test_requires_crs_attrs(self):
        ds = _make_minimal_crs_dataset()
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            crs_attrs=["crs_wkt", "spatial_ref"],
            require_bbox=False,
        )
        assert any("PASS" in str(entry) for entry in report.results)

    def test_cf_grid_mapping_passes(self):
        ds = _make_minimal_crs_dataset(cf_compliant=True)
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any(
            "CF grid mapping" in str(entry) and "PASS" in str(entry)
            for entry in report.results
        )

    def test_cf_grid_mapping_fails_without_grid_mapping_name(self):
        ds = _make_minimal_crs_dataset(cf_compliant=False)
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any(
            "missing" in str(entry).lower() and "FAIL" in str(entry)
            for entry in report.results
        )

    def test_cf_grid_mapping_skipped_by_default(self):
        ds = _make_minimal_crs_dataset(cf_compliant=False)
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            crs_attrs=["crs_wkt"],
            require_bbox=False,
        )
        assert all("CF grid mapping" not in str(entry) for entry in report.results)

    def test_cf_grid_mapping_missing_crs_wkt(self):
        ds = _make_minimal_crs_dataset(cf_compliant=False)
        del ds.crs.attrs["crs_wkt"]
        report = check_georeferencing(
            ds,
            require_geozarr=False,
            crs_attrs=[],
            require_bbox=False,
            require_cf_grid_mapping=True,
        )
        assert any("no 'crs_wkt'" in str(entry).lower() for entry in report.results)
