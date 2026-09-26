import xarray as xr
from conftest import make_projected_dataset

from mlcast_dataset_validator.checks.data_vars.georeferencing import (
    check_georeferencing,
)

KWARGS = dict(
    require_geozarr=True,
    require_grid_mapping=True,
    crs_attrs=["spatial_ref", "crs_wkt"],
    require_bbox=True,
    require_cf_grid_mapping=True,
)


def _by_status(report, status):
    return [r for r in report.results if r.status == status]


def test_complete_grid_mapping_passes(projected_dataset):
    report = check_georeferencing(projected_dataset, **KWARGS)
    assert not report.has_fails()
    consistency = [
        r for r in _by_status(report, "PASS") if "consistency" in r.requirement
    ]
    # both the CF attributes and the WKT are compared with lat/lon
    assert len(consistency) == 2


def test_missing_grid_mapping_name_fails_with_hint():
    ds = make_projected_dataset(drop_crs_attrs=("grid_mapping_name",))
    report = check_georeferencing(ds, **KWARGS)
    fails = _by_status(report, "FAIL")
    assert len(fails) == 1
    assert "grid_mapping_name" in fails[0].detail
    # the message tells the contributor what to add, derived from the WKT
    assert "transverse_mercator" in fails[0].detail


def test_cf_attributes_only_are_used_not_the_wkt():
    # keep crs_wkt but strip every projection parameter: the WKT must not be
    # used as a fallback to make the CF check pass
    ds = make_projected_dataset(
        drop_crs_attrs=(
            "grid_mapping_name",
            "longitude_of_central_meridian",
            "false_easting",
            "scale_factor_at_central_meridian",
        )
    )
    assert check_georeferencing(ds, **KWARGS).has_fails()


def test_inconsistent_projection_parameter_fails():
    # false easting off by 50 km: the CF attributes no longer reproduce lat/lon
    ds = make_projected_dataset(crs_attr_overrides={"false_easting": 550000.0})
    report = check_georeferencing(ds, **KWARGS)
    fails = _by_status(report, "FAIL")
    assert len(fails) == 1
    assert "CF grid mapping attributes" in fails[0].detail
    assert "away from the stored latitude/longitude" in fails[0].detail


def test_kilometre_coordinates_are_scaled_for_the_cf_crs():
    ds = make_projected_dataset(x_units="km")
    report = check_georeferencing(ds, **KWARGS)
    cf_rows = [
        r for r in report.results if r.detail.startswith("CF grid mapping attributes")
    ]
    assert cf_rows and cf_rows[0].status == "PASS"


def test_without_cf_requirement_behaviour_is_unchanged():
    ds = make_projected_dataset(drop_crs_attrs=("grid_mapping_name",))
    kwargs = {**KWARGS, "require_cf_grid_mapping": False}
    assert not check_georeferencing(ds, **kwargs).has_fails()


def test_one_dimensional_latlon_gives_warning_not_fail(projected_dataset):
    ds = projected_dataset.assign_coords(
        lat=(
            "y",
            projected_dataset.lat.isel(x=0).values,
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        lon=(
            "x",
            projected_dataset.lon.isel(y=0).values,
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    )
    assert isinstance(ds, xr.Dataset)
    report = check_georeferencing(ds, **KWARGS)
    assert not report.has_fails()
    assert report.has_warnings()
