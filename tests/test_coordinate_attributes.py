from conftest import make_projected_dataset

from mlcast_dataset_validator.checks.coords.names import check_coordinate_attributes


def _fails(report):
    return [r for r in report.results if r.status == "FAIL"]


def test_complete_attributes_pass(projected_dataset):
    report = check_coordinate_attributes(projected_dataset)
    assert not report.has_fails()
    assert len([r for r in report.results if r.status == "PASS"]) == 4


def test_projected_coordinate_without_attributes_fails():
    # 'x' is still identified by its name, so the missing metadata is reported
    ds = make_projected_dataset(coord_attr_overrides={"x": {"units": "m"}})
    report = check_coordinate_attributes(ds)
    fails = _fails(report)
    assert len(fails) == 1
    assert "'x'" in fails[0].requirement
    assert "standard_name" in fails[0].detail and "axis" in fails[0].detail
    assert "units" not in fails[0].detail.replace("expected", "")


def test_wrong_latitude_units_fails():
    ds = make_projected_dataset(
        coord_attr_overrides={"lat": {"standard_name": "latitude", "units": "degrees"}}
    )
    report = check_coordinate_attributes(ds)
    fails = _fails(report)
    assert len(fails) == 1 and "'lat'" in fails[0].requirement
    assert "degrees_north" in fails[0].detail


def test_kilometre_units_accepted():
    ds = make_projected_dataset(x_units="km")
    assert not check_coordinate_attributes(ds).has_fails()


def test_categories_can_be_skipped():
    ds = make_projected_dataset(coord_attr_overrides={"x": {}, "y": {}})
    assert (
        check_coordinate_attributes(ds, require_projected_attrs=False).has_fails()
        is False
    )
    assert check_coordinate_attributes(ds).has_fails() is True
