from typing import Sequence

import xarray as xr

from ...specs.reporting import ValidationReport, log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"


@log_function_call
def check_georeferencing(
    ds: xr.Dataset,
    *,
    require_geozarr: bool,
    crs_attrs: Sequence[str],
    require_bbox: bool,
    require_cf_grid_mapping: bool = False,
) -> ValidationReport:
    """Check georeferencing requirements.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    require_geozarr : bool
        Whether to require GeoZarr-compliant georeferencing.
    crs_attrs : Sequence[str]
        List of required attribute names on the CRS variable.
    require_bbox : bool
        Whether to require spatial bounding box coordinates.
    require_cf_grid_mapping : bool
        If True, require a ``grid_mapping`` attribute on data variables and
        validate that the referenced CRS variable has all CF-compliant grid
        mapping attributes derived from the WKT string via pyproj.
    """
    report = ValidationReport()
    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        data_array = ds[data_var]
        if require_cf_grid_mapping and "grid_mapping" not in data_array.attrs:
            report.add(
                SECTION_ID,
                f"Grid mapping for {data_var}",
                "FAIL",
                f"Data variable '{data_var}' is missing 'grid_mapping' attribute",
            )
            continue

        grid_mapping = data_array.attrs.get("grid_mapping", None)
        if grid_mapping and grid_mapping in ds.variables:
            crs_var = ds[grid_mapping]
            missing_attrs = [attr for attr in crs_attrs if attr not in crs_var.attrs]
            if missing_attrs:
                report.add(
                    SECTION_ID,
                    f"CRS attributes for {data_var}",
                    "FAIL",
                    f"CRS variable '{grid_mapping}' is missing attributes: {missing_attrs}",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"CRS attributes for {data_var}",
                    "PASS",
                    f"CRS variable '{grid_mapping}' has all required attributes",
                )

            if require_cf_grid_mapping:
                _check_cf_grid_mapping(report, grid_mapping, crs_var)
        else:
            report.add(
                SECTION_ID,
                f"Grid mapping for {data_var}",
                "FAIL",
                f"Data variable '{data_var}' references a non-existent grid mapping variable",
            )

    return report


def _check_cf_grid_mapping(
    report: ValidationReport,
    grid_mapping: str,
    crs_var: xr.DataArray,
) -> None:
    """Validate CF-compliant grid mapping attributes on the CRS variable."""
    wkt = crs_var.attrs.get("crs_wkt", "")
    if not wkt:
        report.add(
            SECTION_ID,
            "CF grid mapping validation",
            "FAIL",
            f"CRS variable '{grid_mapping}' has no 'crs_wkt' attribute to derive CF mapping from.",
        )
        return

    try:
        import pyproj

        crs = pyproj.CRS.from_wkt(wkt)
        cf_attrs = crs.to_cf()
        missing_cf = {k for k in cf_attrs if k not in crs_var.attrs}
        if missing_cf:
            report.add(
                SECTION_ID,
                "CF grid mapping attributes",
                "FAIL",
                f"CRS variable '{grid_mapping}' is missing CF grid mapping attributes: "
                f"{missing_cf}.\n"
                "Fix by adding this to your dataset generation code:\n"
                f"  >>> from pyproj import CRS\n"
                f"  >>> crs = CRS.from_wkt(ds.crs.attrs['crs_wkt'])\n"
                f"  >>> ds.crs.attrs.update(crs.to_cf())",
            )
        else:
            report.add(
                SECTION_ID,
                "CF grid mapping attributes",
                "PASS",
                f"CRS variable '{grid_mapping}' has all required CF grid mapping attributes: "
                f"{set(cf_attrs.keys())}.",
            )
    except ImportError:
        report.add(
            SECTION_ID,
            "CF grid mapping attributes",
            "WARNING",
            "pyproj is not installed; skipping CF grid mapping validation.",
        )
    except Exception as e:
        report.add(
            SECTION_ID,
            "CF grid mapping attributes",
            "WARNING",
            f"Could not validate CF grid mapping attributes: {e}",
        )
