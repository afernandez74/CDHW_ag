"""
download_arco_era5.py
=====================
Download ERA5 data from the ARCO-ERA5 Google Cloud Public Dataset
and save as yearly NetCDF files 

Dataset reference:
  https://cloud.google.com/storage/docs/public-datasets/era5
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
"""

import argparse
import logging
import os

import gcsfs
import xarray as xr
from tqdm import tqdm

import dask
dask.config.set({"array.slicing.split_large_chunks": True})

# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────

# Geographic bounding box: [North, West, South, East]  (degrees)
AREA = [55, -5, 41, 16]

# ARCO-ERA5 variable names to download.
VARIABLES = [
    # "volumetric_soil_water_layer_1",   # swvl1
    # "volumetric_soil_water_layer_2",   # swvl2
    "2m_temperature",   # 2m temperature
    "2m_dewpoint_temperature",   # 2m dewpoint temperature
    # "10m_u_component_of_wind",   # 10m u-component of wind
    # "10m_v_component_of_wind",   # 10m v-component of wind
    # "100m_u_component_of_wind",  # 100m u-component of wind
    # "100m_v_component_of_wind",  # 100m v-component of wind
    # "ssrd",  # surface solar radiation downwards
]

# Temporal domain 
YEAR_START = 1980
YEAR_END   = 2024          # inclusive; script will cap at last available date

# Output directory — NetCDF files are written here, one per year
OUTPUT_DIR = os.path.join(os.environ.get('ERA5_dat'), 'temps')


# ARCO-ERA5 Zarr store on Google Cloud Storage (public, no auth needed)
ZARR_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# NetCDF encoding — adjust chunk sizes / compression to taste
NC_ENCODING_DEFAULTS = {
    "zlib": True,
    "complevel": 4,
    "dtype": "float32",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def open_arco_era5(zarr_store: str) -> xr.Dataset:
    """
    Open the ARCO-ERA5 Zarr store anonymously
    """
    log.info("Opening ARCO-ERA5 Zarr store …")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(zarr_store, gcs=fs, check=False)
    ds = xr.open_zarr(store, consolidated=True)

    log.info("Store opened — available time range: %s → %s",
             str(ds.time.values[0])[:10],
             str(ds.time.values[-1])[:10])
    return ds

def select_variables(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Keep only the requested variables and raise a clear error if any are missing."""
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        available = sorted(ds.data_vars)
        raise ValueError(
            f"Variable(s) not found in dataset: {missing}\n"
            f"Available variables:\n  " + "\n  ".join(available)
        )
    return ds[variables]


def select_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    """
    Spatial subset in ERA5 native 0–360 longitude coordinates.
    Works safely for domains crossing Greenwich.
    """
    north, west, south, east = area

    west_360 = west % 360
    east_360 = east % 360

    lat_slice = slice(north, south)

    if west_360 <= east_360:
        ds = ds.sel(
            latitude=lat_slice,
            longitude=slice(west_360, east_360)
        )

    else:
        # Crosses Greenwich
        ds_west = ds.sel(
            latitude=lat_slice,
            longitude=slice(west_360, 360)
        )

        ds_east = ds.sel(
            latitude=lat_slice,
            longitude=slice(0, east_360)
        )

        ds = xr.concat([ds_west, ds_east], dim="longitude")

    log.info(
        "Spatial subset: lat [%.2f → %.2f], lon [%.2f → %.2f] (%d grid cells)",
        float(ds.latitude.max()),
        float(ds.latitude.min()),
        float(ds.longitude.min()),
        float(ds.longitude.max()),
        ds.sizes["longitude"]
    )

    return ds

def build_encoding(ds: xr.Dataset) -> dict:
    """Build per-variable NetCDF encoding dict."""
    return {var: NC_ENCODING_DEFAULTS.copy() for var in ds.data_vars}


def download_year(ds: xr.Dataset, year: int, output_dir: str) -> None:
    """Select one calendar year, load into memory, and write to NetCDF."""
    out_path = os.path.join(output_dir, f"era5_{year}.nc")

    if os.path.exists(out_path):
        log.info("Year %d — file already exists, skipping: %s", year, out_path)
        return

    t_min = str(ds.time.values[0])[:10]
    t_max = str(ds.time.values[-1])[:10]
    year_start = f"{year}-01-01"
    year_end   = f"{year}-12-31"

    if year_start > t_max or year_end < t_min:
        log.warning("Year %d — no data available in store, skipping.", year)
        return

    t0 = max(year_start, t_min)
    t1 = min(year_end,   t_max)

    log.info("Year %d — selecting %s → %s …", year, t0, t1)
    
    ds_year = ds.sel(time=slice(t0, t1))

    log.info(
        "Year %d — writing %d timesteps directly to disk (streaming)",
        year, ds_year.sizes["time"]
    )

    encoding = build_encoding(ds_year)

    ds_year.to_netcdf(
        out_path,
        encoding=encoding,
        compute=True
    )
    
    ds_year.close()
    log.info("Year %d — done ✓", year)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ARCO-ERA5 data for a single year or all years."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Calendar year to download (e.g. 1995). "
            "If omitted, all years from YEAR_START to YEAR_END are downloaded sequentially."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open remote dataset
    ds_full = open_arco_era5(ZARR_STORE)
    
    # Select variables
    log.info("Selecting variables: %s", VARIABLES)
    ds = select_variables(ds_full, VARIABLES)

    # Spatial subset
    ds = select_area(ds, AREA)

    log.info(
        "Subset size: %d lat × %d lon × %d time",
        ds.sizes["latitude"],
        ds.sizes["longitude"],
        ds.sizes["time"]
    )
    
    ds = ds.chunk({"time": 24})

    if args.year is not None:
        # ── Single-year mode (used by SLURM array tasks) ──────────────────
        log.info("Single-year mode: year = %d", args.year)
        download_year(ds, args.year, OUTPUT_DIR)
    else:
        # ── Sequential mode (local / interactive use) ─────────────────────
        last_available_year = int(str(ds.time.values[-1])[:4])
        year_end_eff = min(YEAR_END, last_available_year)
        years = list(range(YEAR_START, year_end_eff + 1))
        log.info("Sequential mode: downloading %d years (%d → %d) …",
                 len(years), years[0], years[-1])
        for year in tqdm(years, desc="Years", unit="yr"):
            try:
                download_year(ds, year, OUTPUT_DIR)
            except Exception as exc:
                log.error("Year %d — FAILED: %s", year, exc)

    log.info("Finished. Output directory: %s", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
