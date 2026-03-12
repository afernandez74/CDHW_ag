"""
ERA5 ETL Pipeline: ARCO-ERA5 NetCDF to Zarr Converter
------------------------------------------
Purpose:
    Converts raw annual ERA5 NetCDF files into a single consolidated Zarr store.
    Optimizes data for analysis by performing expensive operations upfront.

Transformations:
    - Wraps Longitude from [0, 360] to [-180, 180].
    - Sorts coordinates (time, lat, lon) for contiguous disk access.
    - Renames variables to their CF-compliant 'short_name'.
    - Re-chunks data for high-performance time-series analysis over the Netherlands.

Output: 
    A 'ready-to-analyze' Zarr store with consolidated metadata.
"""

import os
import xarray as xr
from pathlib import Path

# Spatial tile size for time-series focused chunks.
LAT_CHUNK = 10
LON_CHUNK = 10


def convert_nc_to_zarr(folder_name, base_dir):
    """
    ETL: Processes raw NetCDF for a specific variable and bakes 
    standardized coordinates into a high-performance Zarr store.
    """
    # 1. Setup paths: BASE_DAT / folder_name / (nc_files)
    input_path = Path(base_dir) / folder_name
    
    # Define a subfolder for the processed data
    output_folder = input_path / "processed"
    output_folder.mkdir(parents=True, exist_ok=True) # Creates 'processed' folder if missing
    
    zarr_path = output_folder / f"{folder_name}_cleaned.zarr"
    
    # 2. Find files
    files = sorted(input_path.glob("*.nc"))
    if not files:
        print(f"No .nc files found in {input_path}")
        return

    print(f"--- Processing {folder_name} ---")
    
    # 3. Load and Transform
    # We open with parallel=True to speed up the initial read
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True, chunks = {})
    
    # Standardize Longitude and Sort (Crucial for performance)
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    ds = ds.sortby(['time', 'latitude', 'longitude'])
    
    # Rename variables based on their internal metadata 'short_name'
    rename_dict = {var: ds[var].attrs.get('short_name', var) for var in ds.data_vars}
    ds = ds.rename(rename_dict)
    
    # 4. Re-chunk for region using -1 for spatial dims
    ds = ds.chunk({"time": -1, "latitude": LAT_CHUNK, "longitude": LON_CHUNK})
    
    # 5. Write to Zarr
    # consolidated=True makes loading the Zarr later almost instantaneous
    ds.to_zarr(zarr_path, mode='w', consolidated=True)
    print(f"Success! Saved to: {zarr_path}\n")

# --- EXECUTION ---
#
BASE_DAT = Path(os.environ["ERA5_dat"])

# List the folders you want to process
FOLDER = "temps"

convert_nc_to_zarr(FOLDER, BASE_DAT)