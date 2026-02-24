"""
00_preprocess.py

One-time preprocessing step: converts raw ERA5 NetCDF files to Zarr format
for efficient chunked access in subsequent analysis scripts.

Includes the following variables, necessary for the Compound Drought & Heatwave Analysis: 
1. 2m air temperature
2. 2m dew point temperature
3. Surface short-wave radiation downwards
4. Volumetric soil water layer 1
5. Volumetric soil water layer 2

Run once before any other scripts:
    python scripts/00_preprocess.py

Output:
    data/processed/ERA5_NL.zarr
"""
#%%
from pathlib import Path
import xarray as xr
import os
print("Working directory:", os.getcwd())

#%%
# ── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH  = Path('./../data/ERA5/')
ZARR_PATH = Path('./../data/ERA5_zarr/ERA5_NL.zarr')

# ── Guard: skip if already done ──────────────────────────────────────────────
if ZARR_PATH.exists():
    print(f"Zarr store already exists at {ZARR_PATH}. Delete it to rerun.")
else:
    print("Converting ERA5 NetCDFs to Zarr — this runs once and takes ~10-20 min...")

    file_list = sorted([str(p) for p in RAW_PATH.glob("**/*.nc")])

    ds = xr.open_mfdataset(
        file_list,
        combine='by_coords',
        chunks={'valid_time': 24 * 365},
        parallel=False
    )[['t2m', 'd2m', 'ssrd','swvl1','swvl2']]

    # uniform chunk size for zarr
    chunk_size = 24 * 365  # 8760 hours — uniform and clean

    ds_rechunked = ds.chunk({'valid_time': chunk_size, 'latitude': -1, 'longitude': -1})
    
    # Write to Zarr
    ds_rechunked.to_zarr(ZARR_PATH, mode='w')

    print(f"Done. Zarr store written to {ZARR_PATH}")

# %%
