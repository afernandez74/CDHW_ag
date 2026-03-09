#!/usr/bin/env python3
"""
00_download_ERA5.py

ERA5 hourly download worker for Snellius HPC.
Called by SLURM array job — reads VAR and YEAR from environment variables.

Usage (direct):
    VAR=t2m YEAR=1980 python 00_download_ERA5.py

Usage (via SLURM):
    sbatch submit_download.sh
"""

import cdsapi
import os
import sys
from pathlib import Path
from config import VARIABLES, AREA, RAW_DIR

# ── Read job parameters from environment ──────────────────────────────────────
VAR_SHORT = os.environ.get('VAR')
YEAR      = os.environ.get('YEAR')

if VAR_SHORT is None or YEAR is None:
    print("ERROR: VAR and YEAR must be set as environment variables.")
    sys.exit(1)

YEAR = int(YEAR)

if VAR_SHORT not in VARIABLES:
    print(f"ERROR: '{VAR_SHORT}' not in config.VARIABLES. "
          f"Valid options: {list(VARIABLES.keys())}")
    sys.exit(1)

CDS_NAME = VARIABLES[VAR_SHORT]

# ── Output path ───────────────────────────────────────────────────────────────
out_dir  = Path(RAW_DIR) / VAR_SHORT
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f'ERA5_{VAR_SHORT}_{YEAR}.nc'

# ── Skip if already downloaded ────────────────────────────────────────────────
if out_file.exists() and out_file.stat().st_size > 0:
    print(f"Already exists and non-empty: {out_file}. Skipping.")
    sys.exit(0)

# ── Download ──────────────────────────────────────────────────────────────────
print(f"Downloading {VAR_SHORT} ({CDS_NAME}) for {YEAR}...")

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type' : 'reanalysis',
        'variable'     : CDS_NAME,
        'year'         : str(YEAR),
        'month'        : [f'{m:02d}' for m in range(1, 13)],
        'day'          : [f'{d:02d}' for d in range(1, 32)],
        'time'         : [f'{h:02d}:00' for h in range(24)],
        'area'         : AREA,
        'format'       : 'netcdf',
    },
    str(out_file)
)

print(f"Done: {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
