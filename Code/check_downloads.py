#!/usr/bin/env python3
#%%
"""
check_downloads.py

Scans the raw download directory and reports which variable/year
combinations are missing or empty. Run after the SLURM array completes.

Usage:
    python check_downloads.py
    python check_downloads.py --resubmit   # prints sbatch commands to rerun missing jobs
"""

import argparse
from pathlib import Path
from config import VARIABLES, YEAR_START, YEAR_END, RAW_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--resubmit', action='store_true')
args = parser.parse_args()

VARS      = list(VARIABLES.keys())
missing   = []
corrupted = []

for var in VARS:
    for year in range(YEAR_START, YEAR_END + 1):
        f = Path(RAW_DIR) / var / f'ERA5_{var}_{year}.nc'
        if not f.exists():
            missing.append((var, year))
        elif f.stat().st_size < 1e6:   # <1MB is suspiciously small for hourly ERA5
            corrupted.append((var, year))

print(f"Missing  : {len(missing)}")
print(f"Corrupted: {len(corrupted)}")

issues = missing + corrupted
if issues:
    print("\nProblem files:")
    for var, year in issues:
        print(f"  {var} {year}")

    if args.resubmit:
        print("\nResubmit commands:")
        for var, year in issues:
            print(f"  VAR={var} YEAR={year} sbatch --array=0 submit_download.sh")
else:
    print("\nAll files present and non-empty. Ready for preprocessing.")

# %%
