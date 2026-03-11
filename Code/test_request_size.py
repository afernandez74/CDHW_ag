# test_request_size.py
import cdsapi
import os
from pathlib import Path
from config import VARIABLES, AREA, RAW_DIR

VAR_SHORT   = os.environ.get('VAR', 'd2m')
DECADE_START = int(os.environ.get('DECADE_START', '1980'))
N_YEARS     = int(os.environ.get('N_YEARS', '2'))

years = [str(y) for y in range(DECADE_START, DECADE_START + N_YEARS)]
out_file = Path(RAW_DIR) / VAR_SHORT / f'test_{VAR_SHORT}_{years[0]}_{years[-1]}.nc'
out_file.parent.mkdir(parents=True, exist_ok=True)

print(f"Testing {N_YEARS} years: {years[0]}–{years[-1]}")

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type' : 'reanalysis',
        'variable'     : VARIABLES[VAR_SHORT],
        'year'         : years,
        'month'        : [f'{m:02d}' for m in range(1, 13)],
        'day'          : [f'{d:02d}' for d in range(1, 32)],
        'time'         : [f'{h:02d}:00' for h in range(24)],
        'area'         : AREA,
        'format'       : 'netcdf',
    },
    str(out_file)
)

print(f"SUCCESS: {N_YEARS} years works — {out_file.stat().st_size / 1e9:.2f} GB")