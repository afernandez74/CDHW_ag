#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Soil Moisture Deficit Index (SMDI) from ERA5 soil moisture data.

Workflow converted from load_soil_moisture_data.ipynb.
Algorithm: Narasimhan and Srinivasan (2005), adapted for daily resolution (Guo et al. 2023).
Baseline period: 1980-2023.

@author: afer
"""

#%%
# Loading ERA5 soil moisture data from Zarr store
# NOTE: run 00_pre_process_ERA5_tozarr.py first to create the Zarr store

import xarray as xr
import numpy as np
import numba
import pandas as pd
import geopandas as gpd
import rioxarray  
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dask
#%% Load soil moisture dataset; analysis on all land grid cells, map shows Europe coast outline

# ── Paths ─────────────────────────────────────────────────────────────────────
ZARR_DIR           = Path('./../Data/ERA5_FRBENLDE_zarr/ERA5_FRBENLDE_regridded.zarr/')
COUNTRIES_SHP_PATH = Path('./../Data/countries_shp/ne_110m_admin_0_countries.shp')

# ── Load per-variable Zarr stores and merge into one Dataset ──────────────────
sm = xr.merge([
    xr.open_zarr(ZARR_DIR / 'ERA5_FRBENLDE_swvl1.zarr'),
    xr.open_zarr(ZARR_DIR / 'ERA5_FRBENLDE_swvl2.zarr'),
])

# ── Set spatial metadata (no clip: run analysis on all grid cells in the data) ─
sm = sm.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
sm = sm.rio.write_crs("EPSG:4326")

# Use full domain for analysis (all land grid cells in the data)
sm_region = sm

# Load Europe coast outline for map display only (optional)
countries_gdf = gpd.read_file(str(COUNTRIES_SHP_PATH))

print(sm_region)
print(f"\nLatitude  range : {float(sm_region.latitude.min()):.2f} – {float(sm_region.latitude.max()):.2f}")
print(f"Longitude range : {float(sm_region.longitude.min()):.2f} – {float(sm_region.longitude.max()):.2f}")
print(f"Time range      : {str(sm_region.valid_time.values[0])[:10]} – {str(sm_region.valid_time.values[-1])[:10]}")
#%% define functions
def calculate_sm_doy_climatology(sm_daily, baseline_start=None, baseline_end=None):
    """
    Compute DOY climatological median, min, and max of daily soil moisture.
    These are the inputs required for the SD calculation in SMDI.
    """
    years = pd.DatetimeIndex(sm_daily['valid_time'].values).year
    baseline_start = baseline_start or str(years.min())
    baseline_end   = baseline_end   or str(years.max())

    sm_baseline = (
        sm_daily
        .sel(valid_time=slice(baseline_start, baseline_end))
        .chunk({'valid_time': -1, 'latitude': -1, 'longitude': -1})
        .assign_coords(doy=lambda x: x.valid_time.dt.dayofyear)
    )

    sm_median_doy = sm_baseline.groupby('doy').median('valid_time')
    sm_min_doy    = sm_baseline.groupby('doy').min('valid_time')
    sm_max_doy    = sm_baseline.groupby('doy').max('valid_time')

    return sm_median_doy, sm_min_doy, sm_max_doy

def calculate_sd(sm_daily, sm_median_doy, sm_min_doy, sm_max_doy):
    """
    Calculate daily Soil water Deficit/surplus (SD) scaled to [-100, +100].
    """
    sm_daily = sm_daily.assign_coords(doy=sm_daily.valid_time.dt.dayofyear)

    median_aligned = sm_median_doy.sel(doy=sm_daily.doy)
    min_aligned    = sm_min_doy.sel(doy=sm_daily.doy)
    max_aligned    = sm_max_doy.sel(doy=sm_daily.doy)

    sd_surplus = (sm_daily - median_aligned) / (max_aligned - median_aligned) * 100
    sd_deficit = (sm_daily - median_aligned) / (median_aligned - min_aligned) * 100

    sd = xr.where(sm_daily >= median_aligned, sd_surplus, sd_deficit)
    sd.attrs.update({'long_name': 'Soil water Deficit/Surplus', 'units': 'dimensionless [-100, 100]'})
    return sd

@numba.njit(nopython=True, parallel=True, cache=True)
def calculate_smdi_numba(sd_array, alpha):
    """
    Recursively calculate SMDI values over a 3D (time, lat, lon) array.

    Implements the recursive formula from Narasimhan & Srinivasan (2005):
        SMDI_t = alpha * SMDI_(t-1) + SD_t / 50

    Uses numba JIT compilation with parallelisation over the spatial dimensions
    for performance. The time loop must remain sequential due to the recursion
    dependency (each timestep depends on the previous).

    NaN/inf values in sd_array are handled gracefully — when SD is non-finite
    (e.g. grid cells with missing data), the previous SMDI
    value is decayed without adding a new SD contribution.

    Parameters
    ----------
    sd_array : numpy.ndarray
        3D array of Soil water Deficit/Surplus values (time, latitude, longitude),
        scaled to [-100, +100]. Must be np.float64.
    alpha : float
        Memory decay factor controlling how much weight is given to the previous
        timestep. Default is 0.5 

    Returns
    -------
    smdi_array : numpy.ndarray
        3D array of SMDI values (time, latitude, longitude), same shape as
        sd_array. Positive values indicate wet conditions; negative values
        indicate drought.

    Notes
    -----
    This function is compiled by numba on first call (JIT), causing a one-time
    delay of a few seconds. The `cache=True` flag saves the compiled version to
    disk so subsequent Python sessions skip recompilation.
    Do not wrap this function in Dask or Xarray — it operates purely on NumPy
    arrays. Use calculate_smdi() for the Xarray-compatible interface.
    """
    n_time, n_lat, n_lon = sd_array.shape
    smdi_array = np.zeros_like(sd_array)
    smdi_array[0, :, :] = sd_array[0, :, :] / 50.0
    for lat in numba.prange(n_lat):
        for lon in range(n_lon):
            for t in range(1, n_time):
                sd_val = sd_array[t, lat, lon]
                smdi_prev = smdi_array[t-1, lat, lon]
                if np.isfinite(sd_val):
                    smdi_array[t, lat, lon] = alpha * smdi_prev + sd_val / 50.0
                else:
                    smdi_array[t, lat, lon] = alpha * smdi_prev
    return smdi_array

def calculate_smdi(sd, alpha=0.5):
    """
    Calculate Soil Moisture Deficit Index (SMDI) using a recursive formula
    following Narasimhan and Srinivasan (2005).

    Recursion: SMDI_t = alpha * SMDI_(t-1) + SD_t / 50

    The recursive step is implemented in numba for performance since it is
    inherently sequential and cannot be vectorized across time.

    Parameters
    ----------
    sd : xarray.DataArray
        Daily Soil water Deficit/Surplus values (dimensionless, [-100, 100]);
        must have 'valid_time', 'latitude', 'longitude' dimensions.
    alpha : float, optional
        Memory decay factor. Default is 0.5 following Narasimhan & Srinivasan (2005).

    Returns
    -------
    smdi : xarray.DataArray
        SMDI values with same coordinates as input sd.
    """
    # Ensure data is computed and in memory before passing to numba
    # (numba cannot operate on lazy Dask arrays)
    sd_computed = sd.compute()

    # Extract raw NumPy array — numba requires pure NumPy, not xarray
    sd_array = sd_computed.values.astype(np.float64)

    # Run recursive calculation
    smdi_array = calculate_smdi_numba(sd_array, alpha)

    # Wrap result back into an xarray DataArray, restoring all coordinates
    smdi = xr.DataArray(
        smdi_array,
        coords=sd_computed.coords,
        dims=sd_computed.dims,
        attrs={
            'long_name'   : 'Soil Moisture Deficit Index',
            'alpha'       : alpha,
        }
    )

    return smdi

#%% 
# Calculate weighted average soil moisture based on thickness of soil layers
# Thickness of soil layer 1: 0-7cm
# Thickness of soil layer 2: 7-28cm
# Weighted average soil moisture = (swvl1 * 7 + swvl2 * 21) / (7 + 21)
# Decimal multipliers: 0.25 (0-7cm) and 0.75 (7-28cm)

sm_region_wavg = (sm_region.swvl1 * 0.25 + sm_region.swvl2 * 0.75)
sm_region_wavg_dly= sm_region_wavg.resample(valid_time='1D').mean()
sm_region_wavg_dly = dask.compute(sm_region_wavg_dly)[0]
#%% 
# Sample gridcell of SM daily avg results 
# 1. Box & Whisker plot and empirical distribution for central gridcell's sm_daily_avg

# Select a central sample gridcell
lat_idx = sm_region_wavg_dly.latitude.size // 2
lon_idx = sm_region_wavg_dly.longitude.size // 2

central_lat = sm_region_wavg_dly.latitude.values[lat_idx]
central_lon = sm_region_wavg_dly.longitude.values[lon_idx]

sm_central = sm_region_wavg_dly.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='valid_time', how='any')

central_sm_values = sm_central.values

fig, axs = plt.subplots(1, 2, figsize=(11,5))
fig.suptitle('SM Daily Avg: Central Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)), fontsize=14)

# Box and whisker plot
axs[0].boxplot(central_sm_values, vert=True, patch_artist=True, showmeans=True)
axs[0].set_ylabel('SM Daily Avg (m3/m3)')
axs[0].set_title('Box & Whisker Plot')

# Empirical distribution (histogram with KDE)
axs[1].hist(central_sm_values, bins=40, density=True, alpha=0.6, color='tab:blue', edgecolor="black", label='Histogram')
sns.kdeplot(central_sm_values, ax=axs[1], color='tab:red', label='KDE')

axs[1].set_xlabel('SM Daily Avg (m3/m3)')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical Distribution')
axs[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(Path(path_save, "sm_daily_avg_central_gridcell_distribution.png"), dpi=300)
plt.show()




#%% 
# Compute Soil Moisture Deficit Index (SMDI)
# Calculate daily Soil Moisture Deficit Index (SMDI) from ERA5 soil moisture (all land grid cells in the data).
# Algorithm: Based on Narasimhan and Srinivasan (2005), adapted for daily resolution (Guo et al. 2023).
# Baseline period: 1980-2023.
# SMDI calculated from weighted average soil moisture from two top soil layers

sm_median_doy, sm_min_doy, sm_max_doy = calculate_sm_doy_climatology(
    sm_region_wavg_dly,
)

# Step 2: Soil water Deficit/Surplus [-100, 100]
sd = calculate_sd(sm_region_wavg_dly, sm_median_doy, sm_min_doy, sm_max_doy)

# Step 3: Recursive SMDI
smdi = calculate_smdi(sd, alpha=0.5)

#%% 
# Sample plot of SMDI results
# 1. Box & Whisker plot and empirical distribution for central gridcell's vpd_max_daily

# Select a central sample gridcell
lat_idx = smdi.latitude.size // 2
lon_idx = smdi.longitude.size // 2

central_lat = smdi.latitude.values[lat_idx]
central_lon = smdi.longitude.values[lon_idx]

smdi_central = smdi.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='valid_time', how='any')

central_smdi_values = smdi_central.values

fig, axs = plt.subplots(1, 2, figsize=(11,5))
fig.suptitle('SMDI: Central Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)), fontsize=14)

# Box and whisker plot
axs[0].boxplot(central_smdi_values, vert=True, patch_artist=True, showmeans=True)
axs[0].set_ylabel('SMDI (dimensionless)')
axs[0].set_title('Box & Whisker Plot')

# Empirical distribution (histogram with KDE)
axs[1].hist(central_smdi_values, bins=40, density=True, alpha=0.6, color='tab:blue', edgecolor="black", label='Histogram')
sns.kdeplot(central_smdi_values, ax=axs[1], color='tab:red', label='KDE')

axs[1].set_xlabel('SMDI (dimensionless)')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical Distribution')
axs[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(Path(path_save, "smdi_central_gridcell_distribution.png"), dpi=300)
plt.show()

# 2. Plot mean and standard deviation envelope timeseries for DOY climatology (mean year)
import calendar
from matplotlib.ticker import MultipleLocator, FuncFormatter

fig, ax = plt.subplots(figsize=(10,5))
days = sm_median_doy['doy']
median = sm_median_doy.isel(latitude=lat_idx, longitude=lon_idx)
min = sm_min_doy.isel(latitude=lat_idx, longitude=lon_idx)
max = sm_max_doy.isel(latitude=lat_idx, longitude=lon_idx)

median_vals = median.values
min_vals = min.values
max_vals = max.values

ax.plot(days, median_vals, label='Median SM', color='navy')
ax.fill_between(days, min_vals, max_vals, color='lightblue', alpha=0.5, label='Max and Min bounds')

# Set x-axis to display month names at the first day of each month
month_starts = [pd.Timestamp(year=2001, month=i, day=1).dayofyear for i in range(1,13)]  # 2001 is not a leap year
month_names = [calendar.month_abbr[i] for i in range(1, 13)]

ax.set_xticks(month_starts)
ax.set_xticklabels(month_names)

ax.set_xlabel('Month')
ax.set_ylabel('SM (dimensionless)')
ax.set_title('Mean Year: SM Climatology (Max and Min bounds)\nCentral Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)))
ax.legend()
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig(Path(path_save, "sm_mean_std_envelope_central_gridcell.png"), dpi=300)
plt.show()


#%% Quick map of average soil moisture (coastal outline of Europe only for boundary)
# Map plot with LambertConformal projection; extent from data, Europe coast as outline

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
    central_longitude=5,
    central_latitude=40
))

lon_min = sm_region.longitude.min().values
lon_max = sm_region.longitude.max().values
lat_min = sm_region.latitude.min().values
lat_max = sm_region.latitude.max().values
extent = [lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Europe coastal outline only; no land/ocean fill so all grid cells in the data are visible
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle='--')

gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
gl.top_labels = False
gl.right_labels = False

im = sm_region_wavg_dly.mean(dim='valid_time').plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='YlGnBu',
    cbar_kwargs={
        'label': f"Average {sm_region.swvl1.attrs.get('long_name', 'swvl1')} ({sm_region.swvl1.attrs.get('units', '')})",
        'shrink': 0.8,
        'pad': 0.05
    }
)

ax.set_title('Difference in average soil moisture (swvl2 - swvl1)\n1980-2023', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()


# %%
# Select a central gridcell for time series visualization
lat_idx = smdi.latitude.size // 2
lon_idx = smdi.longitude.size // 2

central_lat = smdi.latitude.values[lat_idx]
central_lon = smdi.longitude.values[lon_idx]

smdi_central_cell = smdi.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='valid_time', how='any')

# Create time series plot
fig, ax = plt.subplots(figsize=(14, 6))

# Plot SMDI time series for the central gridcell
smdi_central_cell.plot(ax=ax, color='steelblue', linewidth=0.5, alpha=0.7, label='SMDI (central gridcell)')

# Calculate and plot 3-month (90-day) running mean
smdi_running_mean = smdi_central_cell.rolling(valid_time=90, center=True).mean()
smdi_running_mean.plot(ax=ax, color='black', linewidth=1.5, alpha=0.8, label='3-month running mean')

# Add drought threshold lines
ax.axhline(y=-1.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Mild drought threshold')
ax.axhline(y=-2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Moderate drought threshold')
ax.axhline(y=-3.0, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='Severe drought threshold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# Mark known drought events
known_droughts = {
    '2003': ('2003-06-01', '2003-09-30'),
    '2018': ('2018-05-01', '2018-09-30'),
    '2022': ('2022-05-01', '2022-09-30')
}

for year, (start, end) in known_droughts.items():
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
               alpha=0.2, color='red', label=f'known drought' if year == '2003' else '')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('SMDI', fontsize=12)
ax.set_title('Daily SMDI & 3-month Running Mean - Central gridcell example\n1980-2023', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.show()

print("Time series plot created.")
#%% Save SMDI as netcdf
path_save = Path('./../Results/SMDI/')

year_o = str(int(smdi.valid_time.dt.year.isel(valid_time=0).item()))
year_f = str(int(smdi.valid_time.dt.year.isel(valid_time=-1).item()))
lat_o, lat_f = str(int(smdi.latitude.values[0])), str(int(smdi.latitude.values[-1]))
lon_o, lon_f = str(int(smdi.longitude.values[0])), str(int(smdi.longitude.values[-1]))

name = f'SMDI_{year_o}_{year_f}_latlon_{lat_o}_{lat_f}_{lon_o}_{lon_f}.nc'
save_toggle = input("\nSave SMDI results? (Y or N) \n ...")
if save_toggle.upper() == 'Y':
    smdi.to_netcdf(Path(path_save, name))
    print(f"SMDI saved to: {Path(path_save, name)}")

# %%
