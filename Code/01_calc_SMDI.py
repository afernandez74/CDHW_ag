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
# imports

import os
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
#%% 
# Load soil moisture dataset and crop to shape


glob_dat_path = Path(os.environ["ERA5_dat"])
folder = "swv"
dat_path = glob_dat_path / folder / "processed" / f"{folder}_cleaned.zarr"


sm = ( #soil moisture dataset
    xr.open_zarr(dat_path, consolidated=True, chunks={})
)

# save path for figures
path_save = Path("./../Figs/NAC_PPT/")



#%%
# Perform spatial clip to area of interest
# Countries in Analysis: 

countries = ['France', 'Belgium', 'Netherlands', 'Germany']

# Load countries shapefile
global_countries_path = Path('./../Data/countries_shp/ne_110m_admin_0_countries.shp')

# Read countries shapefile
countries_gdf = gpd.read_file(str(global_countries_path))

# Define region
region = countries_gdf[countries_gdf['SOVEREIGNT'].isin(countries)].dissolve()  

# Set the spatial dimensions so rioxarray knows which axes are spatial
sm = sm.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
sm = sm.rio.write_crs("EPSG:4326")  # ERA5 is in WGS84

# Clip to shape, keeping all cells that intersect the border
sm_clip = sm.rio.clip(
    region.geometry,
    region.crs,
    all_touched=True  
)

#%% define functions
def calculate_sm_doy_climatology(sm_daily, baseline_start=None, baseline_end=None):
    """
    Compute DOY climatological median, min, and max of daily soil moisture.
    These are the inputs required for the SD calculation in SMDI.
    """
    years = pd.DatetimeIndex(sm_daily['time'].values).year
    baseline_start = baseline_start or str(years.min())
    baseline_end   = baseline_end   or str(years.max())

    sm_baseline = (
        sm_daily
        .sel(time=slice(baseline_start, baseline_end))
        .chunk({'time': -1, 'latitude': -1, 'longitude': -1})
        .assign_coords(doy=lambda x: x.time.dt.dayofyear)
    )

    sm_median_doy = sm_baseline.groupby('doy').median('time')
    sm_min_doy    = sm_baseline.groupby('doy').min('time')
    sm_max_doy    = sm_baseline.groupby('doy').max('time')

    return sm_median_doy, sm_min_doy, sm_max_doy

def calculate_sd(sm_daily, sm_median_doy, sm_min_doy, sm_max_doy):
    """
    Calculate daily Soil water Deficit/surplus (SD) scaled to [-100, +100].
    """
    sm_daily = sm_daily.assign_coords(doy=sm_daily.time.dt.dayofyear)

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
    (e.g. grid cells masked by the Netherlands boundary), the previous SMDI
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
        must have 'time', 'latitude', 'longitude' dimensions.
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
            'units'       : 'dimensionless',
            'description' : 'Recursive soil moisture drought index scaled from daily soil water volume ERA5 data',
            'reference'   : 'Narasimhan & Srinivasan (2005), Guo et al. (2023)',
            'daily_input' : 'Daily mean soil moisture from depth-weighted average of ERA5 swvl1 (0–7 cm) and swvl2 (7–28 cm)',
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

sm_clip_wavg = (sm_clip.swvl1 * 0.25 + sm_clip.swvl2 * 0.75)
# sm_clip_wavg_dly = sm_clip_wavg.resample(time='1D').mean().compute()
# sm_clip_wavg_dly = sm_clip_wavg_dly.chunk({'time': 365, 'latitude': -1, 'longitude': -1})
sm_clip_wavg_dly = (
    sm_clip_wavg
    .chunk({'time': 43_800, 'latitude': -1, 'longitude': -1})
    .resample(time='1D')
    .mean()
    .compute()   
)




#%% Sample gridcell of SM daily avg results 
# 1. Box & Whisker plot and empirical distribution for central gridcell's sm_daily_avg

# Select a central sample gridcell
lat_idx = sm_clip_wavg_dly.latitude.size // 2
lon_idx = sm_clip_wavg_dly.longitude.size // 2

central_lat = sm_clip_wavg_dly.latitude.values[lat_idx]
central_lon = sm_clip_wavg_dly.longitude.values[lon_idx]

sm_central = sm_clip_wavg_dly.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='time', how='any')

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




#%% Compute Soil Moisture Deficit Index (SMDI)
# Calculate daily Soil Moisture Deficit Index (SMDI) from ERA5 soil moisture data for the Netherlands.
# Algorithm: Based on Narasimhan and Srinivasan (2005), adapted for daily resolution (Guo et al. 2023).
# Baseline period: 1980-2023.
# SMDI calculated from weighted average soil moisture from two top soil layers

sm_median_doy, sm_min_doy, sm_max_doy = calculate_sm_doy_climatology(
    sm_clip_wavg_dly,
)

# Step 2: Soil water Deficit/Surplus [-100, 100]
sd = calculate_sd(sm_clip_wavg_dly, sm_median_doy, sm_min_doy, sm_max_doy)

# Step 3: Recursive SMDI
smdi = calculate_smdi(sd, alpha=0.5)

#%% Sample plot of SMDI results

# 1. Box & Whisker plot and empirical distribution for central gridcell's vpd_max_daily

# Select a central sample gridcell
lat_idx = smdi.latitude.size // 2
lon_idx = smdi.longitude.size // 2

central_lat = smdi.latitude.values[lat_idx]
central_lon = smdi.longitude.values[lon_idx]

smdi_central = smdi.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='time', how='any')

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
# plt.savefig(Path(save_path, "sm_mean_std_envelope_central_gridcell.png"), dpi=300)
plt.show()


#%% 
# Mmap of weighted average soil moisture
# Map plot with LambertConformal projection

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
    central_longitude=5,
    central_latitude=40
))

# set spatial extent for map based on data
lon_min = sm_clip.longitude.min().values
lon_max = sm_clip.longitude.max().values
lat_min = sm_clip.latitude.min().values
lat_max = sm_clip.latitude.max().values
extent = [lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# add coastlines and country borders
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle='--')
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)

# gridlines
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
gl.top_labels = False
gl.right_labels = False

# the actual map
im = sm_clip_wavg_dly.mean(dim='time').plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='YlGnBu',
    cbar_kwargs={
        'label': f"Average {sm_clip.swvl1.attrs.get('long_name', 'swvl1')} ({sm_clip.swvl1.attrs.get('units', '')})",
        'shrink': 0.8,
        'pad': 0.05
    }
)

ax.set_title('Average soil moisture (swvl2 + swvl1)\n1980-2023', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()


# %%
# Select a central gridcell for time series visualization
# lat_idx = smdi.latitude.size // 2
# lon_idx = smdi.longitude.size // 2

# central_lat = smdi.latitude.values[lat_idx]
# central_lon = smdi.longitude.values[lon_idx]
central_lat = 52
central_lon = 5
# smdi_central_cell = smdi.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='time', how='any')
smdi_central_cell = smdi.sel(latitude=central_lat, longitude=central_lon).dropna(dim='time', how='any')

# Create time series plot
fig, ax = plt.subplots(figsize=(14, 6))

# Plot SMDI time series for the central gridcell
smdi_central_cell.plot(ax=ax, color='steelblue', linewidth=1.0, alpha=0.7, label='SMDI (central gridcell)')

# Calculate and plot 3-month (90-day) running mean
smdi_running_mean = smdi_central_cell.rolling(time=90, center=True).mean()
smdi_running_mean.plot(ax=ax, color='black', linewidth=2.0, alpha=0.8, label='3-month running mean')

# Add drought threshold lines
ax.axhline(y=-1.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Mild drought threshold')
ax.axhline(y=-2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Moderate drought threshold')
ax.axhline(y=-3.0, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='Severe drought threshold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# # Mark known drought events
# known_droughts = {
#     '2003': ('2003-06-01', '2003-09-30'),
#     '2018': ('2018-05-01', '2018-09-30'),
#     '2022': ('2022-05-01', '2022-09-30')
# }
# Mark known drought for ppt
known_droughts = {
    '2018': ('2018-06-01', '2018-08-30'),
    '2020': ('2020-06-01', '2020-08-30'),    
    '2022': ('2022-06-01', '2022-08-30')
}

for year, (start, end) in known_droughts.items():
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
               alpha=0.2, color='red', label=f'known drought' if year == '2003' else '')

ax.set_xlim(pd.to_datetime('2015-01-01'),pd.to_datetime('2024-01-01'))


ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('SMDI', fontsize=16)
ax.set_title('Daily SMDI & 3-month Running Mean - Utrech, NL\n2015-2024', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)
ax.legend(loc='lower left', fontsize=12)
plt.tight_layout()
plt.show()

print("Time series plot created.")

#%%
# ============================================================
# 3-Panel Spatial Map: SMDI Agricultural Drought Characterisation
#   (a) Mean SMDI — JJA potato bulking stage (Jun–Aug)
#   (b) Linear trend in JJA SMDI (yr⁻¹)
#   (c) Mean length of sustained drought spells during JJA
#       (consecutive days with SMDI < -1.0)
# ============================================================
#
# Note on sign convention: SMDI drought thresholds are NEGATIVE
#   SMDI < -1  →  mild drought
#   SMDI < -2  →  moderate drought
#   SMDI < -3  →  severe drought
#
# Note on memory: SMDI is recursive (α=0.5), so sustained negative
# values represent genuinely accumulated soil water deficits.
# Spell length is therefore more physically meaningful here than
# for a memoryless index, and spatially highlights regions where
# the soil stays depleted through the bulking window.
#
# Commented alternatives for panel (a):
#   - Spring preconditioning: smdi_amj = smdi.sel(time=smdi.time.dt.month.isin([4,5,6]))
#     (Trnka et al. 2015, Nature Climate Change — late-spring SM deficit
#      predicts agricultural drought better than summer average alone)
#   - 10th percentile of JJA SMDI: worst-year spatial footprint
#     smdi_panel_a = smdi_jja.quantile(0.10, dim='time').compute()
#   - Cumulative drought stress (∑ SMDI < -1 during JJA, normalised by years):
#     maps directly onto FAO crop water-stress integrals (Steduto et al. 2012)
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from scipy import stats

drought_threshold = -1.0   # mild drought; change to -2.0 for moderate

# ── 1. Mean JJA SMDI (potato bulking stage) ──────────────────────────────────
smdi_jja      = smdi.sel(time=smdi.time.dt.month.isin([6, 7, 8]))
smdi_panel_a  = smdi_jja.mean(dim='time').compute()
panel_a_title = '(a)  Mean SMDI — Potato Bulking Stage\n(June–August)'
panel_a_cbar  = 'Mean SMDI — JJA (dimensionless)'

# ── 2. Linear Trend of JJA SMDI (yr⁻¹, per grid cell) ───────────────────────
#
# Trend is computed on annual JJA means to remove the within-season
# autocorrelation introduced by the recursive SMDI formula. Using annual
# summaries gives one value per year per cell, making OLS residuals
# more independent than fitting to the raw daily series.

smdi_jja_annual = smdi_jja.resample(time='YE').mean().compute()
years_numeric   = smdi_jja_annual.time.dt.year.values.astype(float)

def ols_slope_peryear_smdi(y):
    mask = np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    slope, _, _, _, _ = stats.linregress(years_numeric[mask], y[mask])
    return float(slope)   # already yr⁻¹ (input is annual means)

smdi_trend = xr.apply_ufunc(
    ols_slope_peryear_smdi,
    smdi_jja_annual,
    input_core_dims=[['time']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float],
).compute()

# ── 3. Mean drought spell length during JJA ──────────────────────────────────
#
# For each year, find runs of consecutive JJA days where SMDI < threshold,
# then average the spell lengths spatially. The result is: "on average, when
# a drought episode starts during potato bulking, how many days does it last?"
# A 14-day spell is agronomically far more damaging than 14 isolated days
# because soil moisture cannot recharge fast enough to prevent yield loss.

def mean_spell_length_jja(smdi_da, threshold=-1.0):
    """
    For each grid cell, compute the mean length (days) of consecutive
    JJA periods where SMDI < threshold, averaged across all years.

    Parameters
    ----------
    smdi_da : xr.DataArray  — full SMDI time series (time, lat, lon)
    threshold : float       — drought onset threshold (default -1.0)

    Returns
    -------
    mean_spell : xr.DataArray  — (lat, lon) mean JJA spell length (days)
    """
    # Work on JJA subset only
    da_jja = smdi_da.sel(time=smdi_da.time.dt.month.isin([6, 7, 8])).compute()

    in_drought = (da_jja < threshold).values   # bool (time, lat, lon)
    n_time, n_lat, n_lon = in_drought.shape
    mean_spell_arr = np.full((n_lat, n_lon), np.nan)

    for i in range(n_lat):
        for j in range(n_lon):
            ts = in_drought[:, i, j].astype(float)

            # Skip masked / all-NaN cells
            if np.all(~np.isfinite(da_jja.values[:, i, j])):
                continue

            # Find run-length-encoded spell lengths
            spell_lengths = []
            count = 0
            for t in range(n_time):
                if ts[t] == 1:
                    count += 1
                else:
                    if count > 0:
                        spell_lengths.append(count)
                    count = 0
            if count > 0:                        # close any open spell
                spell_lengths.append(count)

            mean_spell_arr[i, j] = (
                np.mean(spell_lengths) if spell_lengths else 0.0
            )

    mean_spell = xr.DataArray(
        mean_spell_arr,
        coords={'latitude': da_jja.latitude, 'longitude': da_jja.longitude},
        dims=['latitude', 'longitude'],
        attrs={'long_name': f'Mean JJA drought spell length (SMDI < {threshold})',
               'units': 'days'}
    )
    return mean_spell

print("Computing mean JJA drought spell lengths (this may take ~30 s)...")
smdi_spell_len = mean_spell_length_jja(smdi, threshold=drought_threshold)
print(f"  Spell length range: "
      f"{float(smdi_spell_len.min().values):.1f} – "
      f"{float(smdi_spell_len.max().values):.1f} days")

# ── Shared map utilities ──────────────────────────────────────────────────────
lats   = smdi.latitude.values
lons   = smdi.longitude.values
extent = [lons.min() - 0.5, lons.max() + 0.5,
          lats.min() - 0.5, lats.max() + 0.5]
proj   = ccrs.PlateCarree()

def style_axis(ax, title, label_left=True, label_bottom=True):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND,      facecolor='#f0ede8', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d6e8f5', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#444444', zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6, edgecolor='#666666',
                   linestyle='--', zorder=2)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor='#7ab8d4',
                   alpha=0.5, zorder=2)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=0.4, color='grey', alpha=0.5, linestyle=':')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = label_left
    gl.bottom_labels = label_bottom
    gl.xlocator      = mticker.MultipleLocator(5)
    gl.ylocator      = mticker.MultipleLocator(3)
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {'size': 8}
    gl.ylabel_style  = {'size': 8}
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

def add_colorbar(fig, ax, im, label, extend='both'):
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.04, fraction=0.046, aspect=28, extend=extend)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar

def percentile_norm(data_2d, n_breaks=10, cmap=plt.cm.YlOrRd):
    """
    Build a BoundaryNorm from data-percentile-spaced breakpoints so the
    full colormap range maps onto the actual data spread (same approach
    as SVDI panel c — ensures spatial variability is visible regardless
    of the absolute data range).
    """
    valid = data_2d[np.isfinite(data_2d)]
    boundaries = np.unique(
        np.round(np.nanpercentile(valid, np.linspace(0, 100, n_breaks + 1)), 2)
    )
    if len(boundaries) < 3:
        boundaries = np.linspace(valid.min(), valid.max(), n_breaks + 1)
    return mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N), boundaries

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 7),
    subplot_kw={'projection': proj},
    gridspec_kw={'wspace': 0.12}
)

# ── Panel (a): Mean JJA SMDI ──────────────────────────────────────────────────
mean_vals = smdi_panel_a.values

# Use np.fmax instead of built-in max to avoid shadowing by earlier sm_min/sm_max
# variable assignments in the climatology diagnostic cell
vlim_a = np.fmax(abs(np.nanpercentile(mean_vals, 2)),
                 abs(np.nanpercentile(mean_vals, 98)))

im1 = axes[0].pcolormesh(
    lons, lats, mean_vals,
    cmap='RdBu',                       # blue = wet, red = dry (intuitive for SM)
    vmin=-vlim_a, vmax=vlim_a,
    transform=proj, shading='auto', zorder=1
)
style_axis(axes[0], panel_a_title, label_left=True, label_bottom=True)
add_colorbar(fig, axes[0], im1, panel_a_cbar)

# # Zero contour to mark the wet/dry boundary
# axes[0].contour(
#     lons, lats, mean_vals,
#     levels=[0], colors='black', linewidths=0.8,
#     linestyles='-', transform=proj, zorder=3
# )

# ── Panel (b): Linear Trend in JJA SMDI ──────────────────────────────────────
trend_vals = smdi_trend.values
vlim_b = np.fmax(abs(np.nanpercentile(trend_vals, 2)),
                 abs(np.nanpercentile(trend_vals, 98)))

im2 = axes[1].pcolormesh(
    lons, lats, trend_vals,
    cmap='RdBu',
    vmin=-vlim_b, vmax=vlim_b,
    transform=proj, shading='auto', zorder=1
)
style_axis(axes[1], '(b)  Linear Trend in JJA SMDI\n(yr⁻¹, from annual JJA means)',
           label_left=False, label_bottom=True)
add_colorbar(fig, axes[1], im2, 'SMDI trend  (yr⁻¹)')
axes[1].contour(
    lons, lats, trend_vals,
    levels=[0], colors='black', linewidths=0.8,
    linestyles='-', transform=proj, zorder=3
)

# ── Panel (c): Mean drought spell length — percentile-adaptive colormap ───────
spell_vals = smdi_spell_len.values
cmap_spell = plt.cm.YlOrRd
norm_spell, boundaries_c = percentile_norm(spell_vals, n_breaks=10, cmap=cmap_spell)

im3 = axes[2].pcolormesh(
    lons, lats, spell_vals,
    cmap=cmap_spell, norm=norm_spell,
    transform=proj, shading='auto', zorder=1
)
style_axis(axes[2],
           f'(c)  Mean JJA Drought Spell Length\n'
           f'(consecutive days SMDI < {drought_threshold:.0f})',
           label_left=False, label_bottom=True)
cbar3 = add_colorbar(fig, axes[2], im3, 'Mean spell length (days)', extend='neither')
# cbar3.ax.set_title('percentile-spaced\ncolor breaks', fontsize=7,
#                    color='#555555', pad=4)

print(f"Panel (c) colour boundaries (data-percentile-spaced, days):")
print(np.array2string(boundaries_c, precision=1, separator=', '))

# ── Shared title ──────────────────────────────────────────────────────────────
year_start = int(smdi.time.dt.year.values[0])
year_end   = int(smdi.time.dt.year.values[-1])

fig.suptitle(
    f'Soil Moisture Deficit Index (SMDI) — Agricultural Drought Characterisation\n'
    f'France · Belgium · Netherlands · Germany  |  {year_start}–{year_end}',
    fontsize=14, fontweight='bold', y=1.01
)

# plt.savefig(
#     Path(path_save, "smdi_spatial_3panel.png"),
#     dpi=300, bbox_inches='tight', facecolor='white'
# )
plt.show()
print("Figure saved.")

#%% Save SMDI as Zarr
path_save = Path("./../Results/SMDI")
path_save.mkdir(parents=True, exist_ok=True)

# Attach metadata so the file is self-describing
smdi.attrs.update({
    "long_name"       : "Soil Moisture Deficit Index",
    "alpha"           : 0.5,
    "baseline_period": f"{smdi.time.dt.year.values[0]}-{smdi.time.dt.year.values[-1]}",
    "source"          : "ERA5 swvl1/swvl2 via ARCO-ERA5",
    "created"         : pd.Timestamp.now(tz="UTC").isoformat(),
})

zarr_path = path_save / "smdi.zarr"
smdi.to_zarr(zarr_path, mode="w", consolidated=True)
print(f"SMDI saved to: {zarr_path}")
# %%
