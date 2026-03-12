#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Standardized VPD Drought Index (SVDI)

This script loads ERA5 temperature and dewpoint data, calculates VPD,
and computes the Standardized VPD Drought Index using DOY-based climatology,
following the method in Gamelin et al. 2025 https://doi.org/10.1007/s00382-025-07691-y

@author: afer
"""

#%%
# Imports
import os
import xarray as xr
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#%% 
# Load temperature and dewpoint temperature datasets 
glob_dat_path = Path(os.environ["ERA5_dat"])
folder = "temps"
dat_path = glob_dat_path / folder / "processed" / f"{folder}_cleaned.zarr"
 
ds = xr.open_zarr(dat_path, consolidated=True, chunks={})
 
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
ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
ds = ds.rio.write_crs("EPSG:4326")  # ERA5 is in WGS84
 
# Clip to region shape, keeping all cells that intersect the border
ds_clip = ds.rio.clip(
    region.geometry,
    region.crs,
    all_touched=True   # include intersecting edge cells
)

t2m = ds_clip.t2m
d2m = ds_clip.d2m
 

#%% Define functions

def calc_e(T):
    """
    Calculates saturated vapor pressure (e) in Pascals (Pa) 
    using the Tetens formula based on temperature in Celsius (°C).
    
    Parameters:
    -----------
    T : xarray.DataArray
        Temperature in Celsius
        
    Returns:
    --------
    e_pa : xarray.DataArray
        Saturated vapor pressure in Pascals
    """
    # Calculate saturated vapor pressure in millibars
    e_mb = 6.1078 * np.exp((17.269 * T) / (237.3 + T))
    
    # Convert millibars to Pascals (1 mb = 100 Pa)
    e_pa = e_mb * 100.0
    
    return e_pa

def calculate_vpd(t2m, d2m):
    """
    Calculate hourly VPD (Vapor Pressure Deficit) from ERA5 temperature and dewpoint data.
    
    Parameters:
    -----------
    t2m : xarray.DataArray
        Hourly 2m air temperature (Kelvin)
    d2m : xarray.DataArray
        Hourly 2m dewpoint temperature (Kelvin)
        
    Returns:
    --------
    vpd_hourly : xarray.DataArray
        Hourly VPD (Pa), aligned with input coordinates/times
    """
    # Convert temperatures to Celsius
    t_celsius = t2m - 273.15
    t_dpt_celsius = d2m - 273.15

    # Calculate saturated vapor pressure for air temperature and dewpoint
    e_sat = calc_e(t_celsius)  # Saturated vapor pressure at air temperature
    e_actual = calc_e(t_dpt_celsius)  # Actual vapor pressure at dewpoint temperature

    # Calculate VPD (Vapor Pressure Deficit) in Pa
    vpd_hourly = e_sat - e_actual

    return vpd_hourly


def calculate_doy_climatology(
    vpd_daily,
    baseline_start=None,
    baseline_end=None,
):
    """
    Calculate day-of-year (DOY) climatology statistics for daily maximum VPD.

    Parameters
    ----------
    vpd_daily : xarray.DataArray
        Daily maximum VPD values (Pa); must have 'time' (datetime) coordinate.
    baseline_start : str or None, optional
        Start year (YYYY) for climatology period. If None, uses minimum year in vpd_daily.
    baseline_end : str or None, optional
        End year (YYYY) for climatology period. If None, uses maximum year in vpd_daily.

    Returns
    -------
    vpd_mean_doy : xarray.DataArray
        Mean daily maximum VPD for each DOY across the baseline period.
    vpd_std_doy : xarray.DataArray
        Standard deviation of daily maximum VPD for each DOY across the baseline period.
    """
    # Determine default baseline years if not specified
    years = pd.DatetimeIndex(vpd_daily["time"].values).year
    baseline_start = baseline_start or str(years.min())
    baseline_end   = baseline_end   or str(years.max())

    # Select baseline period
    vpd_baseline = vpd_daily.sel(time=slice(baseline_start, baseline_end))

    # Rechunk for efficient groupby
    vpd_baseline = vpd_baseline.chunk({'time': -1, 'latitude': -1, 'longitude': -1})

    # Add day of year coordinate and compute DOY statistics
    vpd_baseline = vpd_baseline.assign_coords(doy=vpd_baseline.time.dt.dayofyear)

    vpd_mean_doy = vpd_baseline.groupby("doy").mean("time")
    vpd_std_doy  = vpd_baseline.groupby("doy").std("time")

    # Prevent division by zero in SVDI calculation
    vpd_std_doy = xr.where(
        vpd_std_doy.notnull() & (vpd_std_doy > 1e-10),
        vpd_std_doy,
        np.nan   
    )


    return vpd_mean_doy, vpd_std_doy


def calculate_svdi(vpd_daily, vpd_mean_doy, vpd_std_doy):
    """
    Calculate Standardized VPD Drought Index (SVDI) following Gamelin et al. (2022).

    Standardizes daily maximum VPD against DOY climatological mean and standard
    deviation computed over a baseline period.

    Parameters
    ----------
    vpd_daily : xarray.DataArray
        Daily maximum VPD values (Pa); must have 'time' (datetime) coordinate.
    vpd_mean_doy : xarray.DataArray
        Mean daily maximum VPD for each DOY (from calculate_doy_climatology).
    vpd_std_doy : xarray.DataArray
        Standard deviation of daily maximum VPD for each DOY (from calculate_doy_climatology).

    Returns
    -------
    svdi : xarray.DataArray
        Standardized VPD Drought Index (dimensionless). Positive values indicate
        higher-than-normal atmospheric demand; negative values indicate lower-than-normal.
    """
    # Map each date to its DOY
    vpd_daily = vpd_daily.assign_coords(doy=vpd_daily.time.dt.dayofyear)

    # Align climatology to the full time series by DOY
    vpd_mean_aligned = vpd_mean_doy.sel(doy=vpd_daily.doy)
    vpd_std_aligned  = vpd_std_doy.sel(doy=vpd_daily.doy)

    # Standardize: (observed - climatological mean) / climatological std
    svdi = (vpd_daily - vpd_mean_aligned) / vpd_std_aligned

    svdi.attrs.update({
        'long_name'       : 'Standardized VPD Drought Index',
        'units'           : 'dimensionless',
        'description'     : 'Standardized daily maximum VPD anomaly using DOY climatology',
        'reference'       : 'Gamelin et al. (2022)',
        'daily_input'     : 'Daily maximum VPD computed from maximum daily air and dewpoint temperature ',
    })

    return svdi
    

#%% Calculate daily maximum VPD 
print("Calculating daily maximum VPD...")

vpd_hourly = calculate_vpd(t2m, d2m)

# Assumption that VPD max happens during the hottest/driest hour in each day, which are assumed to match but don't necessarily
vpd_daily_max = (
    vpd_hourly
    .chunk({'time': 43_800, 'latitude': -1, 'longitude': -1})
    .resample(time='1D').max()
    .compute()
)

print(f"VPD hourly calculated. Shape: {vpd_daily_max.shape}")
print(f"VPD hourly range: {float(vpd_daily_max.min().values):.2f} to {float(vpd_daily_max.max().values):.2f} Pa")
print(f"VPD hourly mean: {float(vpd_daily_max.mean().values):.2f} Pa")

#%% Calculate DOY climatology for SVDI calculations

print(f"Calculating DOY VPD climatology...")
vpd_mean_doy, vpd_std_doy = calculate_doy_climatology(vpd_daily_max)

#%% 
# Plots for sample gridcell of VPD max  
# Box & Whisker plot and empirical distribution for central gridcell's vpd_max_daily

#central gridcell
lat_idx = vpd_daily_max.latitude.size // 2
lon_idx = vpd_daily_max.longitude.size // 2

central_lat = vpd_daily_max.latitude.values[lat_idx]
central_lon = vpd_daily_max.longitude.values[lon_idx]

# data in selected gridcell 
vpd_central = vpd_daily_max.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='time', how='any')
central_vpd_values = vpd_central.values

#box-and-whisker plot of VPD daily max values for gridcell
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('VPD Daily Max at: ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)), fontsize=14)
 
axs[0].boxplot(central_vpd_values, vert=True, patch_artist=True, showmeans=True)
axs[0].set_ylabel('VPDmax Daily (Pa)')
axs[0].set_title('Box & Whisker Plot')
 
# Empirical probability distribution and histogram of VPD max values for central gridcell
axs[1].hist(central_vpd_values, bins=40, density=True, alpha=0.6, color='tab:blue', edgecolor='black', label='Histogram')
sns.kdeplot(central_vpd_values, ax=axs[1], color='tab:red', label='KDE')
axs[1].set_xlabel('VPDmax Daily (Pa)')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical Distribution')
axs[1].legend()
 
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(Path(path_save, "vpd_max_daily_central_gridcell_distribution.png"), dpi=300)
plt.show()

#%%
# timeseries of VPD climatology 
import calendar
 
fig, ax = plt.subplots(figsize=(10, 5))
days     = vpd_mean_doy['doy']
mean     = vpd_mean_doy.isel(latitude=lat_idx, longitude=lon_idx)
std      = vpd_std_doy.isel(latitude=lat_idx, longitude=lon_idx)
 
mean_vals = mean.values
std_vals  = std.values
 
ax.plot(days, mean_vals, label='Mean VPDmax', color='navy')
ax.fill_between(days, mean_vals - std_vals, mean_vals + std_vals,
                color='lightblue', alpha=0.5, label='Mean ± 1 Std Dev')
 
month_starts = [pd.Timestamp(year=2001, month=i, day=1).dayofyear for i in range(1, 13)]
month_names  = [calendar.month_abbr[i] for i in range(1, 13)]
ax.set_xticks(month_starts)
ax.set_xticklabels(month_names)
 
ax.set_xlabel('Month')
ax.set_ylabel('VPDmax (Pa)')
ax.set_title('Mean Year: VPDmax Climatology (± 1 STD)\nCentral Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)))
ax.legend()
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig(Path(path_save, "vpd_mean_std_envelope_central_gridcell.png"), dpi=300)
plt.show()
#%%
# Calculate SVDI
# SVDI calculated as in Gamelin et al. 2022; 2025. 

print("\nCalculating Standardized VPD Drought Index (SVDI)...")

svdi = calculate_svdi(vpd_daily_max, vpd_mean_doy, vpd_std_doy).compute()

print(f"SVDI calculation complete!")
print(f"SVDI shape: {svdi.shape}")
print(f"SVDI range: {float(svdi.min(skipna=True).values):.2f} to {float(svdi.max(skipna=True).values):.2f}")
print(f"SVDI mean:  {float(svdi.mean(skipna=True).values):.2f}")
print(f"SVDI std:   {float(svdi.std(skipna=True).values):.2f}")

#%% Sample gridcell of SVDI results 
# 1. Box & Whisker plot and empirical distribution for central gridcell's vpd_max_daily

svdi_central = svdi.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='time', how='any')
central_svdi_values = svdi_central.values
 
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('SVDI: Central Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)), fontsize=14)
 
axs[0].boxplot(central_svdi_values, vert=True, patch_artist=True, showmeans=True)
axs[0].set_ylabel('SVDI (dimensionless)')
axs[0].set_title('Box & Whisker Plot')
 
axs[1].hist(central_svdi_values, bins=40, density=True, alpha=0.6, color='tab:blue', edgecolor='black', label='Histogram')
sns.kdeplot(central_svdi_values, ax=axs[1], color='tab:red', label='KDE')
axs[1].set_xlabel('SVDI (dimensionless)')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical Distribution')
axs[1].legend()
 
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(Path(path_save, "svdi_central_gridcell_distribution.png"), dpi=300)
plt.show()

# SVDI time series with running mean and drought thresholds
fig, ax = plt.subplots(figsize=(14, 6))
 
svdi_central.plot(ax=ax, color='steelblue', linewidth=0.5, alpha=0.7, label='SVDI (central gridcell)')
 
svdi_running_mean = svdi_central.rolling(time=90, center=True).mean()
svdi_running_mean.plot(ax=ax, color='black', linewidth=1.5, alpha=0.8, label='3-month running mean')
 
ax.axhline(y=1.0,  color='orange',  linestyle='--', linewidth=1, alpha=0.7, label='Mild stress threshold')
ax.axhline(y=2.0,  color='red',     linestyle='--', linewidth=1, alpha=0.7, label='Moderate stress threshold')
ax.axhline(y=3.0,  color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='Severe stress threshold')
ax.axhline(y=0,    color='black',   linestyle='-',  linewidth=0.5, alpha=0.3)
 
known_droughts = {
    '2003': ('2003-06-01', '2003-09-30'),
    '2018': ('2018-05-01', '2018-09-30'),
    '2022': ('2022-05-01', '2022-09-30'),
}
for year, (start, end) in known_droughts.items():
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
               alpha=0.2, color='red', label='Known drought' if year == '2003' else '')
 
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('SVDI', fontsize=12)
ax.set_title('Daily SVDI & 3-month Running Mean - Central gridcell example\n1980-2023',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.show()


#%%
# Save SVDI as Zarr
 
path_save = Path('./../Results/SVDI')
path_save.mkdir(parents=True, exist_ok=True)
 
svdi.attrs.update({
    "long_name"       : "Standardized VPD Drought Index",
    "baseline_period" : f"{svdi.time.dt.year.values[0]}-{svdi.time.dt.year.values[-1]}",
    "source"          : "ERA5 t2m/d2m via ARCO-ERA5",
    "reference"       : "Gamelin et al. (2025)",
    "created"         : pd.Timestamp.now(tz="UTC").isoformat(),
})
 
zarr_path = path_save / "svdi.zarr"
svdi.to_zarr(zarr_path, mode="w", consolidated=True)
print(f"SVDI saved to: {zarr_path}")
