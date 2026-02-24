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
from pathlib import Path
from cftime import sec_units
import xarray as xr
import dask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rioxarray  
#%% Load datasets: dewpoint temperature and air temperature
dat_path = Path('./../Data/ERA5_zarr/ERA5_NL.zarr')

ds = xr.open_zarr(
    dat_path,
    chunks={'time': 24*365}
)

# Load countries shapefile
global_countries_path = Path('./../Data/countries_shp/ne_110m_admin_0_countries.shp')

# Read countries shapefile
countries_gdf = gpd.read_file(str(global_countries_path))

# Create Netherlands shape
NL_shape = countries_gdf[countries_gdf['ADM0_A3'] == 'NLD']

# Set the spatial dimensions so rioxarray knows which axes are spatial
ds = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
ds = ds.rio.write_crs("EPSG:4326")  # ERA5 is in WGS84

# Clip to Netherlands shape, keeping all cells that intersect the border
ds_nl = ds.rio.clip(
    NL_shape.geometry,
    NL_shape.crs,
    all_touched=True   # include intersecting edge cells
)

# Get solar radiation, temperature, and dewpoint temperature data 
ssrd = ds_nl.ssrd
t2m  = ds_nl.t2m
d2m  = ds_nl.d2m
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
        Daily maximum VPD values (Pa); must have 'valid_time' (datetime) coordinate.
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
    import pandas as pd

    # Determine default baseline years if not specified
    years = pd.DatetimeIndex(vpd_daily["valid_time"].values).year
    baseline_start = baseline_start or str(years.min())
    baseline_end   = baseline_end   or str(years.max())

    # Select baseline period
    vpd_baseline = vpd_daily.sel(valid_time=slice(baseline_start, baseline_end))

    # Rechunk for efficient groupby
    vpd_baseline = vpd_baseline.chunk({'valid_time': -1, 'latitude': -1, 'longitude': -1})

    # Add day of year coordinate and compute DOY statistics
    vpd_baseline = vpd_baseline.assign_coords(doy=vpd_baseline.valid_time.dt.dayofyear)

    vpd_mean_doy = vpd_baseline.groupby("doy").mean("valid_time")
    vpd_std_doy  = vpd_baseline.groupby("doy").std("valid_time")

    # Prevent division by zero in SVDI calculation
    vpd_std_doy = xr.where(vpd_std_doy > 1e-10, vpd_std_doy, 1e-10)

    return vpd_mean_doy, vpd_std_doy


def calculate_svdi(vpd_daily, vpd_mean_doy, vpd_std_doy):
    """
    Calculate Standardized VPD Drought Index (SVDI) following Gamelin et al. (2022).

    Standardizes daily maximum VPD against DOY climatological mean and standard
    deviation computed over a baseline period.

    Parameters
    ----------
    vpd_daily : xarray.DataArray
        Daily maximum VPD values (Pa); must have 'valid_time' (datetime) coordinate.
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
    vpd_daily = vpd_daily.assign_coords(doy=vpd_daily.valid_time.dt.dayofyear)

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
        'daily_input'     : 'Daily maximum VPD computed from daytime hours only',
    })

    return svdi
    
# %% Calculate daily maximum temperature and minimum dewpoint temperature 
# for daytime hours only, and for maximum daily temperature and minimum daily dewpoint temperature

# Mask out hours with no solar radiation
daytime_mask = ssrd > 0


# Calculate daily maximum temperature and minimum dewpoint temperature (minimum relative humidity)
t2m_daily_max = (
    t2m.where(daytime_mask)
       .chunk({'valid_time': -1})         # contiguous time for resample
       .resample(valid_time='1D').max()
)
d2m_daily_min = (
    d2m.where(daytime_mask)
       .chunk({'valid_time': -1})
       .resample(valid_time='1D').min()
)

print("Calculating daily maximum temperature and minimum dew point...")
# this assumes maximum temperature and minimum dew point coincide 
# during hottest/driest time of day. 
t2m_daily_max, d2m_daily_min = dask.compute(t2m_daily_max, d2m_daily_min)

#%% Calculate daily maximum VPD 
print("Calculating daily maximum VPD...")
vpd_daily_max = calculate_vpd(t2m_daily_max, d2m_daily_min)

print(f"VPD hourly calculated. Shape: {vpd_daily_max.shape}")
print(f"VPD hourly range: {float(vpd_daily_max.min().values):.2f} to {float(vpd_daily_max.max().values):.2f} Pa")
print(f"VPD hourly mean: {float(vpd_daily_max.mean().values):.2f} Pa")

#%% Calculate DOY climatology for SVDI calculations

print(f"Calculating DOY VPD climatology...")
vpd_mean_doy, vpd_std_doy = calculate_doy_climatology(vpd_daily_max)

#%% Sample gridcell of VPD max results 
# 1. Box & Whisker plot and empirical distribution for central gridcell's vpd_max_daily

# Select a central sample gridcell
lat_idx = vpd_daily_max.latitude.size // 2
lon_idx = vpd_daily_max.longitude.size // 2

central_lat = vpd_daily_max.latitude.values[lat_idx]
central_lon = vpd_daily_max.longitude.values[lon_idx]

vpd_central = vpd_daily_max.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='valid_time', how='any')

central_vpd_values = vpd_central.values

fig, axs = plt.subplots(1, 2, figsize=(11,5))
fig.suptitle('VPD Daily Maximums: Central Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)), fontsize=14)

# Box and whisker plot
axs[0].boxplot(central_vpd_values, vert=True, patch_artist=True, showmeans=True)
axs[0].set_ylabel('VPDmax Daily (Pa)')
axs[0].set_title('Box & Whisker Plot')

# Empirical distribution (histogram with KDE)
axs[1].hist(central_vpd_values, bins=40, density=True, alpha=0.6, color='tab:blue', edgecolor="black", label='Histogram')
sns.kdeplot(central_vpd_values, ax=axs[1], color='tab:red', label='KDE')

axs[1].set_xlabel('VPDmax Daily (Pa)')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical Distribution')
axs[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(os.path.join(path_save, "vpd_max_daily_central_gridcell_distribution.png"), dpi=300)
plt.show()

# 2. Plot mean and standard deviation envelope timeseries for DOY climatology (mean year)

fig, ax = plt.subplots(figsize=(10,5))
days = vpd_mean_doy['doy']
mean = vpd_mean_doy.isel(latitude=lat_idx, longitude=lon_idx)
std = vpd_std_doy.isel(latitude=lat_idx, longitude=lon_idx)

mean_vals = mean.values
std_vals = std.values

ax.plot(days, mean_vals, label='Mean VPDmax', color='navy')
ax.fill_between(days, mean_vals-std_vals, mean_vals+std_vals, color='lightblue', alpha=0.5, label='Mean ± 1 Std Dev')

ax.set_xlabel('Day of Year')
ax.set_ylabel('VPDmax (Pa)')
ax.set_title('Mean Year: VPDmax Climatology (+/- 1 STD)\nCentral Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)))
ax.legend()
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig(os.path.join(path_save, "vpd_mean_std_envelope_central_gridcell.png"), dpi=300)
plt.show()


#%% Calculate SVDI
# SVDI calculated as in Gamelin et al. 2022; 2025. 

print("\nCalculating Standardized VPD Drought Index (SVDI)...")

svdi = calculate_svdi(vpd_daily_max, vpd_mean_doy, vpd_std_doy)

print(f"\nSVDI calculation complete!")
print(f"SVDI shape: {svdi.shape}")
print(f"SVDI range: {float(svdi.min().values):.2f} to {float(svdi.max().values):.2f}")
print(f"SVDI mean: {float(svdi.mean().values):.2f}")
print(f"SVDI std: {float(svdi.std().values):.2f}")


#%% Sample gridcell of SVDI results 

#%% Sample gridcell of VPD max results 
# 1. Box & Whisker plot and empirical distribution for central gridcell's vpd_max_daily

# Select a central sample gridcell
lat_idx = svdi.latitude.size // 2
lon_idx = svdi.longitude.size // 2

central_lat = svdi.latitude.values[lat_idx]
central_lon = svdi.longitude.values[lon_idx]

svdi_central = svdi.isel(latitude=lat_idx, longitude=lon_idx).dropna(dim='valid_time', how='any')

central_svdi_values = svdi_central.values

fig, axs = plt.subplots(1, 2, figsize=(11,5))
fig.suptitle('SVDI: Central Gridcell ({:.2f}N, {:.2f}E)'.format(float(central_lat), float(central_lon)), fontsize=14)

# Box and whisker plot
axs[0].boxplot(central_svdi_values, vert=True, patch_artist=True, showmeans=True)
axs[0].set_ylabel('SVDI (dimensionless)')
axs[0].set_title('Box & Whisker Plot')

# Empirical distribution (histogram with KDE)
axs[1].hist(central_svdi_values, bins=40, density=True, alpha=0.6, color='tab:blue', edgecolor="black", label='Histogram')
sns.kdeplot(central_svdi_values, ax=axs[1], color='tab:red', label='KDE')

axs[1].set_xlabel('SVDI (dimensionless)')
axs[1].set_ylabel('Density')
axs[1].set_title('Empirical Distribution')
axs[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(os.path.join(path_save, "vpd_max_daily_central_gridcell_distribution.png"), dpi=300)
plt.show()

#%% Save SVDI as netcdf
path_save = Path('./../Results/SVDI/')

year_o = str(int(svdi.valid_time.dt.year.isel(valid_time=0).item()))
year_f = str(int(svdi.valid_time.dt.year.isel(valid_time=-1).item()))
lat_o, lat_f = str(int(svdi.latitude.values[0])), str(int(svdi.latitude.values[-1]))
lon_o, lon_f = str(int(svdi.longitude.values[0])), str(int(svdi.longitude.values[-1]))

name = f'SVDI_{year_o}_{year_f}_latlon_{lat_o}_{lat_f}_{lon_o}_{lon_f}.nc'
save_toggle = input("\nSave SVDI results? (Y or N) \n ...")
if save_toggle.upper() == 'Y':
    svdi.to_netcdf(Path(path_save, name))
    print(f"SVDI saved to: {Path(path_save, name)}")

