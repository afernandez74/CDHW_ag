#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Standardized VPD Drought Index (SVDI)

This script loads ERA5 temperature and dewpoint data, calculates VPD,
and computes the Standardized VPD Drought Index using DOY-based climatology.

@author: afer
"""

#%%
import os
import xarray as xr
import numpy as np

#%% Load datasets: dewpoint temperature and air temperature
dat_path = os.path.expanduser('./../Data/ERA5/')

files = os.listdir(dat_path)

# Dewpoint data file name
dpt_file_name = 'ERA5_hrly_vars_d2m_time_1980_2023_latlon_50_54_3_8.nc'
file_path = os.path.join(dat_path, dpt_file_name)
dpt_ds = xr.open_mfdataset(file_path)

# Temperature and solar radiation data file name
t_ssrd_file_name = 'ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8'
file_path = os.path.join(dat_path, t_ssrd_file_name)
t_ssrd_ds = xr.open_mfdataset(file_path,
                              drop_variables=['u100','v100']
                              )

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

def calculate_vpd_daily_daytime(t2m, d2m, ssrd):
    """
    Calculate daily daytime average VPD from hourly ERA5 data.
    
    Parameters:
    -----------
    t2m : xarray.DataArray
        Hourly 2m air temperature (Kelvin)
    d2m : xarray.DataArray
        Hourly 2m dewpoint temperature (Kelvin)
    ssrd : xarray.DataArray
        Hourly surface solar radiation downward (J/m2)
        
    Returns:
    --------
    vpd_daily : xarray.DataArray
        Daily daytime average VPD (Pa)
    """
    # Convert temperatures to Celsius
    t_celsius = t2m - 273.15
    t_dpt_celsius = d2m - 273.15
    
    # Calculate saturated vapor pressure for air temperature and dewpoint
    e_sat = calc_e(t_celsius)  # Saturated vapor pressure at air temperature
    e_actual = calc_e(t_dpt_celsius)  # Actual vapor pressure at dewpoint temperature
    
    # Calculate VPD (Vapor Pressure Deficit) in Pa
    vpd_hourly = e_sat - e_actual
    
    # Align the datasets (ensures matching time coordinates)
    vpd_aligned, ssrd_aligned = xr.align(vpd_hourly, ssrd, join='inner')
    
    # Create daytime mask (hours with positive solar radiation)
    daytime_mask = ssrd_aligned > 0
    
    # Calculate daily daytime average VPD
    vpd_daily = vpd_aligned.where(daytime_mask).resample(valid_time='1D').mean()
    
    return vpd_daily

def calculate_doy_climatology(vpd_daily, baseline_start='1980', baseline_end='2023'):
    """
    Calculate day-of-year (DOY) climatology statistics for VPD.
    
    Parameters:
    -----------
    vpd_daily : xarray.DataArray
        Daily VPD values
    baseline_start : str
        Start year for climatology period
    baseline_end : str
        End year for climatology period
        
    Returns:
    --------
    vpd_mean_doy : xarray.DataArray
        Mean VPD for each DOY
    vpd_std_doy : xarray.DataArray
        Standard deviation of VPD for each DOY
    """
    # Select baseline period
    vpd_baseline = vpd_daily.sel(valid_time=slice(baseline_start, baseline_end))
    
    # Add day of year coordinate
    vpd_baseline = vpd_baseline.assign_coords(doy=vpd_baseline.valid_time.dt.dayofyear)
    
    # Group by day of year and calculate statistics
    vpd_mean_doy = vpd_baseline.groupby('doy').mean('valid_time')
    vpd_std_doy = vpd_baseline.groupby('doy').std('valid_time')
    
    # Handle zero standard deviation (replace with small value to avoid division by zero)
    vpd_std_doy = xr.where(vpd_std_doy > 1e-10, vpd_std_doy, 1e-10)
    
    return vpd_mean_doy, vpd_std_doy

def calculate_svdi(vpd_daily, vpd_mean_doy, vpd_std_doy):
    """
    Calculate Standardized VPD Drought Index (SVDI).
    
    SVDI is calculated by standardizing daily VPD values using DOY-based
    climatological mean and standard deviation.
    
    Parameters:
    -----------
    vpd_daily : xarray.DataArray
        Daily VPD values
    vpd_mean_doy : xarray.DataArray
        Mean VPD for each DOY
    vpd_std_doy : xarray.DataArray
        Standard deviation of VPD for each DOY
        
    Returns:
    --------
    svdi : xarray.DataArray
        Standardized VPD Drought Index
    """
    # Add day of year coordinate to daily VPD
    vpd_daily = vpd_daily.assign_coords(doy=vpd_daily.valid_time.dt.dayofyear)
    
    # Select corresponding DOY statistics for each day
    vpd_mean_aligned = vpd_mean_doy.sel(doy=vpd_daily.doy)
    vpd_std_aligned = vpd_std_doy.sel(doy=vpd_daily.doy)
    
    # Calculate standardized anomaly: (VPD - mean) / std
    svdi = (vpd_daily - vpd_mean_aligned) / vpd_std_aligned
    
    # Add metadata
    svdi.attrs['long_name'] = 'Standardized VPD Drought Index'
    svdi.attrs['units'] = 'dimensionless'
    svdi.attrs['description'] = 'Standardized VPD anomaly using DOY-based climatology'
    svdi.attrs['baseline_period'] = '1980-2023'
    
    return svdi

#%% Calculate daily daytime average VPD

print("Calculating daily daytime average VPD...")

# Obtain data arrays
t2m = t_ssrd_ds.t2m
d2m = dpt_ds.d2m
ssrd = t_ssrd_ds.ssrd

# Calculate daily daytime average VPD
vpd_daily = calculate_vpd_daily_daytime(t2m, d2m, ssrd)

print(f"VPD daily calculated. Shape: {vpd_daily.shape}")
print(f"VPD daily range: {float(vpd_daily.min().values):.2f} to {float(vpd_daily.max().values):.2f} Pa")
print(f"VPD daily mean: {float(vpd_daily.mean().values):.2f} Pa")

#%% Calculate DOY climatology

print("\nCalculating DOY-based climatology...")

vpd_mean_doy, vpd_std_doy = calculate_doy_climatology(vpd_daily)

print(f"DOY climatology calculated for {len(vpd_mean_doy.doy)} days")
print(f"Mean VPD range: {float(vpd_mean_doy.min().values):.2f} to {float(vpd_mean_doy.max().values):.2f} Pa")
print(f"Std VPD range: {float(vpd_std_doy.min().values):.2f} to {float(vpd_std_doy.max().values):.2f} Pa")

#%% Calculate SVDI

print("\nCalculating Standardized VPD Drought Index (SVDI)...")

svdi = calculate_svdi(vpd_daily, vpd_mean_doy, vpd_std_doy)

print(f"\nSVDI calculation complete!")
print(f"SVDI shape: {svdi.shape}")
print(f"SVDI range: {float(svdi.min().values):.2f} to {float(svdi.max().values):.2f}")
print(f"SVDI mean: {float(svdi.mean().values):.2f}")
print(f"SVDI std: {float(svdi.std().values):.2f}")

#%% Save SVDI as netcdf

path_save = os.path.expanduser('./../Results/SVDI/')
os.makedirs(path_save, exist_ok=True)

name = 'SVDI_time_1980_2023_latlon_50_54_3_8.nc'
save_toggle = input("\nSave SVDI results? (Y or N) \n ...")
if save_toggle.upper() == 'Y':
    svdi.to_netcdf(os.path.join(path_save, name))
    print(f"SVDI saved to: {os.path.join(path_save, name)}")

#%% Sample time series plot (optional)

import matplotlib.pyplot as plt

print("\nGenerating sample time series plot...")

svdi.isel(latitude=0, longitude=0).sel(valid_time=slice('2017-06-01', '2018-06-05')).plot()

plt.xlabel("Date")
plt.ylabel("SVDI (dimensionless)")
plt.title("Standardized VPD Drought Index\nat a single gridcell for 2017-2018")
plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.tick_params(axis='x', rotation=45)

# Optional: Save the figure
plt.savefig(os.path.join(path_save, "svdi_timeseries_sample.png"), dpi=300)
plt.show()

print("Done!")
