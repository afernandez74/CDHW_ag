#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 09:50:25 2025

@author: afer
"""
#%%
import os
import xarray as xr
import math
import numpy as np
import matplotlib.pyplot as plt
#%% load 2 datasets: dewpoint temperature and air temperature
dat_path = os.path.expanduser('./../Data/ERA5/')

files = os.listdir(dat_path)

# dewpoint data file name, created from read_downloads_save_dataset.py
dpt_file_name = 'ERA5_hrly_vars_d2m_time_1980_2023_latlon_50_54_3_8.nc'

file_path = dat_path + dpt_file_name
dpt_ds = xr.open_mfdataset(file_path)

t_ssrd_file_name = 'ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8'
file_path = dat_path + t_ssrd_file_name
t_ssrd_ds = xr.open_mfdataset(file_path,
                              drop_variables =['u100','v100']
                              )

#%% define constants
# from MODIS GPP paper, for crops: 

E_max = 0.001044
t_min_min = -8 #deg C
t_min_max = 12.02 #deg C
VPD_min = 650 #Pa
VPD_max = 4300 #Pa

#%% define functions 
def calc_t_min_scalar(t_min, t_min_min, t_min_max):
    """
    Calculates the TMIN_scalar attenuation factor based on the MODIS GPP algorithm.

    Inputs:
    t_min (xr.DataArray): Daily minimum temperature (°C).
    t_min_min (float): Biome-specific minimum temperature threshold where efficiency is zero (TMINmin, °C).
    t_min_max (float): Biome-specific minimum temperature threshold where efficiency is optimal (TMINmax, °C).

    Returns:
    xr.DataArray: TMIN_scalar (ranging from 0.0 to 1.0).
    """
        
    # Calculate the linear scalar
    scalar = (t_min - t_min_min) / (t_min_max - t_min_min)
    
    result = xr.where(t_min <= t_min_min, 0.0,
                      xr.where(t_min >= t_min_max, 1.0, scalar))
    
    return result

def calc_e(T):
    """
    Calculates saturated vapor pressure (e) in Pascals (Pa) 
    using the Tetens formula based on temperature in Celsius (°C).
    
    e(mb) = 6.1078 * exp((17.269 * T) / (237.3 + T))
    """    
    # Calculate saturated vapor pressure in millibars [1]
    e_mb = 6.1078 * np.exp((17.269 * T) / (237.3 + T))
    
    # Convert millibars to Pascals (1 mb = 100 Pa)
    e_pa = e_mb * 100.0
    
    return e_pa

def calc_vpd_scalar(vpd_da, vpd_min, vpd_max):
    """
    Calculates the VPD_scalar attenuation factor based on the MODIS GPP algorithm logic.

    The VPD_scalar is a simple linear ramp function that is:
    1.0 when VPD <= VPD_min (optimal efficiency)
    0.0 when VPD >= VPD_max (efficiency ceases)
    Decreases linearly between VPD_min and VPD_max.

    Inputs:
    vpd_da (xr.DataArray): Daily daytime average VPD values [Pa]
    vpd_min (float or array-like): Biome-specific VPD threshold where efficiency is 
                                 optimal (VPDmin, Pa).
    vpd_max (float or array-like): Biome-specific VPD threshold where efficiency is 
                                 zero (VPDmax, Pa).

    Returns:
    xr.DataArray: VPD_scalar (ranging from 0.0 to 1.0) with the same dimensions 
                  as the input VPD array.
    """
    
    # 1. Calculate the range of VPD over which attenuation occurs.
    vpd_range = vpd_max - vpd_min
    
    # Ensure VPD_max is strictly greater than VPD_min to avoid division by zero 
    # and to maintain physical soundness (scalar should be 1.0 if range is invalid/zero).
    valid_range = vpd_range > 0

    # 2. Calculate the raw scalar based on the inverse linear ramp function:
    # Scalar = (VPD_max - VPD) / (VPD_max - VPD_min)
    # This formula naturally results in 1.0 at VPD_min and 0.0 at VPD_max.
    scalar_raw = (vpd_max - vpd_da) / vpd_range

    # 3. Apply constraints using xarray.where and numpy.clip:
    # If the range is valid (VPD_max > VPD_min), use the calculated scalar, 
    # clipped between 0.0 and 1.0 to enforce the ramp function thresholds [4].
    # If the range is invalid or zero, assume optimal efficiency (scalar = 1.0).
    vpd_scalar = xr.where(
        valid_range, 
        np.clip(scalar_raw, 0.0, 1.0),
        1.0
    )
    
    return vpd_scalar


#%% Calculate daily Tmin and Tmin_scalar [0,1]

# obtain air temperature data array 
t = t_ssrd_ds.t2m
#convert to celsius
t = t - 273.15

#calculate daily minimum value
t_min = t.resample(valid_time='1D').min()

t_min_scalar = calc_t_min_scalar(t_min, t_min_min, t_min_max)

#%% Calculate daily daytime average VPD 

# obtain dewpoint temperature data array
t_dpt = dpt_ds.d2m

#convert to celsius
t_dpt = t_dpt - 273.15

# Calculate saturated vapor pressure for air temperature and dewpoint
e_sat = calc_e(t)  # Saturated vapor pressure at air temperature
e_actual = calc_e(t_dpt)  # Actual vapor pressure at dewpoint temperature

# Calculate VPD (Vapor Pressure Deficit) in Pa
vpd_hourly = e_sat - e_actual

# Get solar radiation data
ssrd = t_ssrd_ds.ssrd

# Align the datasets (ensures matching time coordinates)
vpd_aligned, ssrd_aligned = xr.align(vpd_hourly, ssrd, join='inner')

# Create daytime mask (hours with positive solar radiation)
daytime_mask = ssrd_aligned > 0  # or use a threshold like > 10

# Calculate daily daytime average VPD
vpd_daytime_daily = vpd_aligned.where(daytime_mask).resample(valid_time='1D').mean()

#%% Calculate daily VPD_scalar [0,1]

vpd_scalar = calc_vpd_scalar(vpd_daytime_daily, VPD_min, VPD_max)

#%% Calculate eps 

eps = E_max * vpd_scalar * t_min_scalar

#%% save eps as netcdf
path_save = os.path.expanduser('./../Results/eps/')
name = 'eps_time_1980_2023_latlon_50_54_3_8.nc'
save_toggle = input("Save CF_solar results? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    eps.to_netcdf(path_save+name)

#%% sample time series
eps.isel(latitude=0, longitude=0).sel(valid_time=slice('2017-06-01', '2018-06-05')).plot()

# Improve axis labels, title, and layout
plt.xlabel("Date")
plt.ylabel("Light Use Efficiency")  # Replace with your variable's unit or label
plt.title("Actual Light Use Efficiency for crops \n at a single gridcell for 2017-2018")  # Adjust as needed
plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.tick_params(axis='x', rotation=45)  # Rotate x labels if crowded

# Optional: Save the figure with high dpi for presentations
plt.savefig("eps_timeseries_presentation.png", dpi=300)

# Show the plot (in interactive environments)
plt.show()