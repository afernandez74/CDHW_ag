#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:35:48 2025

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

# SSRD, temp, wind speed file path
t_ssrd_file_name = 'ERA5_hrly_vars_ssrd_t2m_u100_v100_time_1980_2023_latlon_50_54_3_8'
file_path = dat_path + t_ssrd_file_name

# only load SSRD and temp
t_ssrd_ds = xr.open_mfdataset(file_path,
                              drop_variables =['u100','v100']
                              )

ssrd = t_ssrd_ds.ssrd / 3600 # J/m2 -> W/m2

#%% define constants
C = 0.45 # multiplicative constant

#%% daily timestep of SSRD values

# Create daytime mask (hours with positive solar radiation)
daytime_mask = ssrd > 0  # or use a threshold like > 10

# Calculate daily daytime average VPD
ssrd_daytime = ssrd.where(daytime_mask).resample(valid_time='1D').mean()

#%% save APAR results as netcdf
path_save = os.path.expanduser('./../Results/APAR/')
name = 'APAR_time_1980_2023_latlon_50_54_3_8.nc'
save_toggle = input("Save APAR results? (Y or N) \n ...")
if save_toggle==str.lower(save_toggle):
    ssrd_daytime.to_netcdf(path_save+name)



#%% plot gridcell timeseries
ssrd_daytime.isel(latitude=0, longitude=0).sel(valid_time=slice('2017-06-01', '2018-06-05')).plot()

# Improve axis labels, title, and layout
plt.xlabel("Date")
plt.ylabel("Surface Downward Radiation [W/m2]")  # Replace with your variable's unit or label
plt.title("Daily Daytime average SSRD \n at a single gridcell for 2017-2018")  # Adjust as needed
plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.tick_params(axis='x', rotation=45)  # Rotate x labels if crowded


# Show the plot (in interactive environments)
plt.show()