#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:22:52 2025

@author: afer
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#%% set paths
path = os.path.expanduser('./../Data/ERA5/downloads/')

files = os.listdir(path)

paths = [path+file for file in files if not file.startswith('.')]
#%% read all downloads into single dataset

ds = xr.open_mfdataset(paths)

#%% get variables, lats, lons, and years to name big file

path_save = os.path.expanduser('./../Data/ERA5/')

dat_vars = '_'.join(list(ds.data_vars))
yrs = f'{ds.valid_time[0].dt.year.item()}_{ds.valid_time[-1].dt.year.item()}'
lats = f'{ds.latitude[-1].item():.0f}_{ds.latitude[0].item():.0f}'
lons = f'{ds.longitude[0].item():.0f}_{ds.longitude[-1].item():.0f}'
file_name = f"ERA5_hrly_vars_{dat_vars}_time_{yrs}_latlon_{lats}_{lons}"
#%% save big file
ds.to_netcdf(path_save+file_name+'.nc')

