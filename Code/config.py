#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for all ERA5 download and analysis scripts.
Edit this file only — all other scripts should import from here.
"""

# ── Spatial domain ─────────────────────────────────────────────────────────────
# CDS area format: [North, West, South, East]
AREA = [55, -5, 41, 16]

# ── Temporal domain ────────────────────────────────────────────────────────────
YEAR_START = 1980
YEAR_END   = 2022

# ── Variables ──────────────────────────────────────────────────────────────────
# Each entry: 'short_name': 'CDS variable name'
# Short names are used for filenames and Zarr store naming
VARIABLES = {
    't2m'   : '2m_temperature',
    'd2m'   : '2m_dewpoint_temperature',
    'u10'   : '10m_u_component_of_wind',
    'v10'   : '10m_v_component_of_wind',
    'u100'  : '100m_u_component_of_wind',
    'v100'  : '100m_v_component_of_wind',
    'ssrd'  : 'surface_solar_radiation_downwards',
    'swvl1' : 'volumetric_soil_water_layer_1',
    'swvl2' : 'volumetric_soil_water_layer_2',
}
# Note: 10m and 100m wind speed are downloaded as U and V components
# and combined into wind speed magnitude in the preprocessing step.
# This is the correct approach — CDS does not provide wind speed directly.

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR  = '/scratch/your_username/ERA5/raw/'      # adjust to your Snellius scratch path
ZARR_DIR = '/scratch/your_username/ERA5/zarr/'

