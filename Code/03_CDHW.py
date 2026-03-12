#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compound Drought-Heatwave (CDHW) analysis.

Loads SMDI and SVDI results; identifies drought and compound events (step by step).

@author: afer
"""

#%%
# Imports
import os
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rioxarray
from scipy.ndimage import label
from scipy.stats import linregress


#%% 
# Load SMDI and SVDI

results_path = Path("./../Results/")
 
# load datasets
smdi = xr.open_zarr(results_path / "SMDI" / "smdi.zarr", consolidated=True)
svdi = xr.open_zarr(results_path / "SVDI" / "svdi.zarr", consolidated=True)

# obtain dataarrays
smdi = smdi[list(smdi.data_vars)[0]]
svdi = svdi[list(svdi.data_vars)[0]]

# Unify chunks : fixes remainder-chunk inconsistency
smdi, svdi = xr.unify_chunks(smdi, svdi)

print(f"SMDI: {smdi.dims} | chunks: {dict(smdi.chunksizes)}")
print(f"SVDI: {svdi.dims} | chunks: {dict(svdi.chunksizes)}")

#%% 
# Load arrays into memory
smdi, svdi = xr.align(smdi, svdi, join="inner")
smdi = smdi.compute()
svdi = svdi.compute()
print(f"Aligned time range: {str(smdi.time.values[0])[:10]} → {str(smdi.time.values[-1])[:10]}")


#%%
def identify_smdi_events(smdi, threshold=-2.0):
    """
    Identify SMDI events from SMDI

    Parameters
    ----------
    smdi : xarray.DataArray
        Daily SMDI (e.g. dimensions time, latitude, longitude).
    threshold : float
        Drought threshold. Default -2: values below -2 are considered drought.

    Returns
    -------
    drought_mask : xarray.DataArray
        Boolean mask, True where SMDI < threshold (drought), False otherwise.
    """
    drought_mask = smdi < threshold
    drought_mask.attrs = {'long_name': 'Drought day (SMDI)', 'threshold': threshold}
    return drought_mask


def identify_vpd_events(svdi, threshold=1.0):
    """
    Identify heat stress events from SVDI: any value above the threshold

    Parameters
    ----------
    svdi : xarray.DataArray
        Daily SVDI (e.g. dimensions time, latitude, longitude).
    threshold : float
        Heat stress threshold. Default 1: values above 1 are considered heat stress.

    Returns
    -------
    heat_stress_mask : xarray.DataArray
        Boolean mask, True where SVDI > threshold (heat stress), False otherwise.
    """
    vpd_mask = svdi > threshold
    vpd_mask.attrs = {'long_name': 'VPD stress day (SVDI)', 'threshold': threshold}
    return vpd_mask


def _merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """
    Given a 1D boolean array, merge runs of True separated by <= max_gap days,
    then keep only runs with duration >= min_duration. Returns 1D boolean.
    """
    if not np.any(flag):
        return np.zeros_like(flag, dtype=bool)
    T = len(flag)
    # Find runs of True
    runs = []
    in_run = False
    start = None
    for t in range(T):
        if flag[t] and not in_run:
            start = t
            in_run = True
        elif not flag[t] and in_run:
            runs.append((start, t - 1))
            in_run = False
    if in_run:
        runs.append((start, T - 1))
    if not runs:
        return np.zeros_like(flag, dtype=bool)
    # Merge runs separated by <= max_gap days
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # Keep only runs with duration >= min_duration
    out = np.zeros_like(flag, dtype=bool)
    for s, e in merged:
        if e - s + 1 >= min_duration:
            out[s : e + 1] = True
    return out


def identify_compound_events(
    drought_mask,
    vpd_mask,
    max_offset_days=0,
    max_gap=1,
    min_duration=3,
):
    """
    Identify compound (drought + heat stress) events from drought and VPD masks.

    A day is a compound day if:
    - drought and VPD on the same day, or
    - drought on that day and an extreme VPD day occurs within the next
      max_offset_days days (drought-only day still counts as compound).

    After building these raw compound days, contiguous runs are merged when
    separated by <= max_gap days, and runs shorter than min_duration days
    are dropped.

    Parameters
    ----------
    drought_mask : xarray.DataArray
        Boolean, True on drought days (from identify_smdi_events).
    vpd_mask : xarray.DataArray
        Boolean, True on VPD stress days (from identify_VPD_events).
    max_offset_days : int
        Maximum number of days after a drought day within which an extreme VPD
        day must occur for that drought day to count as compound. 0 = same-day only.
    max_gap : int
        Maximum gap (days) between compound days to merge into one event.
    min_duration : int
        Minimum number of days for a run to count as a compound event.

    Returns
    -------
    compound_mask : xarray.DataArray
        Boolean mask, True on compound event days (after merging and filtering).
    """

    T = drought_mask.sizes['time']
    n_lat = drought_mask.sizes["latitude"]
    n_lon = drought_mask.sizes["longitude"]

    drought = drought_mask.values.astype(bool)
    vpd = vpd_mask.values.astype(bool)

    # Raw compound: same-day OR (drought and VPD in next max_offset_days)
    compound_raw = np.zeros((T, n_lat, n_lon), dtype=bool)
    for t in range(T):
        future_vpd = np.zeros((n_lat, n_lon), dtype=bool)
        if max_offset_days > 0 and t + 1 < T:
            end = min(t + max_offset_days + 1, T)
            future_vpd = np.any(vpd[t + 1 : end], axis=0)
        compound_raw[t] = drought[t] & (vpd[t] | future_vpd)

    # Per grid cell: merge runs (max_gap) and filter by min_duration
    compound_out = np.zeros_like(compound_raw, dtype=bool)
    for i in range(n_lat):
        for j in range(n_lon):
            compound_out[:, i, j] = _merge_and_filter_runs_1d(
                compound_raw[:, i, j], max_gap, min_duration
            )

    compound_mask = xr.DataArray(
        compound_out,
        coords={
            'time'     : drought_mask['time'],
            'latitude' : drought_mask['latitude'],
            'longitude': drought_mask['longitude'],
        },
        dims=['time', 'latitude', 'longitude'],
        attrs={'long_name': 'Compound drought-heatwave event day'},
    )
    return compound_mask


def characterize_compound_events(compound_mask, smdi, svdi):
    """
    Extract per-event metrics for each grid cell.
 
    Returns
    -------
    events_df : pandas.DataFrame
        One row per event with columns: latitude, longitude, start_date,
        end_date, duration, peak_svdi, min_smdi, mean_svdi, mean_smdi.
    """
    events  = []
    n_lat   = compound_mask.sizes['latitude']
    n_lon   = compound_mask.sizes['longitude']
    time_coord = compound_mask['time']
 
    for i in range(n_lat):
        for j in range(n_lon):
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            if not np.any(mask_1d):
                continue
            labeled, n_events = label(mask_1d)
            for event_id in range(1, n_events + 1):
                event_days   = labeled == event_id
                smdi_event   = smdi.isel(latitude=i, longitude=j).values[event_days]
                svdi_event   = svdi.isel(latitude=i, longitude=j).values[event_days]
                times_event  = time_coord.values[event_days]
                events.append({
                    'latitude'  : float(compound_mask.latitude.values[i]),
                    'longitude' : float(compound_mask.longitude.values[j]),
                    'start_date': pd.Timestamp(times_event[0]),
                    'end_date'  : pd.Timestamp(times_event[-1]),
                    'duration'  : len(times_event),
                    'peak_svdi' : float(np.nanmax(svdi_event)),
                    'min_smdi'  : float(np.nanmin(smdi_event)),
                    'mean_svdi' : float(np.nanmean(svdi_event)),
                    'mean_smdi' : float(np.nanmean(smdi_event)),
                })
    return pd.DataFrame(events)
 
 
def compute_annual_metrics(compound_mask, smdi, svdi):
    """
    Compute spatially averaged annual compound event statistics.
 
    Returns
    -------
    metrics_df : pandas.DataFrame
        Indexed by year with columns: n_events, n_compound_days,
        integrated_severity.
    """
    years   = np.unique(compound_mask.time.dt.year.values)
    metrics = {}
 
    for year in years:
        year_sel  = compound_mask.time.dt.year == year
        year_mask = compound_mask.sel(time=compound_mask.time.dt.year == year)
        year_smdi = smdi.sel(time=smdi.time.dt.year == year)
        year_svdi = svdi.sel(time=svdi.time.dt.year == year)
 
        # Event count from spatially averaged mask
        spatial_mean = year_mask.mean(dim=['latitude', 'longitude']) > 0.5
        _, n_events  = label(spatial_mean.values)
 
        # Integrated severity: |SMDI| × SVDI on compound days
        compound_smdi = year_smdi.where(year_mask, 0)
        compound_svdi = year_svdi.where(year_mask, 0)
        severity      = (np.abs(compound_smdi) * compound_svdi).sum()
 
        metrics[int(year)] = {
            'n_events'            : n_events,
            'n_compound_days'     : float(year_mask.sum(dim='time').mean()),
            'integrated_severity' : float(severity.mean()),
        }
    return pd.DataFrame(metrics).T
 
 
def compute_return_periods(compound_mask):
    """
    Calculate mean return period (years between events) per grid cell.
 
    Returns
    -------
    return_period_da : xarray.DataArray
        Return period in years per grid cell.
    event_counts : numpy.ndarray
        Total number of events per grid cell.
    """
    n_years = len(np.unique(compound_mask.time.dt.year.values))
    n_lat   = compound_mask.sizes['latitude']
    n_lon   = compound_mask.sizes['longitude']
 
    return_periods = np.full((n_lat, n_lon), np.nan)
    event_counts   = np.zeros((n_lat, n_lon))
 
    for i in range(n_lat):
        for j in range(n_lon):
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            if not np.any(mask_1d):
                continue
            _, n_events = label(mask_1d)
            event_counts[i, j] = n_events
            if n_events > 0:
                return_periods[i, j] = n_years / n_events
 
    return_period_da = xr.DataArray(
        return_periods,
        coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
        dims=['latitude', 'longitude'],
        attrs={'long_name': 'Mean return period', 'units': 'years'},
    )
    return return_period_da, event_counts

#%% 
# Build masks and compound events

drought_mask  = identify_smdi_events(smdi, threshold=-2.0)
vpd_mask      = identify_vpd_events(svdi, threshold=1.0)
 
compound_mask = identify_compound_events(
    drought_mask,
    vpd_mask,
    max_offset_days=2,
    max_gap=7,
    min_duration=3,
)
print(f"\nCompound mask: {compound_mask.dims}, {compound_mask.sizes}")
print(f"Compound event days (spatial mean): {float(compound_mask.mean()):.4f}")


# %%
events_df = characterize_compound_events(compound_mask, smdi, svdi)
print(f"\nTotal compound events identified: {len(events_df)}")
print(events_df.head())
 
#%%
# Time series plot for central gridcell
 
lat_idx = smdi.latitude.size // 2
lon_idx = smdi.longitude.size // 2
 
smdi_cell     = smdi.isel(latitude=lat_idx, longitude=lon_idx)
svdi_cell     = svdi.isel(latitude=lat_idx, longitude=lon_idx)
compound_days = compound_mask.isel(latitude=lat_idx, longitude=lon_idx).values
dates         = smdi.time.values
 
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
 
for ax in axes:
    ax.fill_between(
        dates,
        -10, 10,
        where=compound_days,
        color='gray', alpha=0.2, zorder=0,
        label='Compound event days' if ax is axes[0] else '_nolegend_'
    )
 
# SMDI panel
axes[0].plot(dates, smdi_cell, color='brown', linewidth=1, alpha=0.5, label='SMDI (daily)', zorder=1)
smdi_rolling = smdi_cell.rolling(time=90, center=True).mean()
axes[0].plot(dates, smdi_rolling, color='brown', linewidth=2.5, label='SMDI (3-mo rolling mean)', zorder=2)
axes[0].axhline(-2.0, color='black', linestyle='--', linewidth=2, alpha=0.95, label='Drought threshold')
axes[0].set_ylabel('SMDI', fontsize=18)
axes[0].set_title(
    f'Grid cell: {float(smdi.latitude[lat_idx]):.2f}°N, {float(smdi.longitude[lon_idx]):.2f}°E',
    fontsize=20, fontweight='bold'
)
axes[0].legend(fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='both', labelsize=13)
 
# SVDI panel
axes[1].plot(dates, svdi_cell, color='orangered', linewidth=1, alpha=0.5, label='SVDI (daily)', zorder=1)
svdi_rolling = svdi_cell.rolling(time=90, center=True).mean()
axes[1].plot(dates, svdi_rolling, color='orangered', linewidth=2.5, label='SVDI (3-mo rolling mean)', zorder=2)
axes[1].axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.95, label='VPD threshold')
axes[1].set_ylabel('SVDI', fontsize=18)
axes[1].set_xlabel('Time', fontsize=18)
axes[1].legend(fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='both', labelsize=13)
 
plt.tight_layout()
plt.show()
 
#%%
# Annual metrics and trend plots
 
annual_metrics = compute_annual_metrics(compound_mask, smdi, svdi)
 
# Dual-axis: frequency + severity
fig, ax1 = plt.subplots(figsize=(14, 6))
 
ax1.bar(annual_metrics.index, annual_metrics['n_compound_days'],
        alpha=0.6, color='orangered', label='Compound days/year')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Compound Days per Year (spatial avg)', fontsize=12, color='orangered')
ax1.tick_params(axis='y', labelcolor='orangered')
 
ax2 = ax1.twinx()
ax2.plot(annual_metrics.index, annual_metrics['integrated_severity'],
         color='darkred', linewidth=2.5, marker='o', markersize=4,
         label='Integrated severity')
ax2.set_ylabel('Integrated Severity (|SMDI| × SVDI)', fontsize=12, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')
 
for year, label_text in {2003: '2003', 2018: '2018', 2022: '2022'}.items():
    ax1.axvline(year, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(year, ax1.get_ylim()[1] * 0.95, label_text,
             ha='right', fontsize=10, style='italic')
 
ax1.set_title('Compound Drought-Heatwave Events (1980–2024)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
 
# 10-year rolling statistics with trend
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
 
years_arr     = annual_metrics.index.values.astype(float)
rolling_days  = annual_metrics['n_compound_days'].rolling(10, center=True).mean()
rolling_sev   = annual_metrics['integrated_severity'].rolling(10, center=True).mean()
 
# Compound days panel
axes[0].plot(annual_metrics.index, annual_metrics['n_compound_days'],
             'o-', alpha=0.3, color='gray', label='Annual')
axes[0].plot(rolling_days.index, rolling_days,
             linewidth=3, color='orangered', label='10-year rolling mean')
 
slope, intercept, _, p, _ = linregress(years_arr, annual_metrics['n_compound_days'])
axes[0].plot(years_arr, slope * years_arr + intercept,
             '--', color='black', linewidth=2,
             label=f'Trend: {slope:.2f} days/yr (p={p:.3f})')
axes[0].set_ylabel('Compound Days/Year', fontsize=12)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].set_title('Temporal Evolution of Compound Events', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
 
# Severity panel
valid        = ~np.isnan(annual_metrics['integrated_severity'])
years_valid  = years_arr[valid]
sev_valid    = annual_metrics['integrated_severity'].values[valid]
 
axes[1].plot(annual_metrics.index, annual_metrics['integrated_severity'],
             'o-', alpha=0.3, color='gray', label='Annual')
axes[1].plot(rolling_sev.index, rolling_sev,
             linewidth=3, color='darkred', label='10-year rolling mean')
 
slope_sev, intercept_sev, _, p_sev, _ = linregress(years_valid, sev_valid)
axes[1].plot(years_arr, slope_sev * years_arr + intercept_sev,
             '--', color='black', linewidth=2, label=f'Trend: p={p_sev:.3f}')
axes[1].set_ylabel('Integrated Severity', fontsize=12)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.show()
 
#%%
# Return period spatial maps
 
return_periods, event_counts = compute_return_periods(compound_mask)
 
# Mask out sparsely affected cells for cleaner maps
return_periods = return_periods.where(return_periods < 25, np.nan)
event_counts_da = xr.DataArray(
    np.where(event_counts > 2, event_counts, np.nan),
    coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
    dims=['latitude', 'longitude'],
)
 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
return_periods.plot(
    ax=axes[0], cmap='RdYlGn_r',
    cbar_kwargs={'label': 'Mean Return Period (years)'}
)
axes[0].set_title('Compound Event Return Period\n(Lower = More Frequent)',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
 
event_counts_da.plot(
    ax=axes[1], cmap='YlOrRd',
    cbar_kwargs={'label': 'Total Compound Events'}
)
axes[1].set_title(f'Total Number of Compound Events\n'
                  f'({str(smdi.time.values[0])[:4]}–{str(smdi.time.values[-1])[:4]})',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
 
plt.tight_layout()
plt.show()

