#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compound Drought-Heatwave (CDHW) analysis.

Loads SMDI and SVDI results; identifies drought and compound events (step by step).

@author: afer
"""

#%%
from pathlib import Path
from cftime import sec_units
import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rioxarray

#%% Load SMDI and SVDI

results_path = Path("./../Results/")
smdi_dir = results_path / "SMDI"
svdi_dir = results_path / "SVDI"

smdi_files = sorted(smdi_dir.glob("*.nc"))
svdi_files = sorted(svdi_dir.glob("*.nc"))

smdi = xr.open_dataarray(smdi_files[0])
svdi = xr.open_dataarray(svdi_files[0])

print("Loaded SMDI and SVDI.")
print(f"SMDI dims: {smdi.dims}")
print(f"SVDI dims: {svdi.dims}")

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
    return drought_mask


def identify_VPD_events(svdi, threshold=1.0):
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
    drought_mask, vpd_mask = xr.align(drought_mask, vpd_mask, join="inner")
    time_dim = drought_mask.dims[0]
    T = drought_mask.sizes[time_dim]
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

    coords = {
        time_dim: drought_mask[time_dim],
        "latitude": drought_mask["latitude"],
        "longitude": drought_mask["longitude"],
    }
    dims = [time_dim, "latitude", "longitude"]
    compound_mask = xr.DataArray(
        compound_out,
        coords=coords,
        dims=dims,
        attrs={"long_name": "compound drought-heatwave event day"},
    )
    return compound_mask

def characterize_compound_events(compound_mask, smdi, svdi):
    """
    Extract metrics for each compound event per grid cell.
    
    Returns
    -------
    events_df : pandas.DataFrame
        One row per event with columns: grid_cell, start_date, end_date, 
        duration, peak_svdi, min_smdi, mean_svdi, mean_smdi
    """
    from scipy.ndimage import label
    
    events = []
    n_lat = compound_mask.sizes['latitude']
    n_lon = compound_mask.sizes['longitude']
    time_coord = compound_mask.valid_time  # or 'time' depending on your coord name
    
    for i in range(n_lat):
        for j in range(n_lon):
            # Extract 1D time series for this grid cell
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            
            if not np.any(mask_1d):
                continue  # No events at this location
            
            # Label contiguous events
            labeled, n_events = label(mask_1d)
            
            for event_id in range(1, n_events + 1):
                event_days = labeled == event_id
                
                # Extract values during event
                smdi_event = smdi.isel(latitude=i, longitude=j).values[event_days]
                svdi_event = svdi.isel(latitude=i, longitude=j).values[event_days]
                times_event = time_coord.values[event_days]
                
                events.append({
                    'latitude': compound_mask.latitude.values[i],
                    'longitude': compound_mask.longitude.values[j],
                    'start_date': pd.Timestamp(times_event[0]),
                    'end_date': pd.Timestamp(times_event[-1]),
                    'duration': len(times_event),
                    'peak_svdi': np.max(svdi_event),
                    'min_smdi': np.min(smdi_event),
                    'mean_svdi': np.mean(svdi_event),
                    'mean_smdi': np.mean(smdi_event),
                })
    
    return pd.DataFrame(events)

def compute_annual_metrics(compound_mask, smdi, svdi):
    years = compound_mask.valid_time.dt.year
    
    metrics = {}
    for year in np.unique(years):
        year_mask = compound_mask.sel(valid_time=compound_mask.valid_time.dt.year == year)
        year_smdi = smdi.sel(valid_time=smdi.valid_time.dt.year == year)
        year_svdi = svdi.sel(valid_time=svdi.valid_time.dt.year == year)
        
        # Event count
        from scipy.ndimage import label
        spatial_mean_mask = year_mask.mean(dim=['latitude', 'longitude']) > 0.5
        labeled, n_events = label(spatial_mean_mask.values)
        
        # Integrated severity: sum of |SMDI| × SVDI during compound days
        compound_smdi = year_smdi.where(year_mask, 0)
        compound_svdi = year_svdi.where(year_mask, 0)
        severity = (np.abs(compound_smdi) * compound_svdi).sum()
        
        # Compound days
        n_days = year_mask.sum()
        
        metrics[int(year)] = {
            'n_events': n_events,
            'n_compound_days': float(n_days.mean()),  # Spatial average
            'integrated_severity': float(severity.mean()),  # Spatial average
        }
    
    return pd.DataFrame(metrics).T

# Compute return period per grid cell
def compute_return_periods(compound_mask):
    """
    Calculate mean return period (years between events) per grid cell.
    """
    from scipy.ndimage import label
    
    n_years = len(np.unique(compound_mask.valid_time.dt.year))
    n_lat = compound_mask.sizes['latitude']
    n_lon = compound_mask.sizes['longitude']
    
    return_periods = np.full((n_lat, n_lon), np.nan)
    event_counts = np.zeros((n_lat, n_lon))
    
    for i in range(n_lat):
        for j in range(n_lon):
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            
            if not np.any(mask_1d):
                continue
            
            # Count events
            labeled, n_events = label(mask_1d)
            event_counts[i, j] = n_events
            
            # Return period = years / number of events
            if n_events > 0:
                return_periods[i, j] = n_years / n_events
    
    return_period_da = xr.DataArray(
        return_periods,
        coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
        dims=['latitude', 'longitude']
    )
    
    return return_period_da, event_counts

#%% Example: build masks and compound events

drought_mask = identify_smdi_events(smdi, threshold=-2.0)
vpd_mask = identify_VPD_events(svdi, threshold=1.0)

compound_mask = identify_compound_events(
    drought_mask,
    vpd_mask,
    max_offset_days=2,
    max_gap=7,
    min_duration=3,
)
print("\nCompound mask (after merge and filter):")
print(compound_mask)

events_df = characterize_compound_events(compound_mask, smdi, svdi)

# %%
# Pick a representative grid cell (e.g., central Netherlands)
lat_idx = len(smdi.latitude) // 2
lon_idx = len(smdi.longitude) // 2

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Data for plotting
smdi_cell = smdi.isel(latitude=lat_idx, longitude=lon_idx)
svdi_cell = svdi.isel(latitude=lat_idx, longitude=lon_idx)
dates = smdi.valid_time

# Highlight compound event days in gray (background)
compound_days = compound_mask.isel(latitude=lat_idx, longitude=lon_idx).values
for ax in axes:
    ax.fill_between(
        dates, 
        smdi_cell.min()-1,  # Go below Y-axis
        smdi_cell.max()+1,  # Go above Y-axis
        where=compound_days,
        color='gray',
        alpha=0.2,
        zorder=0,
        label="Compound event days" if ax==axes[0] else '_nolegend_'
    )

# Plot SMDI: Thin light line (raw), then 3-month rolling mean (thick dark line)
axes[0].plot(
    dates, 
    smdi_cell, 
    color='brown', 
    linewidth=1, 
    alpha=0.5, 
    label='SMDI (daily)', 
    zorder=1,
)
smdi_rolling = smdi_cell.rolling(valid_time=90, center=True).mean()
axes[0].plot(
    dates, 
    smdi_rolling, 
    color='brown', 
    linewidth=2.5,
    label='SMDI (3-mo rolling mean)', 
    zorder=2
)
axes[0].axhline(
    -2.0, 
    color='black', 
    linestyle='--', 
    linewidth=2, 
    alpha=0.95, 
    label='Drought threshold'
)
axes[0].set_ylabel('SMDI', fontsize=18)
axes[0].set_title(
    f'Grid cell: {float(smdi.latitude[lat_idx]):.2f}°N, {float(smdi.longitude[lon_idx]):.2f}°E',
    fontsize=20,
    weight='bold'
)
axes[0].legend(fontsize=15)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='both', labelsize=15)

# Plot SVDI: Thin light line (raw), then 3-month rolling mean (thick dark line)
axes[1].plot(
    dates, 
    svdi_cell, 
    color='orangered', 
    linewidth=1, 
    alpha=0.5, 
    label='SVDI (daily)', 
    zorder=1,
)
svdi_rolling = svdi_cell.rolling(valid_time=90, center=True).mean()
axes[1].plot(
    dates, 
    svdi_rolling, 
    color='orangered', 
    linewidth=2.5, 
    label='SVDI (3-mo rolling mean)', 
    zorder=2
)
axes[1].axhline(
    1.0, 
    color='black', 
    linestyle='--', 
    linewidth=2, 
    alpha=0.95, 
    label='VPD threshold'
)
axes[1].set_ylabel('SVDI', fontsize=18)
axes[1].set_xlabel('Time', fontsize=18)
axes[1].legend(fontsize=15)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()


# %%
# Annual event metrics

annual_metrics = compute_annual_metrics(compound_mask, smdi, svdi)

# Plot dual-axis: frequency + severity
fig, ax1 = plt.subplots(figsize=(14, 6))

# Compound days (bars)
ax1.bar(annual_metrics.index, annual_metrics['n_compound_days'], 
        alpha=0.6, color='orangered', label='Compound days/year')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Compound Days per Year (spatial avg)', fontsize=12, color='orangered')
ax1.tick_params(axis='y', labelcolor='orangered')

# Integrated severity (line)
ax2 = ax1.twinx()
ax2.plot(annual_metrics.index, annual_metrics['integrated_severity'], 
         color='darkred', linewidth=2.5, marker='o', markersize=4,
         label='Integrated severity')
ax2.set_ylabel('Integrated Severity (|SMDI| × SVDI)', fontsize=12, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

# Highlight key years
ax1.axvline(2003, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(2003, ax1.get_ylim()[1]*0.95, '2003 Drought', 
         ha='right', fontsize=10, style='italic')

ax1.axvline(2018, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(2018, ax1.get_ylim()[1]*0.95, '2018 Drought', 
         ha='right', fontsize=10, style='italic')
ax1.axvline(2022, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(2022, ax1.get_ylim()[1]*0.85, '2022 Drought', 
         ha='right', fontsize=10, style='italic')

ax1.set_title('Compound Drought-Heatwave Events in The Netherlands (1950-2024)', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
fig.tight_layout()
# plt.savefig('compound_events_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
# 10-year rolling statistics with trend
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Rolling mean
rolling_days = annual_metrics['n_compound_days'].rolling(10, center=True).mean()
rolling_severity = annual_metrics['integrated_severity'].rolling(10, center=True).mean()

# Plot compound days
axes[0].plot(annual_metrics.index, annual_metrics['n_compound_days'], 
             'o-', alpha=0.3, color='gray', label='Annual')
axes[0].plot(rolling_days.index, rolling_days, 
             linewidth=3, color='orangered', label='10-year rolling mean')

# Add linear trend
from scipy.stats import linregress
years = annual_metrics.index.values
slope, intercept, r, p, stderr = linregress(years, annual_metrics['n_compound_days'])
trend_line = slope * years + intercept
axes[0].plot(years, trend_line, '--', color='black', linewidth=2, 
             label=f'Trend: {slope:.2f} days/year (p={p:.3f})')

axes[0].set_ylabel('Compound Days/Year', fontsize=12)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].set_title('Temporal Evolution of Compound Events', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot severity
axes[1].plot(annual_metrics.index, annual_metrics['integrated_severity'],
             'o-', alpha=0.3, color='gray', label='Annual')
axes[1].plot(rolling_severity.index, rolling_severity,
             linewidth=3, color='darkred', label='10-year rolling mean')

# Severity trend
slope_sev, intercept_sev, r_sev, p_sev, _ = linregress(
    years[~np.isnan(annual_metrics['integrated_severity'])], 
    annual_metrics['integrated_severity'].dropna()
)
trend_line_sev = slope_sev * years + intercept_sev
axes[1].plot(years, trend_line_sev, '--', color='black', linewidth=2,
             label=f'Trend: p={p_sev:.3f}')

axes[1].set_ylabel('Integrated Severity', fontsize=12)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('compound_events_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# %%

return_periods, event_counts = compute_return_periods(compound_mask)

# Create land mask based on return periods
land_mask = return_periods < 2  # Filter out cells with >50 year return period

return_periods = return_periods.where(land_mask, np.nan)
# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Return period map
im1 = return_periods.plot(ax=axes[0], cmap='RdYlGn_r', 
                        #   vmin=1, vmax=15,
                          cbar_kwargs={'label': 'Mean Return Period (years)'})
axes[0].set_title('Compound Event Return Period\n(Lower = More Frequent)', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

# Event frequency map
total_events = xr.DataArray(
    event_counts,
    coords={'latitude': compound_mask.latitude, 'longitude': compound_mask.longitude},
    dims=['latitude', 'longitude']
)

land_mask = event_counts > 30  # Filter out cells with >50 year return period
total_events = total_events.where(land_mask, np.nan)
# Assign NaN to grid cells with 0 events
im2 = total_events.plot(
    ax=axes[1], cmap='YlOrRd',
    cbar_kwargs={'label': 'Total Events (1950-2024)'}
)   
axes[1].set_title('Total Number of Compound Events\n(1950-2024)', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

plt.tight_layout()
# plt.savefig('compound_events_spatial.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
