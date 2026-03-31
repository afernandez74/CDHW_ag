#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compound Drought-Heatwave (CDHW) analysis.

Identifies and characterises compound drought-heatwave events across
France, Belgium, Netherlands, Germany. Results are summarised at the
whole-domain, country, and NUTS-2 regional scale.

@author: afer
"""

#%%
# ── Imports 
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import rioxarray
from scipy.ndimage import label
from scipy.stats import linregress, rankdata, kendalltau, gaussian_kde
import pyvinecopulib as pv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


#%%
# ── Load SMDI and SVDI 
# from 01_calc_SMDI.py and 02_calc_SVDI.py

results_path = Path("./../Results/")

smdi = xr.open_zarr(results_path / "SMDI" / "smdi.zarr", consolidated=True)
svdi = xr.open_zarr(results_path / "SVDI" / "svdi.zarr", consolidated=True)

smdi = smdi[list(smdi.data_vars)[0]]
svdi = svdi[list(svdi.data_vars)[0]]

smdi, svdi = xr.unify_chunks(smdi, svdi)
smdi, svdi = xr.align(smdi, svdi, join="inner")
smdi = smdi.compute()
svdi = svdi.compute()

print(f"SMDI: {smdi.dims} | {dict(smdi.sizes)}")
print(f"Time range: {str(smdi.time.values[0])[:10]} -> {str(smdi.time.values[-1])[:10]}")


#%%
#  Shapefiles and study region 
# All clipping happens once here — every downstream object is land-only.

# countries for analysis
countries = ['France', 'Belgium', 'Netherlands', 'Germany']

# locate shapefile for clipping area of analysis
global_countries_path = Path('./../Data/countries_shp/ne_110m_admin_0_countries.shp')
# read shapefile
countries_gdf = gpd.read_file(str(global_countries_path))

# merge all countries into one shape and 
region_parts = (
    countries_gdf[countries_gdf['SOVEREIGNT'].isin(countries)]
    .dissolve()
    .explode(index_parts=True)
    .reset_index(drop=True)
)
# filter out Corsica (for potato study) based on centroids after explode() function of dissolved geography
corsica_bbox = (8.5, 41.3, 9.6, 43.1)
is_corsica = region_parts.geometry.apply(
    lambda g: (corsica_bbox[0] <= g.centroid.x <= corsica_bbox[2] and
               corsica_bbox[1] <= g.centroid.y <= corsica_bbox[3])
)
region = region_parts[~is_corsica].dissolve()

# individual countries shapefiles for later use
countries_individual = (
    countries_gdf[countries_gdf['SOVEREIGNT'].isin(countries)]
    [['SOVEREIGNT', 'geometry']].copy()
)

# ── NUTS-2 shapefile 
nuts_path = Path('./../Data/NUTS_RG_01M_2021_4326_LEVL_2')
nuts_gdf  = gpd.read_file(str(nuts_path))

nuts_country_codes = ["FR", "BE", "NL", "DE"]
nuts_study = (
    nuts_gdf[
        (nuts_gdf["LEVL_CODE"] == 2) &
        (nuts_gdf["CNTR_CODE"].isin(nuts_country_codes)) &
        (nuts_gdf["NUTS_ID"] != "FR83")        # drop Corsica
    ]
    .copy()
    .reset_index(drop=True)
)
print(f"NUTS-2 regions loaded: {len(nuts_study)}")

#  Clip smdi/svdi to land 
# Agricultural impacts don't apply on ocean or coastal cells
def clip_to_region(da, region_geom, all_touched=False):
    da = da.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(region_geom.geometry, region_geom.crs,
                       all_touched=all_touched, drop=False)

smdi = clip_to_region(smdi, region)
svdi = clip_to_region(svdi, region)

n_land_cells = int(np.isfinite(smdi.isel(time=0).values).sum())
print(f"Land grid cells in domain: {n_land_cells}")

#  Projections and display settings 
# Lambert Azimuthal Equal Area centred on the domain (Eurostat standard).
LAEA = ccrs.LambertAzimuthalEqualArea(central_longitude=10., central_latitude=52.)
PC   = ccrs.PlateCarree()

DOMAIN_EXTENT = [
    float(smdi.longitude.values.min()) - 0.5,
    float(smdi.longitude.values.max()) + 0.5,
    float(smdi.latitude.values.min())  - 0.5,
    float(smdi.latitude.values.max())  + 0.5,
]

country_colors = {
    "France"      : "#4477AA",
    "Germany"     : "#EE6677",
    "Netherlands" : "#228833",
    "Belgium"     : "#CCBB44",
}
benchmark_years = [2003, 2018, 2022]



#%%
# ── Thresholds (two-stage approach following Shan et al. 2024) ────────────────

# Initial identification (permissive)
SMDI_INITIAL = -1.0   # Shan et al. 2024
SVDI_INITIAL =  1.0   # Shan et al. 2024

# Severity filter (strict - at least one day must meet these)
SMDI_SEVERITY = -2.0  # Guo et al. 2023
SVDI_SEVERITY =  1.5  # Conservative

# For characterization metrics (use severity thresholds)
SMDI_THRESH = SMDI_SEVERITY
SVDI_THRESH = SVDI_SEVERITY

#%%
#  Functions 

def identify_smdi_events(smdi, threshold=SMDI_INITIAL): 
    """
    Boolean mask: True where SMDI < threshold

    Parameters
    ----------
    smdi : SMDI dataarray 
    threshold : float
        Initial (permissive) threshold for drought identification
    """
    mask = smdi < threshold
    mask.attrs = {'long_name': 'Drought day (SMDI)', 'threshold': threshold}
    return mask


def identify_vpd_events(svdi, threshold=SVDI_INITIAL): 
    """Boolean mask: True where SVDI > threshold."""
    mask = svdi > threshold
    mask.attrs = {'long_name': 'VPD stress day (SVDI)', 'threshold': threshold}
    return mask

def identify_vpd_events(svdi, threshold=SVDI_THRESH):
    """Boolean mask: True where SVDI > threshold."""
    mask = svdi > threshold
    mask.attrs = {'long_name': 'VPD stress day (SVDI)', 'threshold': threshold}
    return mask

def _merge_and_filter_runs_1d(flag, max_gap, min_duration):
    """
    Merge and filter contiguous runs of True in a 1D boolean array.
    Following Shan et al. (2024):

    Parameters
    ----------
    flag : np.ndarray of bool, shape (T,)
        1D boolean time series. True = a candidate event day

    max_gap : int
        Maximum gap of days allowed between True day events

    min_duration : int
        Minimum number of days a run must span to be retained.

    Returns
    -------
    out : np.ndarray of bool, shape (T,)

        Boolean array of merged and duration-filtered runs


    """
    if not np.any(flag):
        return np.zeros_like(flag, dtype=bool)

    T = len(flag)

    # Identify contiguous runs 
    runs, in_run, start = [], False, None
    for t in range(T):
        if flag[t] and not in_run:
            start, in_run = t, True
        elif not flag[t] and in_run:
            runs.append((start, t - 1))
            in_run = False
    if in_run:                          # close a run that reaches the end
        runs.append((start, T - 1))
    if not runs:
        return np.zeros_like(flag, dtype=bool)

    # Merge runs whose gap is <= max_gap 
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], e)   # extend previous run to cover gap
        else:
            merged.append((s, e))

    # Retain only runs >= min_duration 
    out = np.zeros_like(flag, dtype=bool)
    for s, e in merged:
        if e - s + 1 >= min_duration:
            out[s : e + 1] = True

    return out

def identify_compound_events(drought_mask, vpd_mask,
                              max_offset_days=0, max_gap=1, min_duration=3):
    """
    Identify compound (drought + VPD stress) events.
    Runs are merged across gaps <= max_gap and short runs are discarded.

    drought_mask : xr.DataArray of bool, dims (time, latitude, longitude)
        from identify_smdi_events().
    
    vpd_mask : xr.DataArray of bool, dims (time, latitude, longitude)
        from identify_vpd_events().
    
    max_offset_days : int, optional
        Number of days after a drought day within which a VPD stress day
        must occur for the drought day to count as compound.
    
    max_gap : int
    
    min_duration : int

    Returns
    -------
    compound_mask : xr.DataArray of bool, dims (time, latitude, longitude)
        True on days and grid cells that belong to a compound event after
        merging and duration filtering. Coordinates and dimension names
        from drought_mask
    """
    # extract dimensions
    T, n_lat, n_lon = (drought_mask.sizes[d]
                       for d in ('time', 'latitude', 'longitude'))
    drought = drought_mask.values.astype(bool)
    vpd     = vpd_mask.values.astype(bool)

    compound_raw = np.zeros((T, n_lat, n_lon), dtype=bool)
    # loop through time
    for t in range(T):
        future_vpd = np.zeros((n_lat, n_lon), dtype=bool)
        if max_offset_days > 0 and t + 1 < T:
            future_vpd = np.any(
                vpd[t + 1 : min(t + max_offset_days + 1, T)], axis=0)
        compound_raw[t] = drought[t] & (vpd[t] | future_vpd)

    compound_out = np.zeros_like(compound_raw, dtype=bool)
    for i in range(n_lat):
        for j in range(n_lon):
            compound_out[:, i, j] = _merge_and_filter_runs_1d(
                compound_raw[:, i, j], max_gap, min_duration)

    return xr.DataArray(
        compound_out,
        coords={'time'     : drought_mask['time'],
                'latitude' : drought_mask['latitude'],
                'longitude': drought_mask['longitude']},
        dims=['time', 'latitude', 'longitude'],
        attrs={'long_name': 'Compound drought-heatwave event day'},
    )

def apply_severity_filter(event_mask, smdi, svdi, 
                          smdi_threshold=SMDI_SEVERITY,
                          svdi_threshold=SVDI_SEVERITY):
    """
    Filter events: keep only those with at least one day meeting severity criteria.
    
    Following Shan et al. (2024) two-stage approach: events must contain at least
    one day with SMDI < smdi_threshold OR SVDI > svdi_threshold to be retained.
    
    Parameters
    ----------
    event_mask : xr.DataArray of bool, dims (time, latitude, longitude)
        Events identified with initial (permissive) thresholds
    smdi : xr.DataArray
        Soil moisture deficit values
    svdi : xr.DataArray  
        VPD stress values
    smdi_threshold : float
        Strict threshold - at least one day must be below this
    svdi_threshold : float
        Strict threshold - at least one day must be above this
    
    Returns
    -------
    filtered_mask : xr.DataArray of bool
        Event mask with non-severe events removed
    """
    from scipy.ndimage import label
    
    n_time = event_mask.sizes['time']
    n_lat = event_mask.sizes['latitude']
    n_lon = event_mask.sizes['longitude']
    
    filtered = np.zeros((n_time, n_lat, n_lon), dtype=bool)
    
    for i in range(n_lat):
        for j in range(n_lon):
            mask_1d = event_mask.isel(latitude=i, longitude=j).values
            
            if not np.any(mask_1d):
                continue
            
            # Label individual events at this grid cell
            labeled, n_events = label(mask_1d)
            
            smdi_1d = smdi.isel(latitude=i, longitude=j).values
            svdi_1d = svdi.isel(latitude=i, longitude=j).values
            
            for event_id in range(1, n_events + 1):
                event_days = labeled == event_id
                
                # Check if at least one day meets severity criteria
                smdi_severe = np.any(smdi_1d[event_days] < smdi_threshold)
                svdi_severe = np.any(svdi_1d[event_days] > svdi_threshold)
                
                # Keep event if EITHER criterion is met
                if smdi_severe or svdi_severe:
                    filtered[:, i, j][event_days] = True
    
    return xr.DataArray(
        filtered,
        coords=event_mask.coords,
        dims=event_mask.dims,
        attrs={'long_name': f'{event_mask.attrs.get("long_name", "event")} (severity filtered)',
               'smdi_severity_threshold': smdi_threshold,
               'svdi_severity_threshold': svdi_threshold}
    )

def characterize_compound_events(compound_mask, smdi, svdi,
                                  smdi_thresh=SMDI_THRESH,
                                  svdi_thresh=SVDI_THRESH):
    """
    Single-pass extraction of per-event metrics and spatial severity aggregates.

    Spatial footprint (n_cells_affected) is added via groupby on start_date.

    Returns
    -------
    events_df : pd.DataFrame
        One row per (event x grid cell). Columns: latitude, longitude,
        start_date, end_date, duration, peak_svdi, min_smdi, mean_svdi,
        mean_smdi, csi, cr, ess_drought, ess_vpd, ess_compound,
        n_cells_affected.

    severity_ds : xr.Dataset
        Per-grid-cell aggregates: n_events, mean_csi, mean_cr,
        mean_ess_compound. Cells with fewer than 3 events are NaN-masked.
    """
    n_lat, n_lon = compound_mask.sizes['latitude'], compound_mask.sizes['longitude']
    time_coord   = compound_mask['time']

    n_ev_grid = np.zeros((n_lat, n_lon), dtype=int)
    sum_csi   = np.zeros((n_lat, n_lon))
    sum_cr    = np.zeros((n_lat, n_lon))
    sum_ess_c = np.zeros((n_lat, n_lon))
    events    = []

    for i in range(n_lat):
        for j in range(n_lon):
            mask_1d = compound_mask.isel(latitude=i, longitude=j).values
            if not np.any(mask_1d):
                continue

            labeled, n_events = label(mask_1d)
            smdi_cell = smdi.isel(latitude=i, longitude=j).values
            svdi_cell = svdi.isel(latitude=i, longitude=j).values

            for eid in range(1, n_events + 1):
                ev       = labeled == eid
                smdi_ev  = smdi_cell[ev]
                svdi_ev  = svdi_cell[ev]
                times_ev = time_coord.values[ev]

                n_total    = len(smdi_ev)
                n_compound = int(np.sum(
                    (smdi_ev < smdi_thresh) & (svdi_ev > svdi_thresh)))
                ess_d = float(np.sum(
                    np.maximum(0., np.abs(smdi_ev) - abs(smdi_thresh))))
                ess_v = float(np.sum(
                    np.maximum(0., svdi_ev - svdi_thresh)))
                csi   = float(np.sum(np.abs(smdi_ev) * svdi_ev))
                cr    = n_compound / n_total
                ess_c = float(np.sqrt(ess_d * ess_v))

                n_ev_grid[i, j] += 1
                sum_csi[i, j]   += csi
                sum_cr[i, j]    += cr
                sum_ess_c[i, j] += ess_c

                events.append({
                    'latitude'    : float(compound_mask.latitude.values[i]),
                    'longitude'   : float(compound_mask.longitude.values[j]),
                    'start_date'  : pd.Timestamp(times_ev[0]),
                    'end_date'    : pd.Timestamp(times_ev[-1]),
                    'duration'    : n_total,
                    'peak_svdi'   : float(np.nanmax(svdi_ev)),
                    'min_smdi'    : float(np.nanmin(smdi_ev)),
                    'mean_svdi'   : float(np.nanmean(svdi_ev)),
                    'mean_smdi'   : float(np.nanmean(smdi_ev)),
                    'csi'         : csi,
                    'cr'          : cr,
                    'ess_drought' : ess_d,
                    'ess_vpd'     : ess_v,
                    'ess_compound': ess_c,
                })

    events_df = pd.DataFrame(events)

    footprint = (
        events_df.groupby('start_date')['latitude']
        .count()
        .rename('n_cells_affected')
        .reset_index()
    )
    events_df = events_df.merge(footprint, on='start_date', how='left')

    with np.errstate(invalid='ignore'):
        mean_csi   = np.where(n_ev_grid >= 3, sum_csi   / n_ev_grid, np.nan)
        mean_cr    = np.where(n_ev_grid >= 3, sum_cr    / n_ev_grid, np.nan)
        mean_ess_c = np.where(n_ev_grid >= 3, sum_ess_c / n_ev_grid, np.nan)

    coords = {'latitude' : compound_mask.latitude,
               'longitude': compound_mask.longitude}
    severity_ds = xr.Dataset({
        'n_events'         : xr.DataArray(n_ev_grid,  coords=coords,
                                          dims=['latitude','longitude']),
        'mean_csi'         : xr.DataArray(mean_csi,   coords=coords,
                                          dims=['latitude','longitude']),
        'mean_cr'          : xr.DataArray(mean_cr,    coords=coords,
                                          dims=['latitude','longitude']),
        'mean_ess_compound': xr.DataArray(mean_ess_c, coords=coords,
                                          dims=['latitude','longitude']),
    })

    return events_df, severity_ds


def compute_annual_metrics(compound_mask, smdi, svdi, events_df, n_cells):
    """
    Annual intensive and extensive compound event statistics.

    Intensive (per-event averages, sensitive to localised extremes):
        n_events, integrated_severity, mean_ess, mean_csi, mean_cr

    Extensive (area-weighted totals, captures regional agricultural burden):
        total_ess, total_ess_per_cell, n_affected_cells, area_fraction

    Parameters
    ----------
    compound_mask : xr.DataArray
    smdi, svdi    : xr.DataArray
    events_df     : pd.DataFrame from characterize_compound_events
    n_cells       : int — land cells in domain/country

    Returns
    -------
    pd.DataFrame indexed by year
    """
    years  = np.unique(compound_mask.time.dt.year.values)
    ev     = events_df.copy()
    ev['year'] = pd.to_datetime(ev['start_date']).dt.year
    metrics = {}

    for year in years:
        year_mask = compound_mask.sel(time=compound_mask.time.dt.year == year)
        year_smdi = smdi.sel(time=smdi.time.dt.year == year)
        year_svdi = svdi.sel(time=svdi.time.dt.year == year)
        year_ev   = ev[ev['year'] == year]

        spatial_mean = year_mask.mean(dim=['latitude','longitude']) > 0.5
        _, n_ev      = label(spatial_mean.values)

        integ = float(
            (np.abs(year_smdi.where(year_mask, 0)) *
             year_svdi.where(year_mask, 0)).sum().mean()
        )

        total_ess = float(year_ev['ess_compound'].sum()) if len(year_ev) else 0.
        n_aff     = (year_ev[['latitude','longitude']].drop_duplicates().shape[0]
                     if len(year_ev) else 0)

        metrics[int(year)] = {
            'n_events'           : n_ev,
            'integrated_severity': integ,
            'mean_ess'           : float(year_ev['ess_compound'].mean())
                                   if len(year_ev) else np.nan,
            'mean_csi'           : float(year_ev['csi'].mean())
                                   if len(year_ev) else np.nan,
            'mean_cr'            : float(year_ev['cr'].mean())
                                   if len(year_ev) else np.nan,
            'total_ess'          : total_ess,
            'total_ess_per_cell' : total_ess / n_cells if n_cells > 0 else np.nan,
            'n_affected_cells'   : n_aff,
            'area_fraction'      : n_aff / n_cells if n_cells > 0 else np.nan,
        }

    return pd.DataFrame(metrics).T


# ── Plotting helpers 

def add_benchmarks(ax):
    """Dashed vertical lines at benchmark drought years."""
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for yr in benchmark_years:
        ax.axvline(yr, color="0.45", lw=1.1, ls="--", alpha=0.7)
        ax.text(yr + 0.2, ylim[0] + span * 0.88,
                str(yr), fontsize=8, style="italic", color="0.4")


def add_trend(ax, years_arr, values_arr, color):
    """OLS trend line; skipped if fewer than 5 valid years."""
    valid = ~np.isnan(values_arr)
    if valid.sum() < 5:
        return
    slope, intercept, _, p, _ = linregress(years_arr[valid], values_arr[valid])
    ax.plot(years_arr, slope * years_arr + intercept,
            color=color, lw=1.8, ls="--", alpha=0.85,
            label=f"Trend {slope:+.3f}/yr  p={p:.3f}")


def style_map_ax(ax, extent=None):
    """Standard cartopy styling: borders, coastline, land, gridlines."""
    ax.add_feature(cfeature.BORDERS,   lw=0.7,  edgecolor="0.25")
    ax.add_feature(cfeature.COASTLINE, lw=0.7)
    ax.add_feature(cfeature.LAND,      facecolor="0.94", zorder=0)
    if extent:
        ax.set_extent(extent, crs=PC)
    gl = ax.gridlines(crs=PC, draw_labels=True,
                      lw=0.25, color="0.6", alpha=0.5, ls="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    return gl


#%%
# ── Build compound event mask ─────────────────────────────────────────────────

drought_mask  = identify_smdi_events(smdi)
vpd_mask      = identify_vpd_events(svdi)

# Step 1: Identify with permissive thresholds
compound_mask_initial = identify_compound_events(
    drought_mask, vpd_mask,
    max_offset_days=2, max_gap=7, min_duration=3,
)

print(f"\nBefore severity filter:")
print(f"  Compound event days (spatial mean): {float(compound_mask_initial.mean()):.4f}")
print(f"  Total compound days: {int(compound_mask_initial.sum().values):,}")

# Step 2: Apply severity filter (Shan et al. 2024 approach)
compound_mask = apply_severity_filter(
    compound_mask_initial, smdi, svdi,
    smdi_threshold=SMDI_SEVERITY,
    svdi_threshold=SVDI_SEVERITY
)

print(f"\nAfter severity filter:")
print(f"  Compound event days (spatial mean): {float(compound_mask.mean()):.4f}")
print(f"  Total compound days: {int(compound_mask.sum().values):,}")
print(f"  Retention rate: {float(compound_mask.sum() / compound_mask_initial.sum() * 100):.1f}%")

#%%
# ── Characterise compound events ──────────────────────────────────────────────

events_df, severity_ds = characterize_compound_events(compound_mask, smdi, svdi)

print(f"\nTotal compound events identified: {len(events_df):,}")
print(events_df[["start_date","duration","csi","cr",
                 "ess_compound","n_cells_affected"]].describe().round(3))

#%%
# ── Basic compound event statistics ──────────────────────────────────────────

print("=" * 65)
print(f"COMPOUND DROUGHT-HEATWAVE STATISTICS  "
      f"({str(smdi.time.values[0])[:4]}–{str(smdi.time.values[-1])[:4]})")
print("=" * 65)

# Domain-wide
total_events  = len(events_df)
unique_events = events_df.drop_duplicates(subset=["start_date", "end_date"])
print(f"\n── Domain-wide ──────────────────────────────────────────────")
print(f"  Total event-cell records   : {total_events:,}")
print(f"  Unique events (by date)    : {len(unique_events):,}")
print(f"  Events per year            : {total_events / n_years:.1f}")
print(f"  Mean duration              : {events_df['duration'].mean():.1f} days")
print(f"  Median duration            : {events_df['duration'].median():.1f} days")
print(f"  Mean CSI                   : {events_df['csi'].mean():.2f}")
print(f"  Mean CR                    : {events_df['cr'].mean():.2f}")
print(f"  Mean ESS (compound)        : {events_df['ess_compound'].mean():.2f}")

# Per country
print(f"\n── By country ───────────────────────────────────────────────")
fmt = "{:<14} {:>10} {:>12} {:>12} {:>10} {:>10}"
print(fmt.format("Country", "N records", "Events/yr", "Mean dur (d)",
                 "Mean CSI", "Mean CR"))
print("  " + "-" * 63)

for country in countries:
    c_ev = events_gdf[events_gdf["country"] == country]
    if len(c_ev) == 0:
        continue
    print("  " + fmt.format(
        country,
        f"{len(c_ev):,}",
        f"{len(c_ev) / n_years:.1f}",
        f"{c_ev['duration'].mean():.1f}",
        f"{c_ev['csi'].mean():.2f}",
        f"{c_ev['cr'].mean():.2f}",
    ))

# Benchmark years
print(f"\n── Benchmark years (domain-wide) ────────────────────────────")
fmt2 = "{:<6} {:>10} {:>14} {:>12} {:>10}"
print(fmt2.format("Year", "N records", "Total ESS", "Mean dur (d)", "Mean CR"))
print("  " + "-" * 52)
for yr in sorted(benchmark_years):
    yr_ev = events_gdf[events_gdf["year"] == yr]
    if len(yr_ev) == 0:
        print(f"  {yr}   (no compound events detected)")
        continue
    print("  " + fmt2.format(
        str(yr),
        f"{len(yr_ev):,}",
        f"{yr_ev['ess_compound'].sum():.1f}",
        f"{yr_ev['duration'].mean():.1f}",
        f"{yr_ev['cr'].mean():.2f}",
    ))

print("=" * 65)

#%%
# ── Spatial join: country and NUTS-2 ──────────────────────────────────────────

events_gdf = gpd.GeoDataFrame(
    events_df.drop(columns=["index_right"], errors="ignore"),
    geometry=gpd.points_from_xy(events_df['longitude'], events_df['latitude']),
    crs="EPSG:4326",
)

# Country join
events_gdf = gpd.sjoin(
    events_gdf, countries_individual,
    how="left", predicate="within",
).rename(columns={"SOVEREIGNT": "country"})
events_gdf = events_gdf.dropna(subset=["country"])
events_gdf = events_gdf.drop(columns=["index_right"], errors="ignore")

events_gdf["year"] = pd.to_datetime(events_gdf["start_date"]).dt.year

# NUTS-2 join
events_gdf = gpd.sjoin(
    events_gdf,
    nuts_study[["NUTS_ID","NUTS_NAME","geometry"]],
    how="left", predicate="within",
).drop(columns=["index_right"], errors="ignore")
events_gdf = events_gdf.drop_duplicates(subset=["latitude","longitude","start_date"])
print(f"Events matched to NUTS-2: {events_gdf['NUTS_ID'].notna().sum():,}")

# Per-country and per-NUTS land cell counts
country_n_cells = (
    events_gdf.groupby("country")
    .apply(lambda df: df[['latitude','longitude']].drop_duplicates().shape[0])
    .to_dict()
)
nuts_n_cells = (
    events_gdf.dropna(subset=["NUTS_ID"])
    .groupby("NUTS_ID")
    .apply(lambda df: df[['latitude','longitude']].drop_duplicates().shape[0])
    .rename("n_cells")
    .reset_index()
)


#%%
# ── Annual metrics 

annual_domain = compute_annual_metrics(
    compound_mask, smdi, svdi, events_gdf, n_land_cells)

annual_by_country = {}
for country in countries:
    c_poly = countries_individual[countries_individual["SOVEREIGNT"] == country]
    c_ev   = events_gdf[events_gdf["country"] == country].copy()

    def _clip(da):
        return (da.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
                   .rio.write_crs("EPSG:4326")
                   .rio.clip(c_poly.geometry, "EPSG:4326",
                              all_touched=False, drop=False))

    annual_by_country[country] = compute_annual_metrics(
        _clip(compound_mask), _clip(smdi), _clip(svdi),
        c_ev, country_n_cells.get(country, 1)
    )

all_years = np.arange(
    int(smdi.time.dt.year.values[0]),
    int(smdi.time.dt.year.values[-1]) + 1,
)
n_years = len(all_years)


#%%
#%%
# ── PLOT 1: Region-wide summary — total ESS (bars) + 5-yr rolling mean ────────
# Both axes show the same ESS quantity — bars are annual totals, the right
# axis carries the rolling mean so it doesn't compress the bar scale.
# Total ESS = Σ ess_compound across every event × every grid cell in that year.
# This weights spatial extent AND per-event severity equally, so 2011's
# broad footprint outscores 2010's localised extreme.

fig, ax1 = plt.subplots(figsize=(14, 5))

color_bars = "#6baed6"
color_ess  = "#d62728"

ess_arr = (
    events_gdf.groupby("year")["ess_compound"]
    .sum()
    .reindex(all_years, fill_value=0)
    .values
    .astype(float)
)

rolling_ess = pd.Series(ess_arr, index=all_years).rolling(
    5, center=True, min_periods=3).mean()

ax1.bar(all_years, ess_arr, color=color_bars, alpha=0.6, width=0.7,
        label="Annual total ESS")
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Total ESS (sum across all grid cells)", fontsize=12, color=color_bars)
ax1.tick_params(axis="y", labelcolor=color_bars)
ax1.set_xlim(all_years[0] - 0.8, all_years[-1] + 0.8)

ax2 = ax1.twinx()
ax2.plot(all_years, rolling_ess.values, color=color_ess, lw=2.5,
         ls="--", label="5-yr rolling mean")
ax2.set_ylabel("5-yr rolling mean ESS", fontsize=12, color=color_ess)
ax2.tick_params(axis="y", labelcolor=color_ess)
ax2.set_ylim(ax1.get_ylim())   # keep both axes on the same scale

for yr in benchmark_years:
    ax1.axvline(yr, color="0.3", lw=1.2, ls="--", alpha=0.7)
    ax1.text(yr + 0.2, ax1.get_ylim()[1] * 0.90,
             str(yr), fontsize=8, style="italic", color="0.35")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.25)

fig.suptitle(
    f"Domain-wide compound drought-heatwave burden  "
    f"({str(smdi.time.values[0])[:4]}\u2013{str(smdi.time.values[-1])[:4]})\n"
    "Bars = annual total ESS (intensity \u00d7 extent)  \u00b7  Dashed = 5-yr rolling mean",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.show()

#%%
# ── PLOT 2: Return period + event count — whole domain, LAEA projection ───────
# RdYlGn colormap: red = low return period (frequent) / green = rare.
# This is the corrected direction — high event frequency shows as red.
# Country borders overlaid for geographic context.

lats   = severity_ds.latitude.values
lons   = severity_ds.longitude.values

freq_data = xr.where(
    severity_ds['n_events'] > 0,
    severity_ds['n_events'] / n_years,
    np.nan,
).values

freq_valid = freq_data[np.isfinite(freq_data)]
freq_vmin  = np.nanpercentile(freq_valid, 2)
freq_vmax  = np.nanpercentile(freq_valid, 98)

ev_valid = ev_data[np.isfinite(ev_data)]
ev_vmin  = np.nanpercentile(ev_valid, 2)
ev_vmax  = np.nanpercentile(ev_valid, 98)

fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         subplot_kw={"projection": LAEA})

panel_cfg = [
    (freq_data, "Mean events per year",   "YlOrRd", freq_vmin, freq_vmax, "Events / year"),
    (ev_data,   "Total compound events",  "YlOrRd", ev_vmin,   ev_vmax,   "N events"),
]
for ax, (data, title, cmap, vmin, vmax, cb_label) in zip(axes, panel_cfg):
    style_map_ax(ax, extent=DOMAIN_EXTENT)
    im = ax.pcolormesh(lons, lats, data, transform=PC,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.05, fraction=0.046, aspect=30)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)

fig.suptitle(
    f"Compound event frequency  "
    f"({str(smdi.time.values[0])[:4]}–{str(smdi.time.values[-1])[:4]})\n"
    "Left: mean events per year per grid cell  |  Right: total events  |  Cells with <3 events excluded",
    fontsize=16, fontweight="bold",
)
plt.tight_layout()
plt.show()


#%%
# ── PLOT 3: Total ESS per grid cell per year — per country, shared y-axis ─────
# total_ess_per_cell divides total ESS by the number of land cells in that
# country, making Belgium and Netherlands directly comparable to France and
# Germany despite their much smaller total area. sharey=True enforces this.

fig, axes = plt.subplots(
    len(countries), 1,
    figsize=(14, 3.5 * len(countries)),
    sharex=True, sharey=True,
)

for ax, country in zip(axes, countries):
    color  = country_colors[country]
    series = (
        annual_by_country[country]["total_ess_per_cell"]
        .reindex(all_years)
        .astype(float)
    )
    rolling = pd.Series(series.values, index=all_years).rolling(
        5, center=True, min_periods=3).mean()

    ax.bar(all_years, series.values, color=color, alpha=0.4, width=0.7)
    ax.plot(all_years, series.values, color=color, lw=1.3,
            marker="o", markersize=2.5)
    ax.plot(all_years, rolling.values, color=color, lw=2.5,
            label="5-yr rolling mean")
    add_trend(ax, all_years, series.values, color="black")
    add_benchmarks(ax)

    ax.set_ylabel("ESS per grid cell", fontsize=10)
    ax.set_title(country, fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.set_xlim(all_years[0] - 0.5, all_years[-1] + 0.5)

axes[-1].set_xlabel("Year", fontsize=11)
fig.suptitle(
    "Total ESS normalised by grid-cell count — directly comparable across countries\n"
    "Shared y-axis  |  5-yr rolling mean  |  dashed = OLS trend",
    fontsize=12, fontweight="bold", y=1.005,
)
plt.tight_layout()
plt.show()


#%%
# ── PLOT 4: Severity distributions — overlaid KDE ────────────────────────────

metrics_kde = [
    ("csi",          "CSI",                        (-1,   80)),
    ("cr",           "CR  (0=no co-occurrence)",   ( 0,    1)),
    ("ess_compound", "ESS compound",               (-0.5, 40)),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (metric, xlabel, xlim) in zip(axes, metrics_kde):
    for country in countries:
        vals  = events_gdf[events_gdf["country"] == country][metric].dropna().values
        if len(vals) < 10:
            continue
        color = country_colors[country]
        x     = np.linspace(xlim[0], xlim[1], 400)
        kde   = gaussian_kde(vals, bw_method="scott")
        ax.plot(x, kde(x), color=color, lw=2, label=country)
        ax.fill_between(x, kde(x), alpha=0.07, color=color)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(xlim)
    ax.set_title(xlabel, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-event severity distributions by country",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


#%%
# ── PLOT 5: Spatial severity maps — whole domain, LAEA ────────────────────────

severity_panels = [
    ("mean_csi",          "Mean CSI",            "YlOrRd"),
    ("mean_cr",           "Mean CR",             "RdYlGn"),
    ("mean_ess_compound", "Mean ESS (compound)", "YlOrRd"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         subplot_kw={"projection": LAEA})

for ax, (var, title, cmap) in zip(axes, severity_panels):
    data = severity_ds[var].values
    vmin = 0 if var == "mean_cr" else np.nanpercentile(data, 5)
    vmax = 1 if var == "mean_cr" else np.nanpercentile(data, 95)

    style_map_ax(ax, extent=DOMAIN_EXTENT)
    im = ax.pcolormesh(lons, lats, data, transform=PC,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.05, fraction=0.046, aspect=28)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

fig.suptitle(
    "Spatial severity distribution — mean per grid cell\n"
    f"{str(smdi.time.values[0])[:4]}-{str(smdi.time.values[-1])[:4]}  "
    "| cells with <3 events masked",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.show()


#%%
# ── PLOT 6: Metric inter-correlations — domain-wide single panel ──────────────

corr_vars   = ["duration","csi","cr","ess_compound","peak_svdi","min_smdi"]
corr_matrix = events_gdf[corr_vars].corr()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(corr_vars, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(corr_vars, fontsize=9)
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        ax.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(corr_matrix.iloc[i,j]) > 0.6 else "black")
plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
ax.set_title("Metric inter-correlations (domain-wide)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()


#%%
# ── NUTS-2 aggregation ────────────────────────────────────────────────────────

ev_nuts = events_gdf.dropna(subset=["NUTS_ID"]).copy()

# Annual totals per NUTS region
annual_nuts = (
    ev_nuts.groupby(["NUTS_ID","year"])
    .agg(
        total_ess = ("ess_compound", "sum"),
        n_events  = ("ess_compound", "count"),
        mean_ess  = ("ess_compound", "mean"),
        mean_cr   = ("cr",           "mean"),
    )
    .reset_index()
    .merge(nuts_n_cells, on="NUTS_ID", how="left")
)
annual_nuts["ess_per_cell"] = (
    annual_nuts["total_ess"] / annual_nuts["n_cells"].clip(lower=1)
)

# Whole-period aggregates per NUTS region
nuts_totals = (
    ev_nuts.groupby("NUTS_ID")
    .agg(
        n_events  = ("ess_compound", "count"),
        total_ess = ("ess_compound", "sum"),
        mean_ess  = ("ess_compound", "mean"),
        mean_cr   = ("cr",           "mean"),
        mean_dur  = ("duration",     "mean"),
        NUTS_NAME = ("NUTS_NAME",    "first"),
        CNTR_CODE = ("country",      "first"),
    )
    .reset_index()
    .merge(nuts_n_cells, on="NUTS_ID", how="left")
)
nuts_totals["events_per_cell_yr"] = (
    nuts_totals["n_events"] /
    (nuts_totals["n_cells"].clip(lower=1) * n_years)
)
nuts_totals["ess_per_cell"] = (
    nuts_totals["total_ess"] / nuts_totals["n_cells"].clip(lower=1)
)

# Merge to spatial GeoDataFrame for mapping
nuts_map = nuts_study.merge(nuts_totals, on="NUTS_ID", how="left")

print("\nTop 10 NUTS-2 regions by ESS per grid cell:")
print(
    nuts_totals.sort_values("ess_per_cell", ascending=False)
    [["NUTS_NAME","CNTR_CODE","n_events","ess_per_cell","events_per_cell_yr"]]
    .head(10).to_string(index=False)
)


#%%
# ── PLOT 7: NUTS-2 choropleth maps ────────────────────────────────────────────

nuts_panels = [
    ("ess_per_cell",       "ESS per grid cell\n(normalised intensity)", "YlOrRd"),
    ("events_per_cell_yr", "Events per cell per year\n(frequency)",     "YlOrRd"),
    ("mean_cr",            "Mean CR\n(concurrence ratio)",              "RdYlGn"),
    ("mean_dur",           "Mean event duration\n(days)",               "PuBu"),
]

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.05)

for idx, (col, title, cmap) in enumerate(nuts_panels):
    ax = fig.add_subplot(gs[idx // 2, idx % 2], projection=LAEA)
    style_map_ax(ax, extent=DOMAIN_EXTENT)

    vals = nuts_map[col].values.astype(float)
    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals) == 0:
        continue
    if col == "mean_cr":
        vmin_v, vmax_v = 0, 1
    else:
        vmin_v = np.nanpercentile(finite_vals, 5)
        vmax_v = np.nanpercentile(finite_vals, 95)

    norm     = mcolors.Normalize(vmin=vmin_v, vmax=vmax_v)
    cmap_obj = plt.get_cmap(cmap)

    for _, row in nuts_map.iterrows():
        v  = row[col]
        fc = cmap_obj(norm(v)) if np.isfinite(v) else "0.88"
        ax.add_geometries(
            [row.geometry], crs=PC,
            facecolor=fc, edgecolor="0.45", linewidth=0.3, zorder=2,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      pad=0.05, fraction=0.046, aspect=28)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)

fig.suptitle(
    f"Compound drought-heatwave characteristics by NUTS-2 region  "
    f"({str(smdi.time.values[0])[:4]}-{str(smdi.time.values[-1])[:4]})\n"
    "ESS and frequency normalised by grid-cell count",
    fontsize=12, fontweight="bold",
)
plt.show()


#%%
# ── PLOT 8: NUTS-2 annual ESS per cell — top N most affected regions ──────────

top_n    = 8
top_nuts = (
    nuts_totals.sort_values("ess_per_cell", ascending=False)
    .head(top_n)["NUTS_ID"].tolist()
)

cmap_nuts = plt.get_cmap("tab10")

fig, axes = plt.subplots(top_n, 1, figsize=(14, 3.0 * top_n), sharex=True)

for idx, (ax, nuts_id) in enumerate(zip(axes, top_nuts)):
    meta  = nuts_totals[nuts_totals["NUTS_ID"] == nuts_id].iloc[0]
    color = cmap_nuts(idx)

    series = (
        annual_nuts[annual_nuts["NUTS_ID"] == nuts_id]
        .set_index("year")["ess_per_cell"]
        .reindex(all_years)
        .astype(float)
    )
    rolling = series.rolling(5, center=True, min_periods=3).mean()

    ax.bar(all_years, series.values, color=color, alpha=0.4, width=0.7)
    ax.plot(all_years, series.values, color=color, lw=1.3,
            marker="o", markersize=2.5)
    ax.plot(all_years, rolling.values, color=color, lw=2.5,
            label="5-yr rolling mean")
    add_trend(ax, all_years, series.values, color="black")
    add_benchmarks(ax)

    ax.set_ylabel("ESS / cell", fontsize=9)
    ax.set_title(
        f"{meta['NUTS_NAME']}  ({nuts_id})  -  {meta['CNTR_CODE']}",
        fontsize=11, fontweight="bold", color=color,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(all_years[0] - 0.5, all_years[-1] + 0.5)

axes[-1].set_xlabel("Year", fontsize=11)
fig.suptitle(
    f"Annual ESS per grid cell - top {top_n} most affected NUTS-2 regions\n"
    "(ranked by total ESS per cell across full record)",
    fontsize=12, fontweight="bold", y=1.005,
)
plt.tight_layout()
plt.show()


#%%
# ── PLOT 9: NUTS-2 ranking — horizontal bar chart ────────────────────────────

ranking = (
    nuts_totals.sort_values("ess_per_cell", ascending=False)
    .reset_index(drop=True)
)
ranking.index += 1

print("\n-- NUTS-2 region ranking (top 20) --------------------------------------")
print(
    ranking.head(20)[
        ["NUTS_NAME","CNTR_CODE","n_events","ess_per_cell",
         "events_per_cell_yr","mean_cr","mean_dur"]
    ]
    .rename(columns={
        "NUTS_NAME"          : "Region",
        "CNTR_CODE"          : "Country",
        "n_events"           : "N events",
        "ess_per_cell"       : "ESS/cell",
        "events_per_cell_yr" : "Freq (cell/yr)",
        "mean_cr"            : "Mean CR",
        "mean_dur"           : "Mean dur (d)",
    })
    .round(3)
    .to_string()
)

top20      = ranking.head(20)
bar_colors = [country_colors.get(row["CNTR_CODE"], "0.5")
              for _, row in top20.iterrows()]
labels     = [f"{row['NUTS_NAME']} ({row['NUTS_ID']})"
              for _, row in top20.iterrows()]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(top20)), top20["ess_per_cell"].values,
        color=bar_colors, edgecolor="white", height=0.7)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Total ESS per grid cell", fontsize=11)
ax.set_title(
    "Top 20 NUTS-2 regions by compound drought-heatwave burden\n"
    "(ESS normalised by land cell count)",
    fontsize=12, fontweight="bold",
)
ax.grid(True, axis="x", alpha=0.3)

legend_patches = [mpatches.Patch(color=v, label=k)
                  for k, v in country_colors.items()]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.show()

#%%
# ── FIGURE: SMDI × SVDI joint scatter — NL JJA days, 1980–2024 ──────────────

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ── 1. Clip to Netherlands ────────────────────────────────────────────────────
nl_poly = countries_individual[countries_individual["SOVEREIGNT"] == "Netherlands"]

def _clip_nl(da):
    return (da
            .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            .rio.write_crs("EPSG:4326")
            .rio.clip(nl_poly.geometry, "EPSG:4326",
                      all_touched=False, drop=False))

smdi_nl = _clip_nl(smdi)
svdi_nl = _clip_nl(svdi)

# ── 2. Select JJA and compute spatial mean ────────────────────────────────────
def jja(da):
    return da.sel(time=da.time.dt.month.isin([6, 7, 8]))

smdi_jja  = jja(smdi_nl).mean(dim=["latitude", "longitude"], skipna=True)
svdi_jja  = jja(svdi_nl).mean(dim=["latitude", "longitude"], skipna=True)

smdi_vals = smdi_jja.values
svdi_vals = svdi_jja.values
valid     = np.isfinite(smdi_vals) & np.isfinite(svdi_vals)

# ── 3. Split into background and compound quadrant days ───────────────────────
in_compound_quad = (smdi_vals < SMDI_THRESH) & (svdi_vals > SVDI_THRESH) & valid
is_background    = ~in_compound_quad & valid

n_compound_days = int(in_compound_quad.sum())
n_total_days    = int(valid.sum())
quad_frac       = 100 * n_compound_days / n_total_days

# ── 4. Per-day intensity: geometric mean of threshold exceedances ─────────────
exceedance_smdi = np.abs(smdi_vals - SMDI_THRESH)   # how far below -2
exceedance_svdi =        svdi_vals - SVDI_THRESH    # how far above +1
intensity = np.where(
    in_compound_quad,
    np.sqrt(exceedance_smdi * exceedance_svdi),
    np.nan,
)
# Normalise to [0, 1] for colormap
i_vals   = intensity[in_compound_quad]
i_norm   = (i_vals - i_vals.min()) / (i_vals.max() - i_vals.min())
cmap_int = cm.get_cmap("YlOrRd")
colors_c = cmap_int(0.25 + 0.75 * i_norm)   # start at 0.25 to avoid near-white

# ── 5. Colour palette ─────────────────────────────────────────────────────────
C_TEAL        = "#0A8A64"
C_TEAL_QUAD   = "#E0F4EE"
C_NAVY        = "#1F3060"
C_GREY        = "#9BA3AE"
C_THRESH      = "#3B5BA5"

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

x_min = min(np.nanmin(smdi_vals[valid]), -3.5) - 0.3
x_max = max(np.nanmax(smdi_vals[valid]),  0.5) + 0.3
y_min = min(np.nanmin(svdi_vals[valid]), -1.5) - 0.2
y_max = max(np.nanmax(svdi_vals[valid]),  2.5) + 0.2

# Compound quadrant fill
ax.fill_betweenx(
    [SVDI_THRESH, y_max], x_min, SMDI_THRESH,
    color=C_TEAL_QUAD, alpha=0.85, zorder=0
)

# Subtle grid lines
for xv in np.arange(np.ceil(x_min), SMDI_THRESH, 1):
    ax.axvline(xv, color="0.82", lw=0.4, zorder=1)
for yv in np.arange(np.ceil(y_min), y_max, 1):
    ax.axhline(yv, color="0.82", lw=0.4, zorder=1)

# Threshold lines
ax.axvline(SMDI_THRESH, color=C_THRESH, lw=1.4, ls="--", zorder=3,
           label=f"SMDI threshold ({SMDI_THRESH:.1f})")
ax.axhline(SVDI_THRESH, color=C_THRESH, lw=1.4, ls="--", zorder=3,
           label=f"SVDI threshold (+{SVDI_THRESH:.1f})")
# Background days
ax.scatter(smdi_vals[is_background], svdi_vals[is_background],
           c=C_GREY, s=9, alpha=0.35, lw=0, zorder=2,
           label=f"Non-compound JJA days  (n = {int(is_background.sum()):,})")

# Compound days — coloured by intensity
sc = ax.scatter(smdi_vals[in_compound_quad], svdi_vals[in_compound_quad],
                c=colors_c, s=32, alpha=0.90, lw=0.4,
                edgecolors="white", zorder=1,
                label=f"Compound quadrant days  (n = {n_compound_days})")

# Colorbar for intensity
sm = plt.cm.ScalarMappable(
    cmap=cmap_int,
    norm=mcolors.Normalize(vmin=i_vals.min(), vmax=i_vals.max())
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Intensity  √(ΔSMDI × ΔSVDI)", fontsize=12, color=C_NAVY)
cbar.ax.tick_params(labelsize=8)

# Quadrant annotation
ax.text(
    SMDI_THRESH - 0.12, y_max - 0.18,
    f"CDHW\n({quad_frac:.1f}% of days)",
    ha="right", va="top", fontsize=12,zorder=10,
    color=C_TEAL, fontweight="bold", style="italic",
)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("SMDI  (soil moisture deficit index, σ)", fontsize=12, color=C_NAVY)
ax.set_ylabel("SVDI  (standardised VPD drought index, σ)", fontsize=12, color=C_NAVY)
ax.set_title(
    "Joint distribution of drought and heat stress\n"
    "Netherlands  ·  JJA days  ·  1980–2024  (ERA5 spatial mean)",
    fontsize=12, fontweight="bold", color=C_NAVY, pad=10,
)
ax.tick_params(colors=C_NAVY, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("0.75")

leg = ax.legend(fontsize=14, loc="upper right", framealpha=0.90,
                edgecolor="0.80")
for t in leg.get_texts():
    t.set_color(C_NAVY)

plt.tight_layout()
# plt.savefig("scatter_smdi_svdi_nl_jja.png", dpi=180, bbox_inches="tight")
plt.show()
print(f"Compound JJA days: {n_compound_days} / {n_total_days}  ({quad_frac:.1f}%)")

#%%
# Check actual distributions
print("SMDI statistics:")
print(f"  Range: {smdi.min().values} to {smdi.max().values}")
print(f"  Percentiles: 5th={smdi.quantile(0.05).values}, 15th={smdi.quantile(0.15).values}")
print(f"  Value -1.0 corresponds to percentile: {(smdi < -1.0).mean().values * 100}%")
print(f"  Value -1.5 corresponds to percentile: {(smdi < -1.5).mean().values * 100}%")

print("\nSVDI statistics:")
print(f"  Range: {svdi.min().values} to {svdi.max().values}")
print(f"  Percentiles: 85th={svdi.quantile(0.85).values}, 95th={svdi.quantile(0.95).values}")
print(f"  Value 1.0 corresponds to percentile: {(svdi > 1.0).mean().values * 100}%")
print(f"  Value 1.5 corresponds to percentile: {(svdi > 1.5).mean().values * 100}%")
#%%
#%%
# ── PLOT: Netherlands — annual event count + mean ESS (dual axis) ─────────────

nl_ev = events_gdf[events_gdf["country"] == "Netherlands"].copy()
nl_ev["year"] = pd.to_datetime(nl_ev["start_date"]).dt.year

n_events_arr = (
    nl_ev.groupby("year")["start_date"]
    .nunique()
    .reindex(all_years, fill_value=0)
    .values.astype(float)
)

nl_n_cells = country_n_cells.get("Netherlands", 1)

ess_arr = (
    nl_ev.groupby("year")["ess_compound"]
    .sum()
    .reindex(all_years, fill_value=0)
    .values.astype(float)
) / nl_n_cells

C_TEAL = "#0A8A64"
C_NAVY = "#1F3060"
C_RED  = "#C0392B"

fig, ax1 = plt.subplots(figsize=(14, 5))

# ── Bars: event count (left axis) ────────────────────────────────────────────
ax1.bar(all_years, n_events_arr, color=C_TEAL, alpha=0.45, width=0.7,
        label="Annual N events")
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("N compound events", fontsize=12, color=C_TEAL)
ax1.tick_params(axis="y", labelcolor=C_TEAL, labelsize=12)
ax1.tick_params(axis="x", labelsize=12)
ax1.set_xlim(all_years[0] - 0.8, all_years[-1] + 0.8)
ax1.grid(True, alpha=0.25)

# ── Line: mean ESS (right axis) ───────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.plot(all_years, ess_arr, color=C_NAVY, lw=2.2, ls="--",
         label="Mean ESS")
ax2.set_ylabel("ESS per grid cell", fontsize=12, color=C_NAVY)
ax2.tick_params(axis="y", labelcolor=C_NAVY, labelsize=12)

# ── Benchmark year lines ──────────────────────────────────────────────────────
ylim = ax1.get_ylim()
for yr in benchmark_years:
    ax1.axvline(yr, color=C_RED, lw=1.2, ls="--", alpha=0.8)
    ax1.text(yr + 0.2, ylim[0] + (ylim[1] - ylim[0]) * 0.90,
             str(yr), fontsize=12, style="italic", color=C_RED)

# ── Legend ────────────────────────────────────────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper left")

fig.suptitle(
    f"Netherlands — compound drought-heatwave events  ({all_years[0]}–{all_years[-1]})\n"
    "Bars = annual event count  ·  Dashed = total ESS normalised by land cell count",
    fontsize=16, fontweight="bold",
)
plt.tight_layout()
plt.show()
#%%
# ── Copula analysis and Joint Return Periods ──────────────────────────────────
# (continuation follows here)
# %%
