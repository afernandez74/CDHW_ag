#!/usr/bin/env python3
"""
test_download.py

Quick visual verification of a downloaded ERA5 NetCDF file.
Checks spatial extent, temporal coverage, and data values.

Usage:
    python test_download.py
"""
#%%
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#%% ── Load file ───────────────────────────────────────────────────────────────

VAR  = "volumetric_soil_water_layer_1"   # variable to inspect
YEAR = 2020                               # year of the latest test download

# Resolve output directory from environment variable — same as the download script
ERA5_dat = os.environ.get("ERA5_dat", "./era5_output")
TEST_DIR = Path(ERA5_dat + "_test")

file_path = TEST_DIR / f"era5_{YEAR}.nc"
print(f"Loading: {file_path}")
print(f"File size: {file_path.stat().st_size / 1e6:.1f} MB\n")

ds = xr.open_dataset(file_path)
ds = ds.assign_coords(
    longitude=((ds.longitude + 180) % 360) - 180
).sortby("longitude")
da = ds[VAR]

# Detect time dimension name (ARCO-ERA5 uses 'time', older files may use 'valid_time')
time_dim = "time" if "time" in da.dims else "valid_time"

#%% ── Print summary ───────────────────────────────────────────────────────────

print(ds)
print(f"\nVariable     : {VAR}")
print(f"Latitude     : {float(da.latitude.min()):.2f} – {float(da.latitude.max()):.2f}  (expected 41.0 – 55.0)")
print(f"Longitude    : {float(da.longitude.min()):.2f} – {float(da.longitude.max()):.2f}  (expected 355.0 – 16.0)")
print(f"Time start   : {str(da[time_dim].values[0])[:16]}")
print(f"Time end     : {str(da[time_dim].values[-1])[:16]}")
print(f"N timesteps  : {da.sizes[time_dim]}  (expected {365*24} or {366*24})")
print(f"Units        : {da.attrs.get('units', 'not set')}")
print(f"Value range  : {float(da.min()):.4f} – {float(da.max()):.4f} {da.attrs.get('units', '')}")

#%% ── Plots ───────────────────────────────────────────────────────────────────

label = da.attrs.get("units", "m³/m³")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"ERA5 {VAR} — {YEAR} download verification", fontsize=12, fontweight="bold")

# ── Plot 1: Annual mean map ───────────────────────────────────────────────────
ax = axes[0, 0]
annual_mean = da.mean(dim=time_dim)
im = ax.pcolormesh(
    annual_mean.longitude, annual_mean.latitude,
    annual_mean.values, cmap="YlGnBu", shading="auto"
)
plt.colorbar(im, ax=ax, label=label)
ax.set_title("Annual mean")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

# ── Plot 2: Single timestep snapshot ─────────────────────────────────────────
ax = axes[0, 1]
snapshot = da.isel({time_dim: 12})   # noon on Jan 1
im = ax.pcolormesh(
    snapshot.longitude, snapshot.latitude,
    snapshot.values, cmap="YlGnBu", shading="auto"
)
plt.colorbar(im, ax=ax, label=label)
ax.set_title(f"Single timestep: {str(snapshot[time_dim].values)[:13]}")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

# ── Plot 3: Hourly time series for central grid cell ──────────────────────────
ax = axes[1, 0]
lat_idx = da.sizes["latitude"] // 2
lon_idx = da.sizes["longitude"] // 2
central = da.isel(latitude=lat_idx, longitude=lon_idx)

ax.plot(np.arange(len(central)), central.values,
        linewidth=0.4, alpha=0.7, color="steelblue")
ax.set_title(f"Hourly time series — central cell\n"
             f"({float(central.latitude):.2f}°N, {float(central.longitude):.2f}°E)")
ax.set_xlabel("Hour of year")
ax.set_ylabel(f"{VAR}\n({label})")
ax.grid(True, alpha=0.3)

# ── Plot 4: Monthly mean cycle ────────────────────────────────────────────────
ax = axes[1, 1]
monthly = central.resample({time_dim: "1ME"}).mean()
ax.bar(np.arange(1, len(monthly) + 1), monthly.values,
       color="steelblue", edgecolor="navy", alpha=0.8)
ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_title("Monthly mean cycle — central cell")
ax.set_ylabel(f"{VAR}\n({label})")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_fig = f"test_{VAR}_{YEAR}.png"
# plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to {out_fig}")
# %%