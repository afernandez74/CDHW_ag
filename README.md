# Compound Hot–Drought Risk Assessment in The Netherlands

## Project Overview

A Python-based framework to identify, characterize, and quantify compound hot–drought events in The Netherlands using ERA5 reanalysis data, with a focus on agricultural hazard exposure for wheat and potato crops during their respective growing seasons.

## Data Requirements

- ERA5-Land hourly reanalysis (temperature, dewpoint, soil moisture) from 1950 to present, accessed via the Copernicus Climate Data Store.
- Pre-computed hourly Soil Moisture Deficit Index (SMDI) derived from ERA5-Land.
- National and provincial crop yield statistics for wheat and potatoes from CBS (Statistics Netherlands).

## Expected Outputs

- Time series and trend analysis of compound event frequency, duration, and intensity over the ERA5 record.
- Copula-based joint probability estimates and return periods for extreme compound events.
- Spatial maps of compound event characteristics and soil–atmosphere dependence strength across the Netherlands.
- Agricultural hazard exposure metrics aligned with crop-sensitive phenological windows.

