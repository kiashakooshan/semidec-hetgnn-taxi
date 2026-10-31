> EXTERNAL DOCUMENT written by a third-party assistant. Its claims are NOT accepted until `scripts/validate_station_coordinates.py` passes (see docs/ASSUMPTIONS.md A-29).

# Recovered coordinates for the 266 CCRNN taxi stations (and 250 Citi Bike stations)

Date: 2026-10-30. This document explains how the lat/lon of the 266 virtual taxi stations
of the CCRNN/CGC dataset (the data behind taxi_data.h5 / all_graph.h5) were recovered,
with validation numbers, and what the delivered files contain.

## TL;DR
The 266 station coordinates were never published anywhere (checked: CGC repo, CCRNN
mirrors, FlashST-DATA on Hugging Face, PM-DMNet Data.zip — none contain coordinates).
They were **recovered exactly** (median error ~5 m) by inverting the Gaussian-kernel
similarity matrices in `all_graph.h5` and anchoring to the real coordinates of 250
Citi Bike stations identified by exact time-series matching.

## Files
1. `taxi_266_station_coordinates.csv` — one row per taxi station:
   `station_id` (CCRNN id 1..266, matches column order of `taxi_data.h5`),
   `lat`, `lon`, `x_km`, `y_km` (equirectangular km about the bike-station centroid
   (40.752, -73.976)), `n_bike_anchors` (how many distance constraints anchored it).
2. `bike_250_station_mapping.csv` — `h5_column` -> real Citi Bike `real_station_id`,
   `station_name`, `lat`, `lon`, `match_corr` (all exactly 1.0000).

## Method
1. **Bike stations are real.** The CGC bike data (bike_data.h5, 250 stations) comes from
   Citi Bike. Downloaded the official Citi Bike trip records for Apr 1 - Jun 30, 2016
   (2016-citibike-tripdata.zip, 435 MB) and built per-station 30-min pickup series for
   all 484 stations. Correlation-matched each of the 250 h5 columns against all 484 real
   series: **all 250 matched with correlation = 1.0000 and zero collisions**, yielding
   exact station identities and coordinates.
2. **The similarity matrices encode exact Euclidean km.** `all_graph.h5` keys `dis_tt`,
   `dis_bb`, `dis_tb` are Gaussian kernels `w = exp(-d^2)` (sigma = 1, d in km) with a
   2 km cutoff. Verified on bike pairs: with matched real coordinates, the estimate of
   sigma is 1.00 +- 0.00 over 19,756 pairs (correlation 0.99998, residual std 0.00).
   Hence `d = sqrt(-ln w)` recovers exact km distances for all nonzero entries.
3. **Trilateration.** Each taxi station was positioned by nonlinear least squares on its
   bike-anchor distances (`dis_tb` rows, median 75 anchors per station) and, if needed,
   its taxi-taxi distances. Results: trilateration RMS **median 0.005 km**; consistency
   with all 19,596 `dis_tt` entries: RMS **0.004 km**, max error 0.013 km; pairs with
   `dis_tt == 0` are all >= 1.99 km apart (the 2 km cutoff), confirming the model.

## Validation summary
| Check | Result |
|---|---|
| Bike matching (250 stations) | corr = 1.0000, 0 collisions |
| sigma on 19,756 bike pairs | 1.000 +- 0.000 |
| Trilateration RMS | 5 m (median) |
| dis_tt consistency (19,596 pairs) | RMS 4 m, max 13 m |
| Station extent | 6.77 x 12.07 km, lat [40.686, 40.795], lon [-74.017, -73.937] |
| vs paper's stated region | 8.42 x 14.45 km rectangle — station cloud fits inside |

Geographically the cloud is Manhattan below ~96th St + Downtown Brooklyn + a sliver of
Long Island City, which matches the CCRNN paper's description (Manhattan + part of
Brooklyn, 8.42 x 14.45 km).

## Caveats
- Station 0 (CCRNN id 1, Downtown Brooklyn, near BAM) has only 2 usable constraints
  (1 bike anchor 77 m away, 1 taxi neighbor 1.34 km away). Its two mirror solutions are
  ~130 m apart; the delivered one satisfies both constraints (RMS ~ 0). Treat its
  position as accurate to ~100 m.
- Coordinates are cluster centroids of 2016 taxi pickups; they are "virtual stations",
  not fixed infrastructure.
- `x_km/y_km` use a flat-earth approximation about (40.752, -73.976); for the CLN
  coverage / delay experiments the km frame is the right one to use directly.

## Usage in this project (semidec-hetgnn-taxi)
- Path A (station-level experiments, D12): use `taxi_266_station_coordinates.csv` for
  CLN assignment (Algorithm 2 / grid_assignment), delay experiments (Fig. 7-11),
  proximity edges A_p, and visualization. The `x_km/y_km` columns plug straight into
  `src/semidec/partition/*`.
- The 2 km kernel cutoff means `dis_tt` is exactly "stations within 2 km" — useful if
  you want to rebuild A_p from data instead of coordinates.
- Path B (TLC zones) remains untouched; these coordinates simply unblock the
  station-level branch.
