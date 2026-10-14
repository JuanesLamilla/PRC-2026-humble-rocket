# Data sources

The prize conditions require every external dataset to be openly accessible, openly licensed and
documented. This file lists every source the final model reads, how it is fetched, and the
attribution each licence asks for. No data file is stored in this repository; every archive is
re-created by the script named in its row, with a SHA-256 manifest.

## Challenge data

| | |
|---|---|
| Source | PRC Data Challenge 2026 organizers (EUROCONTROL Performance Review Commission) |
| Content | all 2025 movements at the ten airports (training, labelled), January and July 2026 (ranking set, departure labels withheld), submission template |
| Access | organizer-issued keys; `mc mirror` into `data/raw/` (see [`docs/05-reproduction.md`](05-reproduction.md)) |
| Terms | the organizers' terms; not redistributed here |

## External sources read by the final model

| Source | Used for | Fetch | Licence | Attribution |
|---|---|---|---|---|
| ADSB.lol globe history, daily releases 2025-01 to 2026-07 ([2025](https://github.com/adsblol/globe_history_2025), [2026](https://github.com/adsblol/globe_history_2026)), filtered to 12 km around each airport | ground-track features `adsb_ground`, `adsb_ground_v2`; the routing flag of the no-ADS-B experts | `scripts/adsb_build.py`, `scripts/adsb_v2.py archive` | ODbL 1.0 | "Contains data from ADSB.lol, made available under the Open Database License (ODbL)." |
| OurAirports [`runways.csv`, `airports.csv`](https://ourairports.com/data/) | runway-threshold coordinates for the ADS-B distance features; airport reference points | `scripts/fetch_external.py ourairports` | public domain | none required; credited |
| Iowa Environmental Mesonet [ASOS/METAR archive](https://mesonet.agron.iastate.edu/ASOS/) for the ten ICAO stations | visibility, ceiling, low-visibility and frozen-precipitation features (`metar_visibility`, `winter_ops`) | `scripts/fetch_external.py metar --years 2025 2026` | US public-sector data (upstream NOAA Integrated Surface Database) | "Iowa Environmental Mesonet, Iowa State University; upstream NOAA ISD" |
| OpenStreetMap `aeroway` geometry via the [Overpass API](https://overpass-api.de/) | stand-to-threshold taxiway distance, detour ratio, runway crossings (`taxi_distance`) | `scripts/fetch_external.py osm` | ODbL 1.0 | "© OpenStreetMap contributors", https://www.openstreetmap.org/copyright |
| [X-Plane Scenery Gateway](https://gateway.x-plane.com/) `apt.dat`, recommended pack per airport | stand size class, stand type, parked heading, stand-to-threshold distance (`stand_xplane`) | `scripts/fetch_external.py xplane` | GPL-2.0-or-later | "X-Plane Scenery Gateway, Laminar Research and the scenery authors" |
| [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api), hourly at the ten airport coordinates | weather inputs of the stamp-mixture post-processor (`scripts/mixture_v2.py`) | `scripts/fetch_external.py openmeteo` | CC BY 4.0 | "Weather data by Open-Meteo.com" |
| Airport websites (EDDF, EDDM, EGLL, LSZH) | night-curfew hours, hard-coded in `src/prc/features/daylight_curfew.py` with the source URLs | none (constants) | published operating facts | sources cited in the module |

Airport coordinates for solar position are hard-coded from OurAirports. Solar position is
computed with the NOAA low-precision algorithm.

The ADS-B source was checked with the organizers before use: open trajectory data is allowed when
declared (organizer clarification on the challenge forum, 2026-09-17). OpenSky Network data beyond
the challenge dataset (Trino, raw ADS-B, state vectors), which the organizers ruled out, is not
used by any feature block or build script.

## Derived data

`scripts/adsb_v2.py archive` learns a stand-coordinate table from the ADS-B tracks. It is a
derived database of ADSB.lol data and stays under ODbL 1.0 if redistributed. It is not
redistributed here; the script rebuilds it.
