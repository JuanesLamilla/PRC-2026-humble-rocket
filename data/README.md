# Data

Nothing in this folder is committed to git except this file. To reproduce the
model, download each source below into the indicated location, then follow the
reproduction steps in the root `README.md`. The team feature blocks of
`src/prc` have their own data story: `docs/data-sources.md` and
`docs/05-reproduction.md`.

## Layout

```
data/
├── raw/        challenge movement tables + the EUROCONTROL ATFM day table
└── external/   ADS-B ground-track feature files + the ATFM workbook
```

## 1. Challenge data (required) -> `data/raw/`

The movement tables of the PRC Data Challenge 2026, provided by the organizers
to registered teams through their MinIO endpoint:

- `training_2025-01-01_2025-02-01.parquet` … `training_2025-12-01_2026-01-01.parquet`
  (twelve months of labelled 2025 movements)
- `ranking.parquet` (the initial evaluation set: January and July 2026; the
  primary members' test target)
- `final_ranking.parquet` (the full submission set: January, February, June,
  July 2026)
- `final_submitting.parquet` (submission template)

Registration and access instructions: <https://prc-data-challenge-2026.netlify.app/data.html>.
The data is provided under the challenge's own terms; it is not redistributed here.

## 2. EUROCONTROL airport arrival ATFM delay (required) -> `data/external/`

Daily arrival-regulation delay per airport, published by the EUROCONTROL
Performance Review Unit:

- <https://www.eurocontrol.int/performance/data/download/xls/Airport_Arrival_ATFM_Delay.xlsx>
  (portal: <https://ansperformance.eu/data/>)

Place the workbook at `data/external/Airport_Arrival_ATFM_Delay.xlsx`; stage 0
converts it to `data/raw/apt_arr_atfm.parquet` (challenge airports only). The
portal publishes the file without an explicit licence text; we use it as open
data, as the organizers allow for declared EUROCONTROL sources. Our snapshot
was downloaded on 2026-10-05.

## 3. ADS-B ground-track features (required) -> `data/external/`

Two parquet files with per-departure ground-movement features:

- `adsb_ground.parquet`: time from first observed ground movement to take-off,
  ground path length, stop and hold durations, mean ground speed, distance of
  the first sighting to the runway threshold, coverage flags;
- `adsb_ground_v2.parquet`: stand-anchored context, meaning the pushback
  signature (duration, path, displacement), pre-movement stationary time, and
  distance of the first point to the stand.

They are derived from the ADSB.lol global history releases
(<https://github.com/adsblol/globe_history_2025>,
<https://github.com/adsblol/globe_history_2026>), filtered to within 12 km of
the ten challenge airports, with runway thresholds from OurAirports
(<https://ourairports.com/data/>, public domain). The files hold derived
per-flight features only, never positions or tracks.

*Contains data from ADSB.lol, made available under the Open Database License
(ODbL) 1.0.*

## 4. Strike and disruption calendar (included)

`assets/strike_calendar.csv` is committed with the code: a hand-compiled
calendar of strikes and major disruptions at the ten airports (January 2025 -
July 2026), from public news reports. Each row carries date, airports, event
type and local start/end where reported.
