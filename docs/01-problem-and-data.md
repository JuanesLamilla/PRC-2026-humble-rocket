# 1. Problem and data

## 1.1 Task

The PRC Data Challenge 2026 asks for the taxi-out time of every departure at ten European
airports in January and July 2026, given all 2025 movements as training data.

```text
TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt
                   (take-off)          (off-block, as recorded by the airport)
```

| | |
|---|---|
| Airports | EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM |
| Training | 2,085,047 departures in 2025 (4.17 M movements incl. arrivals) |
| Ranking set | 344,841 departures in January and July 2026 |
| Metric | RMSE in seconds over the ranking set |

The ranking set contains every field of the departure except the off-block time and the label:
take-off time, runway, stand, schedule, aircraft, and the Network Manager (NM) flight-plan
fields, including `AOBT_3_flt`, NM's off-block time. Arrivals in the test months are complete,
labels included. The organizers allow all of these fields as inputs.

Two column families matter throughout:

- `_mvt` columns come from the airport's movement record. Every departure has them.
- `_flt` columns come from the NM flight plan, joined by the organizers. 1.08 % of 2025
  departures and 1.53 % of the ranking departures have no flight-plan record; on those rows every
  `_flt` column is null. This document calls them the **no-flight-plan** rows.

## 1.2 What the label measures

The organizers score the off-block stamp as the airports reported it, without cleaning. The
2025 labels show that this stamp is not always a physical off-block time.

| Recording pattern | How it shows in the data | Frequency in 2025 |
|---|---|---|
| Schedule default | off-block stamp equals the scheduled time within 10 s, so the label equals `C = MVT − SCHED` | 4.80 % of departures; 18.1 % at LIRF |
| Day rollover | off-block stamp one day early; label ≈ 86,400 s | 15 departures (13 at LIRF) |
| Early stamp | label above 1 h on a row with an NM off-block time; in a group of ~630 such rows the airport stamp precedes NM's by a median 3,055 s | 0.10 % (2,048 rows) |
| NM proxy on a derived value | `AOBT_3_flt` is take-off minus a per-airport constant (LTFM ≈ 1,200 s and 1,500 s; LIRF ≈ 595 s; EDDM ≈ 300 s; EGLL ≈ 600 s), not a measurement | 42 % of LTFM rows in 2025 |

The arrival records of January and July 2026 carry the same patterns at the same rates (for
example LIRF in-block equal to schedule: 8.6 % and 10.6 % in 2026 against 8.7 % and 8.9 % in
2025). The ranking labels therefore contain the same mixture.

**Consequence.** Under RMSE the best prediction is the conditional mean of the recorded label,
not of the physical taxi time. A model has to predict the probability of each recording pattern
and the value the label takes under it. Section 2 describes how the model does this.

## 1.3 Where the error is

Squared error is concentrated in a few rows. Shares of the final model's 2025 out-of-fold squared
error ([`docs/03-validation.md`](03-validation.md) describes the protocol):

| Rows | Count | Share of rows | Share of squared error | RMSE (s) |
|---|---:|---:|---:|---:|
| Day rollovers | 15 | < 0.01 % | 28.8 % | 46,534 |
| No flight plan (rollovers excluded) | 22,456 | 1.08 % | 15.6 % | 887 |
| With flight plan, C ≥ 1 h | 179,947 | 8.63 % | 17.0 % | 327 |
| With flight plan, C < 1 h, label > 1 h | 115 | 0.01 % | 0.4 % | 1,901 |
| Everything else | 1,882,514 | 90.29 % | 38.2 % | 151 |

`C = MVT − SCHED` is the delay of the take-off against the schedule. It is known on every test
row, and it is the value a schedule-default label takes. Rows with large C carry most of the
label mixture.

The 15 rollover rows decide much of any pooled comparison. Every comparison in this repository is
therefore reported on all rows and with the rollover rows removed (`ex_ro`).

## 1.4 Differences between 2025 and 2026

| Difference | Evidence | Effect |
|---|---|---|
| LIRF has almost no ADS-B ground coverage in July 2026 | share of LIRF departures with an ADS-B ground track: 0.956 in July 2025, 0.002 in July 2026 (0.80 in January 2026) | members trained with ADS-B features see an input state at LIRF that 2025 barely contains; handled by routing ([`docs/02-method.md`](02-method.md) §2.4) |
| EHAM no-flight-plan rate | 1.6 % in 2025, 5.0 % in January 2026 | more rows in the hardest group |
| LTFM derived-AOBT share | 42 % in 2025, 54 % in July 2026 | more rows where `AOBT_3_flt` is not a measurement |
| ADS-B archive gaps in early 2025 | the ADSB.lol heatmap files for 2025-01-01 to 2025-04-20 drop ground tracks at EGLL, EDDF and EHAM | those 110 days are rebuilt from full traces ([`docs/05-reproduction.md`](05-reproduction.md)) |

## 1.5 Data used

The challenge tables plus open external sources: ADS-B ground tracks (ADSB.lol), METAR
observations (Iowa Environmental Mesonet), Open-Meteo reanalysis, OpenStreetMap aeroway
geometry, X-Plane Scenery Gateway stand data and OurAirports runway data. Licences, fetch
scripts and attributions: [`docs/data-sources.md`](data-sources.md).
