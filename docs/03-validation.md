# 3. Validation

The public leaderboard scores one fixed set of 344,841 rows. Its result was used only to check
whole models. Every decision about a component was made on 2025 data, with the rules below
written before the run.

## 3.1 Folds

| Scheme | Training | Validation | Use |
|---|---|---|---|
| `month_loo` | 11 months of 2025 | the held-out month; 12 folds | main estimate: every 2025 row gets one out-of-fold prediction |
| `tail_nov_dec` | January to October 2025 | November and December 2025 | forward check: training strictly before validation |
| `month_jan_jul` | 2025 without January and July | January and July 2025 | early development and screening; the two months the test months correspond to |

Random K-fold splits are never used as evidence: they leak within-month conditions (weather,
runway configuration, recording behaviour) between training and validation.

Post-processors (lane head, mixture, stacker) are fitted leave-one-month-out on the members'
out-of-fold predictions, with their own hyper-parameters chosen inside the training months.
Their test-side models are fitted on all twelve months.

## 3.2 Paired comparisons

A change is always compared against a twin: the same configuration without the change, same seeds,
same folds. `scripts/paired_delta.py` reports the RMSE difference on each row group.

Intervals come from a **day-block bootstrap**: blocks are airport × UTC date, resampled within
airport-month, 2,000 replicates. Errors within a day are strongly correlated (weather, runway
configuration, one bad recording day at one airport). The spread between seeds understates the
uncertainty by a factor of 10 to 100 and is not used as an interval.

Every comparison is reported on:

| Row group | Why |
|---|---|
| `all` | the objective |
| `ex_ro` | all rows without the 15 day rollovers, which hold 29 % of the final model's squared error |
| January, July (`ex_ro`) | the seasons of the two test months |
| each month | consistency: how many of 12 months improve |
| `ordinary` | rows with a flight-plan record, off the derived-AOBT value, label ≤ 1 h |
| no flight plan | the rows with the largest per-row error |
| `C < 1 h`, `C ≥ 1 h` | label-free split on delay against schedule ([`docs/01-problem-and-data.md`](01-problem-and-data.md) §1.3) |
| each airport | no airport may lose more than 5 s without a stated reason |

## 3.3 Decision rules

Each experiment declares its class in its configuration before it runs. The class fixes the number
it is judged on; it is not changed after the result is visible.

| Class | What it changes | Adopted when |
|---|---|---|
| ordinary | features, encodings, hyper-parameters, members | `ex_ro` interval below zero; January and July not worse (point estimate ≤ +0.5 s, interval not wholly above zero); ≥ 8 of 12 months better; forward check not worse |
| tail | anything that moves under 1 % of rows by more than 1,000 s | `all` interval below zero; the `ex_ro` cost stated in advance and not exceeded; expected effect on the test set computed from a shrunk incidence prior, with a bounded loss if the pattern does not occur |
| combination | member set, stack weights | nested fit; all-row and `ex_ro` reported for the year, January and July |

A gain concentrated in fewer than 100 rows is treated as a tail change. A new feature block must
also pass a drift check: no column shifts by more than 0.25 standard deviations between 2025 and
2026 January/July without an explanation.

## 3.4 How often the rule accepts noise

The rule was calibrated on null comparisons (A/A tests): pairs that differ only by seed, and
pairs of identical mixture fits that differ only by random initialisation.

| Null source | Comparisons | Accepted by "point estimate < 0 and ≥ 7 of 12 months" | Accepted by "interval below zero" |
|---|---:|---:|---:|
| member seeds | 60 | 37 % | 7 % |
| mixture refits | 20 | 35 % | 5 % |

A point-estimate rule accepts about a third of null changes. The interval rule accepts 5 to 7 %. All decisions use the interval.

## 3.5 Confirmation and the winner's curse

When several variants are screened and the best is kept, its screened gain is biased upward. A
variant chosen on one seed is therefore re-run with fresh seeds, and only the fresh-seed result
counts. Example from one selection round: a mixture setting chosen from a
grid improved one member on `C ≥ 1 h` rows by 3.76 s [−6.95, −1.00]; with fresh seeds, across all
seven members and through the stacker, it was 2.32 s worse [−0.37, +5.20] than the final model on
those rows. It was not adopted.

## 3.6 Hyper-parameter search

Hyper-parameters were searched only where a twin comparison could confirm the result. The search
(Optuna TPE) ran on January and July 2025 with one seed and the member's ordinary-row RMSE as the
objective. The selected configuration was then run on `month_loo` with three new seeds and judged
on the ten months the search never saw. Two searches passed this (RealMLP `exp0268`, GRU
`exp0271`); the set-attention search did not.

## 3.7 Changes to the input archives

The ADS-B archive for 2025-01-01 to 2025-04-20 was rebuilt from full traces partway through the
work ([`docs/05-reproduction.md`](05-reproduction.md)). For a comparison between members trained on the two archive
states, the validation features of January to April differ between the two sides. Such comparisons
are read on May to December 2025 only, where the evaluated rows are byte-identical.

## 3.8 Validation and the test set

The leaderboard result was used as a check on whole candidate models. Component decisions were
made on 2025 data. 2025 validation tracks the leaderboard closely for rows with `C < 1 h`.
It does not cover one 2026 population: delayed LIRF departures in July 2026 without ADS-B ground
coverage. This is why the final model is gated on C ([`docs/02-method.md`](02-method.md) §2.8).
