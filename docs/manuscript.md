# Manuscript

**Recording-Aware Ensembles for Taxi-Out Time Estimation at Ten European Airports**

Juan Esteban Lamilla Cuellar, Michael Urbiztondo, Abhinav Parbhakar and
Charles Ancheta (equal contribution). Draft prepared for the
[Journal of Open Aviation Science](https://journals.open.tudelft.nl/joas/).

!!! warning "Status"
    The draft has not been submitted or peer reviewed and may change before
    submission. Please do not cite it as a published result.

<p style="margin: 1em 0;">
  <a class="md-button md-button--primary" href="../assets/manuscript.pdf">Download the PDF</a>
</p>

## Abstract

Taxi-out time, the interval between an aircraft leaving its stand and taking
off, drives fuel burn on the ground and enters the take-off times on which
network planning relies. Many estimation studies treat the recorded off-block
time as a physical event and validate on random splits of short,
single-airport samples. For 2,085,047 departures at ten large European
airports in 2025, we show that the recorded label has exploitable structure.
Flight-plan records appear to have been attached to movements only when the
recorded off-block stamp lay within 3,606 s of the last flight-plan off-block
time, a tolerance we infer from the data, so every matched departure carries a
provable interval for its label, and a missing flight plan is itself
informative. Off-block stamps coincide with the scheduled time on 4.8% of
departures; at one airport (Rome Fiumicino) this happens on 18.1% of
departures against 2.5% one minute either side, which indicates a recording
artefact. Because the error measure is computed on the recorded label, we
model it as a mixture of recording classes. The members are two-stage
gradient-boosted trees, a distributional transformer whose histogram is masked
to the feasible interval, and a class-structured combiner whose experts
include an earlier stack of tabular, sequence and set models. A per-lane
non-negative blend and label-free projections complete the estimator.
Components are selected with month-held, nested cross-validation and, for
ordinary changes, day-block bootstrap intervals under decision rules
calibrated by null comparisons. On held-out months of 2025 the blend reaches
an RMSE of 191.1 s (95% interval 183.5 to 201.7 s) when 15 day-rollover
records are excluded, and 227.4 s on all records. Error concentrates in
departures without a flight plan and long delays. We also describe a
label-free check for input drift and a consistency requirement for the inputs
of stacked models.

## Read it here

<object data="../assets/manuscript.pdf" type="application/pdf" width="100%" height="900">
  <p>Your browser does not display embedded PDFs.
     <a href="../assets/manuscript.pdf">Download the manuscript</a> instead.</p>
</object>

## Source and provenance

The LaTeX source lives in the repository's `manuscript/` folder, in the
journal's template format:

- `main.tex` and `reference.bib`: the manuscript and its bibliography
  (biblatex/biber, numeric citations).
- `figures/`: the figures as PDF.
- `scripts/make_figures_tables.py`: recomputes every table value and data
  figure from the repository data (run from the repository root);
  `scripts/method_overview.dot` draws the method-overview figure with
  Graphviz.
- `PROVENANCE.md`: the source of each number quoted in the paper.

Build with TeX Live (biber, libertine, newtx, sourcesanspro, inconsolata):

```bash
cd manuscript
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
```

The PDF shown on this page is copied from `manuscript/main.pdf` at
documentation build time, so the site always serves the latest built draft.
