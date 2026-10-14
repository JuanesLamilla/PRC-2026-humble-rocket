# Manuscript

Manuscript in the format of the Journal of Open Aviation Science (template:
https://github.com/open-aviation/joas-template).

- `main.tex`: the manuscript (a single file, as the template requires; class file in `extra/joas.cls`).
- `reference.bib`: bibliography for biblatex/biber. Citations are numeric and link to the reference list.
- `figures/`: figures (PDF).
- `scripts/make_figures_tables.py`: recomputes the table values and data figures from the repository data
  (run from the repository root). `scripts/method_overview.dot` draws Figure 2 with Graphviz.
  `scripts/out/` holds the computed values quoted in the paper.
- `PROVENANCE.md`: the source of each number in the paper.

Build (TeX Live with biber, libertine, newtx, sourcesanspro, inconsolata):

```bash
cd manuscript
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
```

`Makefile.joas` is the template's Makefile; rename it to `Makefile` to use `make pdf`.
