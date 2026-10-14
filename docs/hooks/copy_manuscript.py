"""MkDocs hook: copy manuscript/main.pdf into the docs assets before each build.

The PDF is built from manuscript/main.tex (see manuscript/README.md); the copy
under docs/assets/ is generated and gitignored.
"""

import logging
import shutil
from pathlib import Path

log = logging.getLogger("mkdocs")


def on_pre_build(config):
    docs = Path(config["docs_dir"])
    src = docs.parent / "manuscript" / "main.pdf"
    dst = docs / "assets" / "manuscript.pdf"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    else:
        log.warning("manuscript/main.pdf not found; build it first (see manuscript/README.md)")
