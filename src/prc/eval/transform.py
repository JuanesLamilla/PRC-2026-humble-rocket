"""The target transform and its inverse, in one place, with a negative-label guard.

``TAXITIME_SEC_mvt`` is a difference of two timestamps and it really does go negative: the
2026-09-19 code review counted **369 rows with ``y < 0``** (minimum −12 s) and 19 with
``y == 0`` out of 2,085,047. What each transform did with them before this module existed:

* ``none``  — nothing. Untouched here.
* ``sqrt``  — already clipped them with ``np.clip(y, 0, None)``. Untouched here, bit for bit;
  the clip is now merely *counted* instead of silent.
* ``log1p`` — ``np.log1p(-12)`` is ``nan`` and ``np.log1p(-1)`` is ``-inf``. Those went
  straight into ``xgb.train`` as labels, which is what killed the wave-5 log-target cell.
  ``exp0090_xgb_r5_log_target`` was simply not runnable.

So the guard is a clip at 0 applied to the **label only**, for the transform only, and it is
exactly what ``sqrt`` already did — which is why turning it on changes no registered result:
``none`` is the whole frontier and ``sqrt`` is byte-identical. It makes ``log1p`` possible.

A clip, not a shift. A shift (``log1p(y + c)``) would need ``c`` carried into
:func:`inverse`, and every stored OOF and every blend weight would then depend on a constant
that is not in the config digest. 369 rows out of 2.1 M, all within 12 s of zero, are not
worth that; the taxi time of a departure that left before it arrived is noise either way.
The count is logged so the assumption stays visible.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("prc.transform")

KINDS = ("none", "log1p", "sqrt")

#: kinds already reported in this process, so a per-fold/per-seed call does not log 20 times
_REPORTED: set[str] = set()


def n_negative(y: np.ndarray) -> int:
    """How many labels the transform would have to clip."""
    return int(np.count_nonzero(np.asarray(y) < 0.0))


def transform(y: np.ndarray, kind: str, tag: str = "") -> np.ndarray:
    """Forward transform, with negative labels clipped to 0 for ``log1p`` / ``sqrt``."""
    if kind == "none":  # evaluated lazily: log1p of a residual target is a spurious warning
        return y
    if kind not in KINDS:
        raise KeyError(f"unknown target transform {kind!r}; known: {KINDS}")
    y = np.asarray(y)
    n = n_negative(y)
    if n:
        if kind not in _REPORTED:
            _REPORTED.add(kind)
            log.warning(
                "target transform %s: clipping %d label(s) < 0 to 0 (min %.3f)%s; "
                "log1p would otherwise produce nan/-inf labels and sqrt already clipped "
                "them silently. Further occurrences at debug level.",
                kind,
                n,
                float(np.min(y)),
                f" [{tag}]" if tag else "",
            )
        else:
            log.debug("target transform %s: clipped %d label(s) < 0 %s", kind, n, tag)
        y = np.clip(y, 0.0, None)
    return {"log1p": np.log1p(y), "sqrt": np.sqrt(y)}[kind]


def inverse(p: np.ndarray, kind: str) -> np.ndarray:
    """Back-transform a prediction. The clip above is not undone: it applied to the label."""
    if kind == "none":
        return p
    if kind not in KINDS:
        raise KeyError(f"unknown target transform {kind!r}; known: {KINDS}")
    return {"log1p": np.expm1(p), "sqrt": np.square(p)}[kind]
