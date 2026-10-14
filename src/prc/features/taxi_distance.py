"""Stand → runway-threshold taxi distance from OpenStreetMap (idea backlog B1, TODO 4a).

The taxi-time literature's strongest single feature is the distance the aircraft actually
has to travel. The model currently learns that distance implicitly, as a target-encoded
``airport × stand × runway`` lookup — which is excellent on the pairs it has seen thousands
of times and useless on the rare ones (a long-tail runway configuration, a stand that opened
in June). A measured distance generalises across pairs: it says that stand A14 to runway 18
is 3.1 km whether or not that pair appears in training.

Data: OpenStreetMap ``aeroway`` geometry for the ten airports, one Overpass query per airport
(``uv run python scripts/fetch_external.py osm`` → ``data/external/osm_<date>/<ICAO>.json``).
**Licence: ODbL 1.0, © OpenStreetMap contributors** — attribution is required on any
published use; see that directory's ``README.md`` and ``docs/data-sources.md``.

How it is built, per airport:

1. **Graph.** Every ``aeroway=taxiway``, ``aeroway=runway`` and ``aeroway=parking_position``
   way becomes a chain of edges between its OSM nodes, weighted by great-circle length. The
   ways share node ids where they meet, which is what makes the graph connected.
2. **Stands.** Each ``parking_position`` with a ``ref`` gives one stand point. For a way, the
   stand point is the endpoint *farther* from the taxiway network (the aircraft's position)
   and the attachment node is the nearer endpoint; a node-type parking position is snapped to
   the nearest graph node within ``SNAP_M``. ``STAND_mvt`` is matched to ``ref`` through a
   normalisation that strips spaces, punctuation and leading zeros (``A08`` ≡ ``A8``,
   ``V 117`` ≡ ``V117``).
3. **Thresholds.** A runway way's ``ref`` is ``07L/25R``; the threshold for a designator is
   the end of the way *from which* the bearing to the other end matches the designator's
   heading. ``RUNWAY_mvt`` is matched after the same zero-padding
   ``runway_geometry`` uses. Where several ways carry one designator, the longest wins.
4. **Distances.** One Dijkstra per threshold (``scipy.sparse.csgraph`` — already installed
   with scikit-learn, so no new dependency) gives the shortest taxiway-graph distance to
   every stand, plus the predecessor tree, from which the path's runway crossings are counted:
   a crossing is a *different* runway way whose nodes the path passes through.
5. **Apron cluster.** Each stand is assigned the nearest ``aeroway=apron`` polygon centroid,
   as a categorical ``<ICAO>|<n>`` — a terminal / pier identifier that is coarser than the
   stand and finer than the airport.

Everything is static geometry: no timestamps, no labels, nothing that can leak. The block
therefore uses no NM field and does not belong in ``mask_for_no_nm``.

Match rates per airport (and the share of *departure rows* covered, which is what matters —
a handful of unmatched remote stands with two movements a year is not a problem) are in
internal note 2026-09-19_new-blocks-coverage. Where a stand does not match, the columns are
null rather than guessed: no coordinate is ever invented.

Returns all-null columns when the archive is absent, so tests and fresh clones still run.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

#: metres; a node-type parking position further than this from the graph is left unattached
SNAP_M = 150.0
EARTH_R = 6_371_000.0

OUT = [
    "td_straight_m",
    "td_path_m",
    "td_detour_ratio",
    "td_runway_crossings",
    "td_stand_snap_m",
    "td_stand_matched",
    "td_runway_matched",
    "td_apron",
]


def _osm_files() -> list[Path]:
    return archive_files("osm_*", "*.json")


def _td_external() -> str:
    return external_digest(_osm_files(), __name__)


# --------------------------------------------------------------------------- normalisation


def norm_stand(name: str | None) -> str | None:
    """``V 117`` / ``a08`` / ``A-08`` -> ``V117`` / ``A8``. Leading zeros inside the numeric
    part are dropped so that the movement table's ``A08`` meets OSM's ``A8``."""
    if name is None:
        return None
    s = "".join(ch for ch in str(name).upper() if ch.isalnum())
    if not s:
        return None
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            out.append(str(int(s[i:j])))
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def norm_runway(name: str | None) -> str | None:
    """``7l`` / ``07L`` -> ``07L``; non-numeric designators pass through upper-cased."""
    if name is None:
        return None
    s = str(name).strip().upper()
    if not s:
        return None
    digits = ""
    k = 0
    while k < len(s) and s[k].isdigit() and k < 2:
        digits += s[k]
        k += 1
    if not digits:
        return s
    side = s[k:] if s[k:] in ("L", "C", "R") else ""
    return f"{int(digits):02d}{side}"


def base_stand(norm: str | None) -> str | None:
    """``115A`` / ``161Y`` -> ``115`` / ``161``; ``A14`` and ``G11L`` unchanged.

    OSM splits one physical stand into the sub-positions it can be used as (``115A`` /
    ``115B``: one wide-body or two narrow-bodies), while the movement table names the stand
    (``115``). Stripping a trailing letter group from a *numeric* ref is the documented
    fallback; it is never applied to refs that start with a letter, where the trailing letter
    is part of the pier name (``G11L`` at LTFM is not ``G11``). At EDDM this fallback is the
    difference between 57 % and 99 % of departure rows matched.
    """
    if not norm or not norm[0].isdigit():
        return norm
    core = norm.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return core or norm


def _key_table(df: pl.DataFrame) -> pl.DataFrame:
    """The distinct (airport, stand, runway) triples with their normalised forms.

    Normalisation runs over the ~10,000 distinct triples rather than the 4.9 M rows, so the
    exact Python implementations above are used instead of a regex approximation of them.
    """
    return (
        df.select(
            S.AIRPORT,
            pl.col(S.STAND).cast(pl.Utf8).fill_null("").alias("_stand_raw"),
            pl.col(S.RUNWAY).cast(pl.Utf8).fill_null("").alias("_runway_raw"),
        )
        .unique()
        .with_columns(
            pl.col("_stand_raw").map_elements(norm_stand, return_dtype=pl.Utf8).alias("_stand"),
            pl.col("_runway_raw").map_elements(norm_runway, return_dtype=pl.Utf8).alias("_runway"),
        )
    )


# --------------------------------------------------------------------------- geometry


def _haversine(lat1, lon1, lat2, lon2):  # numpy arrays or scalars
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _designator_heading(desig: str) -> float | None:
    head = desig[:2]
    if not head.isdigit():
        return None
    n = int(head)
    return 360.0 if n == 0 else n * 10.0


# --------------------------------------------------------------------------- one airport


class _Airport:
    """The graph, stands and thresholds of one airport, built from its Overpass JSON."""

    def __init__(self, icao: str, elements: list[dict]) -> None:
        self.icao = icao
        self.coord: dict[int, tuple[float, float]] = {}
        self.edges: list[tuple[int, int, float]] = []
        self.runway_nodes: dict[str, set[int]] = {}  # way id (as str) -> node ids
        self.thresholds: dict[str, tuple[float, float, int]] = {}
        self.stands: dict[str, tuple[float, float, int | None, float]] = {}
        self.aprons: list[tuple[float, float]] = []
        self._build(elements)

    # -- construction -----------------------------------------------------
    def _way_points(self, e: dict) -> list[tuple[int, float, float]]:
        nodes, geom = e.get("nodes"), e.get("geometry")
        if not nodes or not geom or len(nodes) != len(geom):
            return []
        pts = []
        for nid, g in zip(nodes, geom, strict=True):
            if g is None or "lat" not in g:
                return []
            pts.append((int(nid), float(g["lat"]), float(g["lon"])))
        return pts

    def _add_chain(self, pts: list[tuple[int, float, float]]) -> None:
        for (a, la, lo), (b, lb, lob) in itertools.pairwise(pts):
            self.coord[a] = (la, lo)
            self.coord[b] = (lb, lob)
            d = float(_haversine(la, lo, lb, lob))
            if d > 0:
                self.edges.append((a, b, d))

    def _build(self, elements: list[dict]) -> None:
        """Movement-area graph first, then the stands hung off it.

        The two phases are separate on purpose: the snapping index must contain only the
        *movement area* (taxiways and runways). Snapping to "the nearest node of anything"
        would happily attach a stand to a neighbouring stand's own parking way, which is
        usually an isolated two-node component, and the stand would then be unreachable.
        """
        parking: list[dict] = []
        for e in elements:
            tags = e.get("tags") or {}
            aeroway = tags.get("aeroway")
            if aeroway in ("taxiway", "runway") and e.get("type") == "way":
                pts = self._way_points(e)
                self._add_chain(pts)
                if aeroway == "runway" and pts:
                    self.runway_nodes[str(e["id"])] = {p[0] for p in pts}
            elif aeroway == "apron" and e.get("type") == "way":
                pts = self._way_points(e)
                if pts:
                    self.aprons.append(
                        (
                            sum(p[1] for p in pts) / len(pts),
                            sum(p[2] for p in pts) / len(pts),
                        )
                    )
            elif aeroway in ("parking_position", "gate"):
                parking.append(e)

        self._index_movement_area()
        self._ma_set = set(self.ma_ids)
        self._add_runway_thresholds(elements)
        for e in parking:  # parking positions are drawn onto the graph as ordinary ways
            if e.get("type") == "way":
                self._add_chain(self._way_points(e))
        self._add_stands(parking)
        self._index_nodes()

    def _index_movement_area(self) -> None:
        """Snapping index over the taxiway / runway nodes only (see :meth:`_build`)."""
        self.ma_ids = sorted(self.coord)
        self.ma_ll = (
            np.array([self.coord[n] for n in self.ma_ids], dtype=float)
            if self.ma_ids
            else np.zeros((0, 2))
        )

    def _index_nodes(self) -> None:
        self.node_ids = sorted(self.coord)
        self.idx = {n: i for i, n in enumerate(self.node_ids)}
        self.node_ll = np.array([self.coord[n] for n in self.node_ids], dtype=float)

    def _add_runway_thresholds(self, elements: list[dict]) -> None:
        best: dict[str, tuple[float, tuple[float, float, int]]] = {}
        for e in elements:
            tags = e.get("tags") or {}
            if tags.get("aeroway") != "runway" or e.get("type") != "way":
                continue
            ref = tags.get("ref")
            pts = self._way_points(e)
            if not ref or len(pts) < 2:
                continue
            a, b = pts[0], pts[-1]
            length = float(_haversine(a[1], a[2], b[1], b[2]))
            brg = _bearing(a[1], a[2], b[1], b[2])
            for raw in str(ref).replace(" ", "").split("/"):
                d = norm_runway(raw)
                hdg = _designator_heading(d) if d else None
                if d is None or hdg is None:
                    continue
                # the threshold of designator `d` is the end you take off *from*
                diff = abs(((brg - hdg + 180.0) % 360.0) - 180.0)
                end = a if diff < 90.0 else b
                if d not in best or length > best[d][0]:
                    best[d] = (length, (end[1], end[2], end[0]))
        self.thresholds = {d: v for d, (_, v) in best.items()}

    def _nearest_node(self, lat: float, lon: float) -> tuple[int | None, float]:
        """Nearest *movement-area* node and its distance in metres."""
        if not len(self.ma_ll):
            return None, float("inf")
        d = _haversine(lat, lon, self.ma_ll[:, 0], self.ma_ll[:, 1])
        k = int(np.argmin(d))
        return self.ma_ids[k], float(d[k])

    def _add_stands(self, parking: list[dict]) -> None:
        for e in parking:
            tags = e.get("tags") or {}
            ref = norm_stand(tags.get("ref") or tags.get("local_ref"))
            if not ref:
                continue
            if e.get("type") == "node":
                lat, lon = float(e["lat"]), float(e["lon"])
                attach_id, attach_lat, attach_lon = None, lat, lon
            else:
                pts = self._way_points(e)
                if len(pts) < 2:
                    continue
                a, b = pts[0], pts[-1]
                # the aircraft stands at the end further from the movement area; the other
                # end is where the parking position meets it
                _, da = self._nearest_node(a[1], a[2])
                _, db = self._nearest_node(b[1], b[2])
                stand_pt, attach_pt = (a, b) if da >= db else (b, a)
                lat, lon = stand_pt[1], stand_pt[2]
                attach_id, attach_lat, attach_lon = attach_pt
            snap = 0.0
            node = attach_id if attach_id is not None and attach_id in self.coord else None
            if node is None or not self._reaches_movement_area(node):
                near, dist = self._nearest_node(attach_lat, attach_lon)
                if near is None or dist > SNAP_M:
                    node, snap = None, float("nan")
                else:
                    # an explicit connector edge, so the stand is genuinely in the graph
                    if attach_id is not None and attach_id in self.coord:
                        self.edges.append((attach_id, near, dist))
                        node = attach_id
                    else:
                        node = near
                    snap = dist
            prev = self.stands.get(ref)
            if prev is None or (prev[2] is None and node is not None):
                self.stands[ref] = (lat, lon, node, snap)

    def _reaches_movement_area(self, node: int) -> bool:
        """True when the node is one of the taxiway / runway nodes themselves."""
        return node in self._ma_set

    # -- distances --------------------------------------------------------
    def table(self) -> list[dict]:
        """One row per (stand, runway designator) with straight-line and path distance."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        rows: list[dict] = []
        if not self.stands or not self.thresholds:
            return rows
        n = len(self.node_ids)
        if n:
            i = [self.idx[a] for a, _, _ in self.edges]
            j = [self.idx[b] for _, b, _ in self.edges]
            w = [d for _, _, d in self.edges]
            graph = coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
        apron_ll = np.array(self.aprons, dtype=float) if self.aprons else None
        rwy_masks = {
            wid: np.array([self.idx[x] for x in nodes if x in self.idx], dtype=int)
            for wid, nodes in self.runway_nodes.items()
        }
        for desig, (rlat, rlon, rnode) in self.thresholds.items():
            dist = pred = None
            if n and rnode in self.idx:
                dist, pred = dijkstra(
                    graph, directed=False, indices=self.idx[rnode], return_predecessors=True
                )
            # the runway ways that carry this designator are not "crossings"
            own = {wid for wid, nodes in self.runway_nodes.items() if rnode in nodes}
            others = {wid: m for wid, m in rwy_masks.items() if wid not in own}
            for ref, (slat, slon, snode, snap) in self.stands.items():
                straight = float(_haversine(slat, slon, rlat, rlon))
                path_m = crossings = None
                if dist is not None and snode is not None and snode in self.idx:
                    d = float(dist[self.idx[snode]])
                    if np.isfinite(d):
                        path_m = d
                        crossings = self._count_crossings(pred, self.idx[snode], others)
                apron = None
                if apron_ll is not None:
                    k = int(np.argmin(_haversine(slat, slon, apron_ll[:, 0], apron_ll[:, 1])))
                    apron = f"{self.icao}|{k}"
                rows.append(
                    {
                        S.AIRPORT: self.icao,
                        "_stand": ref,
                        "_runway": desig,
                        "td_straight_m": straight,
                        "td_path_m": path_m,
                        "td_runway_crossings": crossings,
                        "td_stand_snap_m": None if math.isnan(snap) else snap,
                        "td_apron": apron,
                    }
                )
        return rows

    @staticmethod
    def _count_crossings(pred: np.ndarray, target: int, others: dict[str, np.ndarray]) -> int:
        seen: set[int] = set()
        cur = target
        while cur >= 0:
            seen.add(int(cur))
            nxt = int(pred[cur])
            if nxt == cur or nxt < 0:
                break
            cur = nxt
        return sum(1 for m in others.values() if len(m) and seen.intersection(m.tolist()))


def _lookup() -> pl.DataFrame | None:
    """The (airport, stand, runway) → distance table for every airport in the archive."""
    files = _osm_files()
    if not files:
        return None
    rows: list[dict] = []
    for f in files:
        if f.name == "manifest.json":
            continue
        try:
            elements = json.loads(f.read_text())["elements"]
        except (KeyError, ValueError):
            continue
        rows += _Airport(f.stem, elements).table()
    if not rows:
        return None
    return pl.DataFrame(rows).with_columns(
        pl.col("td_straight_m").cast(pl.Float32),
        pl.col("td_path_m").cast(pl.Float32),
        pl.col("td_runway_crossings").cast(pl.Int8),
        pl.col("td_stand_snap_m").cast(pl.Float32),
        pl.col("td_apron").cast(pl.Categorical),
    )


@feature("taxi_distance", external=_td_external)
def taxi_distance(df: pl.DataFrame) -> pl.DataFrame:
    """Straight-line and taxiway-graph distance from the departure's stand to its runway
    threshold, the detour ratio, the number of other runways crossed on that path, and the
    stand's apron cluster — from OpenStreetMap (ODbL, © OpenStreetMap contributors)."""
    table = _lookup()
    if table is None:
        return df.select(S.MVT_ID).with_columns(
            [
                pl.lit(None, dtype=pl.Categorical if c == "td_apron" else pl.Float32).alias(c)
                for c in OUT
            ]
        )
    num = ["td_straight_m", "td_path_m", "td_runway_crossings", "td_stand_snap_m"]
    # the sub-position fallback: one row per (airport, base stand, runway)
    base_tbl = (
        table.with_columns(
            pl.col("_stand").map_elements(base_stand, return_dtype=pl.Utf8).alias("_stand_base")
        )
        .group_by([S.AIRPORT, "_stand_base", "_runway"])
        .agg(
            [pl.col(c).mean().alias(c) for c in num]
            + [pl.col("td_apron").first().alias("td_apron")]
        )
    )
    keys = _key_table(df).with_columns(
        pl.col("_stand").map_elements(base_stand, return_dtype=pl.Utf8).alias("_stand_base")
    )
    stands = (
        table.select(S.AIRPORT, "_stand").unique().with_columns(pl.lit(2, pl.Int8).alias("_sm"))
    )
    base_stands = (
        base_tbl.select(S.AIRPORT, "_stand_base")
        .unique()
        .with_columns(pl.lit(1, pl.Int8).alias("_smb"))
    )
    runways = (
        table.select(S.AIRPORT, "_runway").unique().with_columns(pl.lit(1, pl.Int8).alias("_rm"))
    )
    keys = (
        keys.join(table, on=[S.AIRPORT, "_stand", "_runway"], how="left")
        .join(base_tbl, on=[S.AIRPORT, "_stand_base", "_runway"], how="left", suffix="_b")
        .join(stands, on=[S.AIRPORT, "_stand"], how="left")
        .join(base_stands, on=[S.AIRPORT, "_stand_base"], how="left")
        .join(runways, on=[S.AIRPORT, "_runway"], how="left")
        .with_columns(
            [pl.coalesce(pl.col(c), pl.col(f"{c}_b")).alias(c) for c in num]
            + [pl.coalesce(pl.col("td_apron"), pl.col("td_apron_b")).alias("td_apron")]
        )
    )
    out = df.select(
        S.MVT_ID,
        S.AIRPORT,
        pl.col(S.STAND).cast(pl.Utf8).fill_null("").alias("_stand_raw"),
        pl.col(S.RUNWAY).cast(pl.Utf8).fill_null("").alias("_runway_raw"),
    ).join(keys, on=[S.AIRPORT, "_stand_raw", "_runway_raw"], how="left")
    return out.select(
        S.MVT_ID,
        pl.col("td_straight_m").cast(pl.Float32),
        pl.col("td_path_m").cast(pl.Float32),
        (pl.col("td_path_m") / pl.col("td_straight_m").clip(1.0, None))
        .cast(pl.Float32)
        .alias("td_detour_ratio"),
        pl.col("td_runway_crossings").cast(pl.Float32),
        pl.col("td_stand_snap_m").cast(pl.Float32),
        # 2 = the stand ref matched exactly, 1 = only through the sub-position fallback, 0 = no
        pl.coalesce(pl.col("_sm"), pl.col("_smb"), pl.lit(0, pl.Int8)).alias("td_stand_matched"),
        pl.col("_rm").fill_null(0).alias("td_runway_matched"),
        pl.col("td_apron"),
    )
