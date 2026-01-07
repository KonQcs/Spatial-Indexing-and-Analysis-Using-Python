
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from rtree import index as rtree_index
import pyqtree

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_JSONL = ROOT / "data" / "clean" / "places_clean.jsonl"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

QUERY_SETS_PATH = REPORTS / "query_sets_allthemes.json"
BUILD_TIMES_CSV = REPORTS / "index_build_times_allthemes.csv"
QUERY_TIMES_CSV = REPORTS / "index_query_times_allthemes.csv"
QUERY_SUMMARY_CSV = REPORTS / "index_query_summary_allthemes.csv"

# ---------------------------
# Parameters
# ---------------------------
EARTH_R = 6371000.0
DEFAULT_THEMES = ["restaurants", "theaters", "libraries"]


@dataclass(frozen=True)
class Projection:
    lat0_rad: float
    lon0_rad: float


def project_m(lat: float, lon: float, prj: Projection) -> Tuple[float, float]:
    """Local equirectangular projection around theme mean -> meters."""
    latr = math.radians(lat)
    lonr = math.radians(lon)
    x = EARTH_R * (lonr - prj.lon0_rad) * math.cos(prj.lat0_rad)
    y = EARTH_R * (latr - prj.lat0_rad)
    return x, y


def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def load_theme_rows(theme: str) -> Tuple[List[Dict[str, Any]], Projection]:
    if not DATA_JSONL.exists():
        raise FileNotFoundError(f"Missing {DATA_JSONL}")

    rows: List[Dict[str, Any]] = []
    with DATA_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lat = r.get("lat")
            lon = r.get("lon")
            if lat is None or lon is None:
                continue
            themes = r.get("themes") or []
            if theme not in themes:
                continue

            cats = r.get("categories") or []
            top_cat = cats[0].get("name") if cats and isinstance(cats[0], dict) else "UNKNOWN"
            rows.append(
                {
                    "fsq_place_id": r.get("fsq_place_id"),
                    "name": r.get("name") or "",
                    "lat": float(lat),
                    "lon": float(lon),
                    "top_category": str(top_cat),
                }
            )

    if not rows:
        raise RuntimeError(f"No data for theme='{theme}' in {DATA_JSONL}")

    lat0 = sum(r["lat"] for r in rows) / len(rows)
    lon0 = sum(r["lon"] for r in rows) / len(rows)
    prj = Projection(lat0_rad=math.radians(lat0), lon0_rad=math.radians(lon0))

    for r in rows:
        x, y = project_m(r["lat"], r["lon"], prj)
        r["x"] = float(x)
        r["y"] = float(y)

    return rows, prj


def safe_rtree_props(capacity: int) -> rtree_index.Property:
    """
    Prevents: NearMinimumOverlapFactor must be < leaf_capacity/index_capacity.
    """
    p = rtree_index.Property()
    p.index_capacity = int(capacity)
    p.leaf_capacity = int(capacity)

    # Must be < both capacities; keep a reasonable value.
    p.near_minimum_overlap_factor = int(min(32, capacity - 1))
    return p


def build_kdtree(points_xy: np.ndarray) -> Tuple[KDTree, float]:
    t0 = time.perf_counter()
    tree = KDTree(points_xy)
    t1 = time.perf_counter()
    return tree, (t1 - t0)


def build_quadtree(points_xy: np.ndarray) -> Tuple[pyqtree.Index, float]:
    minx, miny = float(points_xy[:, 0].min()), float(points_xy[:, 1].min())
    maxx, maxy = float(points_xy[:, 0].max()), float(points_xy[:, 1].max())

    t0 = time.perf_counter()
    qt = pyqtree.Index(bbox=(minx, miny, maxx, maxy))
    for i, (x, y) in enumerate(points_xy):
        qt.insert(i, (float(x), float(y), float(x), float(y)))
    t1 = time.perf_counter()
    return qt, (t1 - t0)


def build_rtree(points_xy: np.ndarray, props: rtree_index.Property) -> Tuple[rtree_index.Index, float]:
    t0 = time.perf_counter()
    idx = rtree_index.Index(properties=props)
    for i, (x, y) in enumerate(points_xy):
        idx.insert(i, (float(x), float(y), float(x), float(y)))
    t1 = time.perf_counter()
    return idx, (t1 - t0)


def make_query_set(rows: List[Dict[str, Any]], seed: int, n_queries: int, radius_m: float) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stores queries in x/y meters (since all indexes use x/y),
    but also stores the generating center lat/lon for traceability.
    """
    rnd = random.Random(seed)

    centers = rnd.sample(rows, k=min(n_queries, len(rows)))

    # for composite: choose from frequent top categories
    cat_counts: Dict[str, int] = {}
    for r in rows:
        cat_counts[r["top_category"]] = cat_counts.get(r["top_category"], 0) + 1
    top_cats = [c for c, _ in sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]]
    if not top_cats:
        top_cats = ["Restaurant"]

    out = {"nn": [], "bbox": [], "radius": [], "composite": []}

    for c in centers:
        cx, cy = float(c["x"]), float(c["y"])

        # NN query: jitter so distance isn't always 0
        dx = rnd.uniform(-80, 80)
        dy = rnd.uniform(-80, 80)
        qx, qy = cx + dx, cy + dy

        out["nn"].append({"qx": qx, "qy": qy, "center_lat": c["lat"], "center_lon": c["lon"], "dx_m": dx, "dy_m": dy})

        half_w = rnd.uniform(300, 1200)
        half_h = rnd.uniform(300, 1200)
        out["bbox"].append(
            {"minx": qx - half_w, "miny": qy - half_h, "maxx": qx + half_w, "maxy": qy + half_h, "half_w_m": half_w, "half_h_m": half_h}
        )

        out["radius"].append({"qx": qx, "qy": qy, "r": float(radius_m)})

        filt = c["top_category"] or rnd.choice(top_cats)
        out["composite"].append({"qx": qx, "qy": qy, "r": float(radius_m), "category_contains": filt})

    return out


# ---------------------------
# Query implementations
# ---------------------------
def kd_nn(tree: KDTree, qx: float, qy: float) -> int:
    _, i = tree.query([qx, qy], k=1)
    return int(i)


def kd_bbox(tree: KDTree, points_xy: np.ndarray, minx: float, miny: float, maxx: float, maxy: float) -> List[int]:
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    rad = 0.5 * math.hypot(maxx - minx, maxy - miny)
    cand = tree.query_ball_point([cx, cy], r=rad)
    out = []
    for i in cand:
        x, y = points_xy[int(i)]
        if minx <= x <= maxx and miny <= y <= maxy:
            out.append(int(i))
    return out


def kd_radius(tree: KDTree, qx: float, qy: float, r: float) -> List[int]:
    cand = tree.query_ball_point([qx, qy], r=r)
    return list(map(int, cand))


def rtree_nn(idx: rtree_index.Index, qx: float, qy: float) -> int:
    it = idx.nearest((qx, qy, qx, qy), 1)
    return int(next(it))


def rtree_bbox(idx: rtree_index.Index, minx: float, miny: float, maxx: float, maxy: float) -> List[int]:
    return list(map(int, idx.intersection((minx, miny, maxx, maxy))))


def rtree_radius(idx: rtree_index.Index, points_xy: np.ndarray, qx: float, qy: float, r: float) -> List[int]:
    cand = idx.intersection((qx - r, qy - r, qx + r, qy + r))
    r2 = r * r
    out = []
    for i in cand:
        x, y = points_xy[int(i)]
        if dist2(float(x), float(y), qx, qy) <= r2:
            out.append(int(i))
    return out


def qt_bbox(qt: pyqtree.Index, minx: float, miny: float, maxx: float, maxy: float) -> List[int]:
    return list(map(int, qt.intersect((minx, miny, maxx, maxy))))


def qt_radius(qt: pyqtree.Index, points_xy: np.ndarray, qx: float, qy: float, r: float) -> List[int]:
    cand = qt.intersect((qx - r, qy - r, qx + r, qy + r))
    r2 = r * r
    out = []
    for i in cand:
        x, y = points_xy[int(i)]
        if dist2(float(x), float(y), qx, qy) <= r2:
            out.append(int(i))
    return out


def qt_nn_expand(qt: pyqtree.Index, points_xy: np.ndarray, qx: float, qy: float) -> int:
    best_i = 0
    best_d2 = float("inf")
    r = 50.0
    for _ in range(18):
        cand = qt.intersect((qx - r, qy - r, qx + r, qy + r))
        if cand:
            for i in cand:
                x, y = points_xy[int(i)]
                d2 = dist2(float(x), float(y), qx, qy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = int(i)
            return best_i
        r *= 2.0
    return best_i


def filter_category(rows: List[Dict[str, Any]], idxs: List[int], contains: str) -> List[int]:
    needle = contains.lower()
    return [i for i in idxs if needle in (rows[i]["top_category"] or "").lower()]


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", default=",".join(DEFAULT_THEMES), help="Comma-separated: restaurants,theaters,libraries")
    ap.add_argument("--n_queries", type=int, default=10)
    ap.add_argument("--radius_m", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1, help="Repeats per query (median/mean across repeats)")
    args = ap.parse_args()

    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    if not themes:
        raise ValueError("No themes provided.")

    all_query_sets: Dict[str, Any] = {}
    build_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []

    for theme in themes:
        rows, prj = load_theme_rows(theme)
        points_xy = np.array([[r["x"], r["y"]] for r in rows], dtype=float)

        # query set per theme
        qset = make_query_set(rows, seed=args.seed, n_queries=args.n_queries, radius_m=args.radius_m)
        all_query_sets[theme] = {
            "theme": theme,
            "n_points": len(rows),
            "seed": args.seed,
            "n_queries": args.n_queries,
            "radius_m": args.radius_m,
            "queries": qset,
        }

        # Build indexes
        kd, kd_t = build_kdtree(points_xy)
        build_rows.append({"theme": theme, "method": "kdtree", "build_s": kd_t, "n_points": len(rows)})

        qt, qt_t = build_quadtree(points_xy)
        build_rows.append({"theme": theme, "method": "quadtree", "build_s": qt_t, "n_points": len(rows)})

        rtrees: Dict[str, rtree_index.Index] = {}

        # RTree variants: default + cap40 + cap100
        p_default = rtree_index.Property()
        idx, t = build_rtree(points_xy, p_default)
        rtrees["rtree_default"] = idx
        build_rows.append({"theme": theme, "method": "rtree_default", "build_s": t, "n_points": len(rows)})

        p40 = safe_rtree_props(40)
        idx, t = build_rtree(points_xy, p40)
        rtrees["rtree_cap40"] = idx
        build_rows.append({"theme": theme, "method": "rtree_cap40", "build_s": t, "n_points": len(rows)})

        p100 = safe_rtree_props(100)
        idx, t = build_rtree(points_xy, p100)
        rtrees["rtree_cap100"] = idx
        build_rows.append({"theme": theme, "method": "rtree_cap100", "build_s": t, "n_points": len(rows)})

        def bench(method: str, qtype: str, qid: int, fn):
            times = []
            res_len = None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                res = fn()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
                res_len = len(res) if isinstance(res, list) else 1
            query_rows.append(
                {
                    "theme": theme,
                    "method": method,
                    "qtype": qtype,
                    "qid": qid,
                    "elapsed_ms_mean": float(statistics.mean(times)),
                    "elapsed_ms_median": float(statistics.median(times)),
                    "result_count": int(res_len),
                }
            )

        # Run queries
        for i in range(len(qset["nn"])):
            qnn = qset["nn"][i]
            qb = qset["bbox"][i]
            qr = qset["radius"][i]
            qc = qset["composite"][i]

            qx, qy = float(qnn["qx"]), float(qnn["qy"])
            minx, miny, maxx, maxy = map(float, (qb["minx"], qb["miny"], qb["maxx"], qb["maxy"]))
            rr = float(qr["r"])
            cat_f = str(qc["category_contains"])

            # KDTree
            bench("kdtree", "nn", i, lambda: kd_nn(kd, qx, qy))
            bench("kdtree", "bbox", i, lambda: kd_bbox(kd, points_xy, minx, miny, maxx, maxy))
            bench("kdtree", "radius", i, lambda: kd_radius(kd, qx, qy, rr))
            bench("kdtree", "composite", i, lambda: filter_category(rows, kd_radius(kd, qx, qy, rr), cat_f))

            # QuadTree
            bench("quadtree", "nn", i, lambda: qt_nn_expand(qt, points_xy, qx, qy))
            bench("quadtree", "bbox", i, lambda: qt_bbox(qt, minx, miny, maxx, maxy))
            bench("quadtree", "radius", i, lambda: qt_radius(qt, points_xy, qx, qy, rr))
            bench("quadtree", "composite", i, lambda: filter_category(rows, qt_radius(qt, points_xy, qx, qy, rr), cat_f))

            # RTree variants
            for rname, ridx in rtrees.items():
                bench(rname, "nn", i, lambda ridx=ridx: rtree_nn(ridx, qx, qy))
                bench(rname, "bbox", i, lambda ridx=ridx: rtree_bbox(ridx, minx, miny, maxx, maxy))
                bench(rname, "radius", i, lambda ridx=ridx: rtree_radius(ridx, points_xy, qx, qy, rr))
                bench(rname, "composite", i, lambda ridx=ridx: filter_category(rows, rtree_radius(ridx, points_xy, qx, qy, rr), cat_f))

        print(f"[OK] theme={theme} points={len(rows)} | built + benchmarked")

    # Write query sets (one JSON for all themes)
    QUERY_SETS_PATH.write_text(json.dumps(all_query_sets, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write build times
    pd.DataFrame(build_rows).to_csv(BUILD_TIMES_CSV, index=False, encoding="utf-8")

    # Write query times
    dfq = pd.DataFrame(query_rows)
    dfq.to_csv(QUERY_TIMES_CSV, index=False, encoding="utf-8")

    # Summary grouped by theme/method/qtype
    summ_rows = []
    for (theme, method, qtype), g in dfq.groupby(["theme", "method", "qtype"]):
        times = list(g["elapsed_ms_mean"].astype(float))
        summ_rows.append(
            {
                "theme": theme,
                "method": method,
                "qtype": qtype,
                "n": int(len(times)),
                "mean_ms": float(statistics.mean(times)),
                "median_ms": float(statistics.median(times)),
                "min_ms": float(min(times)),
                "max_ms": float(max(times)),
            }
        )
    pd.DataFrame(summ_rows).sort_values(["theme", "qtype", "mean_ms"]).to_csv(QUERY_SUMMARY_CSV, index=False, encoding="utf-8")

    print("Wrote:", QUERY_SETS_PATH)
    print("Wrote:", BUILD_TIMES_CSV)
    print("Wrote:", QUERY_TIMES_CSV)
    print("Wrote:", QUERY_SUMMARY_CSV)


if __name__ == "__main__":
    main()
