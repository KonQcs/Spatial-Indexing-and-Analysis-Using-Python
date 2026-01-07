
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "milano_places.sqlite"
QUERY_SETS_PATH = ROOT / "reports" / "query_sets_allthemes.json"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DB_BUILD_CSV = REPORTS / "db_build_times_allthemes.csv"
DB_QUERY_CSV = REPORTS / "db_query_times_allthemes.csv"
DB_SUMMARY_CSV = REPORTS / "db_query_summary_allthemes.csv"

EARTH_R = 6371000.0


@dataclass(frozen=True)
class Projection:
    lat0_rad: float
    lon0_rad: float


def project_m(lat: float, lon: float, prj: Projection) -> Tuple[float, float]:
    latr = math.radians(lat)
    lonr = math.radians(lon)
    x = EARTH_R * (lonr - prj.lon0_rad) * math.cos(prj.lat0_rad)
    y = EARTH_R * (latr - prj.lat0_rad)
    return x, y


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
    CREATE TABLE IF NOT EXISTS places_xy (
      id INTEGER PRIMARY KEY,
      fsq_place_id TEXT UNIQUE,
      name TEXT,
      top_category TEXT,
      themes_json TEXT,
      lat REAL,
      lon REAL,
      x REAL,
      y REAL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS places_rtree USING rtree(
      id,
      minX, maxX,
      minY, maxY
    );

    CREATE INDEX IF NOT EXISTS idx_places_xy_themes ON places_xy(themes_json);
    CREATE INDEX IF NOT EXISTS idx_places_xy_topcat ON places_xy(top_category);
    """
    )
    conn.commit()


def rebuild_xy_and_rtree_for_theme(conn: sqlite3.Connection, theme: str) -> Tuple[Projection, float, int]:
    """
    Rebuilds places_xy and places_rtree for a single theme (restaurants/theaters/libraries).
    Assumes a base table 'places' exists with (fsq_place_id, name, lat, lon, categories_json, themes_json).
    """
    cur = conn.cursor()

    # Load only theme rows to compute projection + insert
    cur.execute(
        """
        SELECT fsq_place_id, name, lat, lon, categories_json, themes_json
        FROM places
        WHERE themes_json LIKE ?
          AND lat IS NOT NULL AND lon IS NOT NULL
        """,
        (f"%{theme}%",),
    )
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No rows found in DB for theme={theme}. Check places.themes_json content.")

    lats = [float(r[2]) for r in rows]
    lons = [float(r[3]) for r in rows]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    prj = Projection(lat0_rad=math.radians(lat0), lon0_rad=math.radians(lon0))

    # clear previous
    conn.execute("DELETE FROM places_xy")
    conn.execute("DELETE FROM places_rtree")
    conn.commit()

    ins_xy = """
    INSERT INTO places_xy (id, fsq_place_id, name, top_category, themes_json, lat, lon, x, y)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    ins_rt = "INSERT INTO places_rtree (id, minX, maxX, minY, maxY) VALUES (?, ?, ?, ?, ?)"

    t0 = time.perf_counter()
    i = 1
    for pid, name, lat, lon, cats_json, themes_json in rows:
        # parse top category
        try:
            cats = json.loads(cats_json) if cats_json else []
            top_cat = cats[0].get("name") if cats and isinstance(cats[0], dict) else "UNKNOWN"
        except Exception:
            top_cat = "UNKNOWN"

        x, y = project_m(float(lat), float(lon), prj)

        conn.execute(ins_xy, (i, pid, name or "", str(top_cat), themes_json or "[]", float(lat), float(lon), float(x), float(y)))
        conn.execute(ins_rt, (i, float(x), float(x), float(y), float(y)))
        i += 1

    conn.commit()
    t1 = time.perf_counter()
    return prj, (t1 - t0), len(rows)


def nn_expand(cur: sqlite3.Cursor, qx: float, qy: float) -> List[int]:
    """
    SQLite RTree does not provide a native NN operator, so we expand a window until candidates exist,
    then pick closest by squared distance.
    """
    r = 50.0
    for _ in range(18):
        cur.execute(
            """
            SELECT p.id, p.x, p.y
            FROM places_rtree r
            JOIN places_xy p ON p.id=r.id
            WHERE r.minX <= ? AND r.maxX >= ?
              AND r.minY <= ? AND r.maxY >= ?
            """,
            (qx + r, qx - r, qy + r, qy - r),
        )
        cand = cur.fetchall()
        if cand:
            best_id = cand[0][0]
            best_d2 = float("inf")
            for pid, x, y in cand:
                d2 = (x - qx) * (x - qx) + (y - qy) * (y - qy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = pid
            return [int(best_id)]
        r *= 2.0
    return [1]


def bbox_query(cur: sqlite3.Cursor, minx: float, miny: float, maxx: float, maxy: float) -> List[int]:
    cur.execute(
        """
        SELECT p.id
        FROM places_rtree r
        JOIN places_xy p ON p.id=r.id
        WHERE r.minX <= ? AND r.maxX >= ?
          AND r.minY <= ? AND r.maxY >= ?
        """,
        (maxx, minx, maxy, miny),
    )
    return [int(x[0]) for x in cur.fetchall()]


def radius_query(cur: sqlite3.Cursor, qx: float, qy: float, rr: float) -> List[int]:
    cur.execute(
        """
        SELECT p.id, p.x, p.y
        FROM places_rtree r
        JOIN places_xy p ON p.id=r.id
        WHERE r.minX <= ? AND r.maxX >= ?
          AND r.minY <= ? AND r.maxY >= ?
        """,
        (qx + rr, qx - rr, qy + rr, qy - rr),
    )
    r2 = rr * rr
    out = []
    for pid, x, y in cur.fetchall():
        if (x - qx) * (x - qx) + (y - qy) * (y - qy) <= r2:
            out.append(int(pid))
    return out


def composite_query(cur: sqlite3.Cursor, qx: float, qy: float, rr: float, cat_contains: str) -> List[int]:
    needle = cat_contains.lower()
    cur.execute(
        """
        SELECT p.id, p.x, p.y, p.top_category
        FROM places_rtree r
        JOIN places_xy p ON p.id=r.id
        WHERE r.minX <= ? AND r.maxX >= ?
          AND r.minY <= ? AND r.maxY >= ?
        """,
        (qx + rr, qx - rr, qy + rr, qy - rr),
    )
    r2 = rr * rr
    out = []
    for pid, x, y, top_cat in cur.fetchall():
        if (x - qx) * (x - qx) + (y - qy) * (y - qy) <= r2:
            if needle in (top_cat or "").lower():
                out.append(int(pid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH}")
    if not QUERY_SETS_PATH.exists():
        raise FileNotFoundError(f"Missing query sets: {QUERY_SETS_PATH} (run 03_indexes_benchmark_allthemes.py first)")

    query_sets_all = json.loads(QUERY_SETS_PATH.read_text(encoding="utf-8"))

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    build_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []

    for theme, blob in query_sets_all.items():
        qset = blob["queries"]
        prj, build_s, n_points = rebuild_xy_and_rtree_for_theme(conn, theme)

        build_rows.append({"theme": theme, "method": "sqlite_rtree", "build_s": build_s, "n_points": n_points})

        cur = conn.cursor()

        def bench(qtype: str, qid: int, fn):
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
                    "method": "sqlite_rtree",
                    "qtype": qtype,
                    "qid": qid,
                    "elapsed_ms_mean": float(statistics.mean(times)),
                    "elapsed_ms_median": float(statistics.median(times)),
                    "result_count": int(res_len),
                }
            )

        for i in range(len(qset["nn"])):
            qnn = qset["nn"][i]
            qb = qset["bbox"][i]
            qr = qset["radius"][i]
            qc = qset["composite"][i]

            qx, qy = float(qnn["qx"]), float(qnn["qy"])
            minx, miny, maxx, maxy = map(float, (qb["minx"], qb["miny"], qb["maxx"], qb["maxy"]))
            rr = float(qr["r"])
            cat_f = str(qc["category_contains"])

            bench("nn", i, lambda: nn_expand(cur, qx, qy))
            bench("bbox", i, lambda: bbox_query(cur, minx, miny, maxx, maxy))
            bench("radius", i, lambda: radius_query(cur, qx, qy, rr))
            bench("composite", i, lambda: composite_query(cur, qx, qy, rr, cat_f))

        print(f"[OK] theme={theme} points={n_points} | db rtree built + benchmarked")

    pd.DataFrame(build_rows).to_csv(DB_BUILD_CSV, index=False, encoding="utf-8")

    dfq = pd.DataFrame(query_rows)
    dfq.to_csv(DB_QUERY_CSV, index=False, encoding="utf-8")

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
    pd.DataFrame(summ_rows).sort_values(["theme", "qtype", "mean_ms"]).to_csv(DB_SUMMARY_CSV, index=False, encoding="utf-8")

    print("Wrote:", DB_BUILD_CSV)
    print("Wrote:", DB_QUERY_CSV)
    print("Wrote:", DB_SUMMARY_CSV)

    conn.close()


if __name__ == "__main__":
    main()
