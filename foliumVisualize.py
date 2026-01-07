
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import folium
from folium.plugins import MarkerCluster
import numpy as np

# backend
from scipy.spatial import KDTree
import pyqtree
from rtree import index as rtree_index

EARTH_R = 6371000.0

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "milano_places.sqlite"
DEFAULT_QUERYSETS = ROOT / "reports" / "query_sets_allthemes.json"
OUT_DIR = ROOT / "reports" / "maps"


# ----------------------------
# Projection
# ----------------------------
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


def inverse_project_m(x: float, y: float, prj: Projection) -> Tuple[float, float]:
    latr = prj.lat0_rad + (y / EARTH_R)
    lonr = prj.lon0_rad + (x / (EARTH_R * math.cos(prj.lat0_rad)))
    return (math.degrees(latr), math.degrees(lonr))


def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


# ----------------------------
# Load points from SQLite
# Requires base table "places" with:
# fsq_place_id, name, lat, lon, categories_json, themes_json
# ----------------------------
def load_theme_points_from_db(db_path: Path, theme: str) -> Tuple[List[Dict[str, Any]], Projection]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

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
    conn.close()

    if not rows:
        raise RuntimeError(f"No rows for theme={theme}. Check places.themes_json or theme string.")

    lats = [float(r[2]) for r in rows]
    lons = [float(r[3]) for r in rows]
    prj = Projection(lat0_rad=math.radians(sum(lats) / len(lats)),
                     lon0_rad=math.radians(sum(lons) / len(lons)))

    out: List[Dict[str, Any]] = []
    for fsq_id, name, lat, lon, cats_json, themes_json in rows:
        try:
            cats = json.loads(cats_json) if cats_json else []
            top_cat = cats[0].get("name") if cats and isinstance(cats[0], dict) else "UNKNOWN"
        except Exception:
            top_cat = "UNKNOWN"

        x, y = project_m(float(lat), float(lon), prj)
        out.append({
            "id": str(fsq_id),
            "name": name or "",
            "lat": float(lat),
            "lon": float(lon),
            "top_category": str(top_cat),
            "x": float(x),
            "y": float(y),
        })

    return out, prj


# ----------------------------
# Index builders
# ----------------------------
def safe_rtree_props(capacity: int = 40) -> rtree_index.Property:
    p = rtree_index.Property()
    p.index_capacity = int(capacity)
    p.leaf_capacity = int(capacity)
    p.near_minimum_overlap_factor = int(min(32, capacity - 1))
    return p


def build_indexes(points_xy: np.ndarray) -> Dict[str, Any]:
    kd = KDTree(points_xy)

    minx, miny = float(points_xy[:, 0].min()), float(points_xy[:, 1].min())
    maxx, maxy = float(points_xy[:, 0].max()), float(points_xy[:, 1].max())
    qt = pyqtree.Index(bbox=(minx, miny, maxx, maxy))
    for i, (x, y) in enumerate(points_xy):
        qt.insert(i, (float(x), float(y), float(x), float(y)))

    rt = rtree_index.Index(properties=safe_rtree_props(40))
    for i, (x, y) in enumerate(points_xy):
        rt.insert(i, (float(x), float(y), float(x), float(y)))

    return {"kdtree": kd, "quadtree": qt, "rtree": rt}


# ----------------------------
# Queries
# ----------------------------
def idx_nn_quadtree(qt, points_xy, qx, qy) -> int:
    # expanding window NN
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


def idx_bbox_rtree(rt, minx, miny, maxx, maxy) -> List[int]:
    return list(map(int, rt.intersection((minx, miny, maxx, maxy))))


def idx_radius_kdtree(kd, qx, qy, r) -> List[int]:
    return list(map(int, kd.query_ball_point([qx, qy], r=r)))


def idx_composite(kd, rows, qx, qy, r, cat_contains: str) -> Tuple[List[int], List[int]]:
    # returns (candidates_in_radius, filtered)
    cand = idx_radius_kdtree(kd, qx, qy, r)
    needle = cat_contains.lower()
    filt = [i for i in cand if needle in (rows[i]["top_category"] or "").lower()]
    return cand, filt


# ----------------------------
# DB backend: build per-theme temp working tables
# ----------------------------
def ensure_work_tables(conn: sqlite3.Connection) -> None:
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
        """
    )
    conn.commit()


def rebuild_xy_and_rtree(conn: sqlite3.Connection, theme: str) -> Projection:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fsq_place_id, name, lat, lon, categories_json, themes_json
        FROM places
        WHERE themes_json LIKE ?
          AND lat IS NOT NULL AND lon IS NOT NULL
        """,
        (f"%{theme}%",),
    )
    base = cur.fetchall()
    if not base:
        raise RuntimeError(f"No DB rows for theme={theme}")

    lats = [float(r[2]) for r in base]
    lons = [float(r[3]) for r in base]
    prj = Projection(lat0_rad=math.radians(sum(lats) / len(lats)),
                     lon0_rad=math.radians(sum(lons) / len(lons)))

    conn.execute("DELETE FROM places_xy")
    conn.execute("DELETE FROM places_rtree")
    conn.commit()

    i = 1
    for fsq_id, name, lat, lon, cats_json, themes_json in base:
        try:
            cats = json.loads(cats_json) if cats_json else []
            top_cat = cats[0].get("name") if cats and isinstance(cats[0], dict) else "UNKNOWN"
        except Exception:
            top_cat = "UNKNOWN"

        x, y = project_m(float(lat), float(lon), prj)
        conn.execute(
            """
            INSERT INTO places_xy (id, fsq_place_id, name, top_category, themes_json, lat, lon, x, y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (i, str(fsq_id), name or "", str(top_cat), themes_json or "[]", float(lat), float(lon), float(x), float(y)),
        )
        conn.execute(
            "INSERT INTO places_rtree (id, minX, maxX, minY, maxY) VALUES (?, ?, ?, ?, ?)",
            (i, float(x), float(x), float(y), float(y)),
        )
        i += 1

    conn.commit()
    return prj


def db_fetch_points_by_ids(conn: sqlite3.Connection, ids: List[int]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    qmarks = ",".join(["?"] * len(ids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, fsq_place_id, name, top_category, lat, lon, x, y
        FROM places_xy
        WHERE id IN ({qmarks})
        """,
        ids,
    )
    out = []
    for rid, fsq_id, name, top_cat, lat, lon, x, y in cur.fetchall():
        out.append({
            "rid": int(rid),
            "id": str(fsq_id),
            "name": name or "",
            "top_category": top_cat or "",
            "lat": float(lat),
            "lon": float(lon),
            "x": float(x),
            "y": float(y),
        })
    return out


def db_bbox(conn, minx, miny, maxx, maxy) -> List[int]:
    cur = conn.cursor()
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


def db_radius_candidates(conn, qx, qy, r) -> List[Tuple[int, float, float, str]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.id, p.x, p.y, p.top_category
        FROM places_rtree r
        JOIN places_xy p ON p.id=r.id
        WHERE r.minX <= ? AND r.maxX >= ?
          AND r.minY <= ? AND r.maxY >= ?
        """,
        (qx + r, qx - r, qy + r, qy - r),
    )
    return [(int(pid), float(x), float(y), str(tc or "")) for pid, x, y, tc in cur.fetchall()]


def db_radius(conn, qx, qy, r) -> List[int]:
    r2 = r * r
    out = []
    for pid, x, y, _tc in db_radius_candidates(conn, qx, qy, r):
        if dist2(x, y, qx, qy) <= r2:
            out.append(pid)
    return out


def db_composite(conn, qx, qy, r, cat_contains: str) -> Tuple[List[int], List[int]]:
    r2 = r * r
    needle = cat_contains.lower()
    cand_ids = []
    filt_ids = []
    for pid, x, y, tc in db_radius_candidates(conn, qx, qy, r):
        if dist2(x, y, qx, qy) <= r2:
            cand_ids.append(pid)
            if needle in tc.lower():
                filt_ids.append(pid)
    return cand_ids, filt_ids


def db_nn_expand(conn, qx, qy) -> int:
    cur = conn.cursor()
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
            best_id = int(cand[0][0])
            best_d2 = float("inf")
            for pid, x, y in cand:
                d2 = dist2(float(x), float(y), qx, qy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = int(pid)
            return best_id
        r *= 2.0
    return 1


# ----------------------------
# Folium helpers
# ----------------------------
def add_title(m: folium.Map, title_html: str) -> None:
    folium.Element(title_html).add_to(m.get_root().html)


def circle_marker(lat, lon, popup: str, color: str, radius: int = 4, fill_opacity: float = 0.85):
    return folium.CircleMarker(
        location=(lat, lon),
        radius=radius,
        popup=popup,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        weight=1,
    )


def build_map_base(center_lat: float, center_lon: float, zoom: int = 14) -> folium.Map:
    return folium.Map(location=(center_lat, center_lon), zoom_start=zoom, control_scale=True)


def write_index_html(out_folder: Path, links: List[Tuple[str, str]]) -> None:
    # links: (label, relative_filename)
    rows = "\n".join([f'<li><a href="{fn}">{label}</a></li>' for label, fn in links])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Maps</title></head>
<body>
<h2>Generated Maps</h2>
<ul>{rows}</ul>
</body></html>"""
    (out_folder / "index.html").write_text(html, encoding="utf-8")


# ----------------------------
# Main visualization
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["indexes", "db"], default="indexes",
                    help="indexes: build KD/Quad/R indexes; db: run queries via SQLite+RTree")
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB))
    ap.add_argument("--query_sets", type=str, default=str(DEFAULT_QUERYSETS))
    ap.add_argument("--max_markers", type=int, default=600,
                    help="cap markers plotted (useful for restaurants)")
    ap.add_argument("--qid", type=int, default=0, help="which query id (0..9) to visualize per qtype")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db_path = Path(args.db)
    qs_path = Path(args.query_sets)

    if not qs_path.exists():
        raise FileNotFoundError(f"Missing query sets: {qs_path}")

    all_qs: Dict[str, Any] = json.loads(qs_path.read_text(encoding="utf-8"))
    backend = args.backend

    out_root = OUT_DIR / backend
    out_root.mkdir(parents=True, exist_ok=True)

    # If DB backend, keep one connection and rebuild working tables per theme
    conn = None
    if backend == "db":
        if not db_path.exists():
            raise FileNotFoundError(f"Missing DB: {db_path}")
        conn = sqlite3.connect(db_path)
        ensure_work_tables(conn)

    # Generate maps per theme
    for theme, blob in all_qs.items():
        theme_out = out_root / theme
        theme_out.mkdir(parents=True, exist_ok=True)

        q = blob["queries"]
        qid = int(args.qid)

        # Load points (always from DB base table for consistency)
        if not db_path.exists():
            raise FileNotFoundError(f"DB not found at {db_path}. This script loads points from SQLite.")
        rows, prj = load_theme_points_from_db(db_path, theme)
        points_xy = np.array([[r["x"], r["y"]] for r in rows], dtype=float)

        # Prepare backend structures
        indexes = None
        if backend == "indexes":
            indexes = build_indexes(points_xy)
        else:
            assert conn is not None
            prj = rebuild_xy_and_rtree(conn, theme)  # rebuild working tables for this theme

        # Utility: fetch point dicts by index-list (indexes backend)
        def rows_by_idxs(idxs: List[int]) -> List[Dict[str, Any]]:
            return [rows[i] for i in idxs]

        # Prepare per-qtype maps
        links: List[Tuple[str, str]] = []

        # ---- NN ----
        nnq = q["nn"][qid]
        qx, qy = float(nnq["qx"]), float(nnq["qy"])
        qlat, qlon = inverse_project_m(qx, qy, prj)

        if backend == "indexes":
            nn_i = idx_nn_quadtree(indexes["quadtree"], points_xy, qx, qy)
            nn_pt = rows[nn_i]
        else:
            nn_id = db_nn_expand(conn, qx, qy)
            nn_pt = db_fetch_points_by_ids(conn, [nn_id])[0]

        m = build_map_base(qlat, qlon)
        add_title(m, f"""
        <div style="position: fixed; top: 10px; left: 10px; z-index:9999;
                    background: white; padding: 10px; border: 1px solid #999;">
            <b>Theme:</b> {theme}<br/>
            <b>Backend:</b> {backend}<br/>
            <b>Query:</b> NN (qid={qid})<br/>
        </div>
        """)
        folium.Marker((qlat, qlon), popup="Query point (NN)", icon=folium.Icon(color="red")).add_to(m)
        folium.Marker((nn_pt["lat"], nn_pt["lon"]),
                      popup=f'NN: {nn_pt["name"]} | {nn_pt["top_category"]} | {nn_pt["id"]}',
                      icon=folium.Icon(color="green")).add_to(m)

        fn = f"{theme}_nn_q{qid}.html"
        m.save(str(theme_out / fn))
        links.append((f"NN qid={qid}", fn))

        # ---- BBOX ----
        bq = q["bbox"][qid]
        minx, miny, maxx, maxy = map(float, (bq["minx"], bq["miny"], bq["maxx"], bq["maxy"]))
        cqx, cqy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        clat, clon = inverse_project_m(cqx, cqy, prj)

        if backend == "indexes":
            idxs = idx_bbox_rtree(indexes["rtree"], minx, miny, maxx, maxy)
            pts = rows_by_idxs(idxs)
        else:
            ids = db_bbox(conn, minx, miny, maxx, maxy)
            pts = db_fetch_points_by_ids(conn, ids)

        m = build_map_base(clat, clon)
        add_title(m, f"""
        <div style="position: fixed; top: 10px; left: 10px; z-index:9999;
                    background: white; padding: 10px; border: 1px solid #999;">
            <b>Theme:</b> {theme}<br/>
            <b>Backend:</b> {backend}<br/>
            <b>Query:</b> BBOX (qid={qid})<br/>
            <b>Returned:</b> {len(pts)} points (plotted up to {args.max_markers})<br/>
        </div>
        """)
        # rectangle corners
        sw = inverse_project_m(minx, miny, prj)
        ne = inverse_project_m(maxx, maxy, prj)
        folium.Rectangle(bounds=[sw, ne], color="blue", weight=2, fill=False).add_to(m)

        cluster = MarkerCluster(name="BBOX results").add_to(m)
        for p in pts[: args.max_markers]:
            circle_marker(p["lat"], p["lon"],
                          popup=f'{p["name"]} | {p["top_category"]} | {p["id"]}',
                          color="blue", radius=4).add_to(cluster)

        fn = f"{theme}_bbox_q{qid}.html"
        m.save(str(theme_out / fn))
        links.append((f"BBOX qid={qid}", fn))

        # ---- RADIUS ----
        rq = q["radius"][qid]
        qx, qy, rr = float(rq["qx"]), float(rq["qy"]), float(rq["r"])
        qlat, qlon = inverse_project_m(qx, qy, prj)

        if backend == "indexes":
            idxs = idx_radius_kdtree(indexes["kdtree"], qx, qy, rr)
            pts = rows_by_idxs(idxs)
        else:
            ids = db_radius(conn, qx, qy, rr)
            pts = db_fetch_points_by_ids(conn, ids)

        m = build_map_base(qlat, qlon)
        add_title(m, f"""
        <div style="position: fixed; top: 10px; left: 10px; z-index:9999;
                    background: white; padding: 10px; border: 1px solid #999;">
            <b>Theme:</b> {theme}<br/>
            <b>Backend:</b> {backend}<br/>
            <b>Query:</b> RADIUS r={int(rr)}m (qid={qid})<br/>
            <b>Returned:</b> {len(pts)} points (plotted up to {args.max_markers})<br/>
        </div>
        """)
        folium.Marker((qlat, qlon), popup="Query center (Radius)", icon=folium.Icon(color="red")).add_to(m)
        folium.Circle((qlat, qlon), radius=rr, color="purple", weight=2, fill=False).add_to(m)

        cluster = MarkerCluster(name="Radius results").add_to(m)
        for p in pts[: args.max_markers]:
            circle_marker(p["lat"], p["lon"],
                          popup=f'{p["name"]} | {p["top_category"]} | {p["id"]}',
                          color="purple", radius=4).add_to(cluster)

        fn = f"{theme}_radius_q{qid}.html"
        m.save(str(theme_out / fn))
        links.append((f"RADIUS qid={qid}", fn))

        # ---- COMPOSITE ----
        cq = q["composite"][qid]
        qx, qy, rr = float(cq["qx"]), float(cq["qy"]), float(cq["r"])
        cat = str(cq["category_contains"])
        qlat, qlon = inverse_project_m(qx, qy, prj)

        if backend == "indexes":
            cand, filt = idx_composite(indexes["kdtree"], rows, qx, qy, rr, cat)
            pts_cand = rows_by_idxs(cand)
            pts_filt = rows_by_idxs(filt)
        else:
            cand_ids, filt_ids = db_composite(conn, qx, qy, rr, cat)
            pts_cand = db_fetch_points_by_ids(conn, cand_ids)
            pts_filt = db_fetch_points_by_ids(conn, filt_ids)

        m = build_map_base(qlat, qlon)
        add_title(m, f"""
        <div style="position: fixed; top: 10px; left: 10px; z-index:9999;
                    background: white; padding: 10px; border: 1px solid #999;">
            <b>Theme:</b> {theme}<br/>
            <b>Backend:</b> {backend}<br/>
            <b>Query:</b> COMPOSITE r={int(rr)}m + category contains "{cat}" (qid={qid})<br/>
            <b>Candidates:</b> {len(pts_cand)} | <b>Filtered:</b> {len(pts_filt)} (plotted up to {args.max_markers})<br/>
        </div>
        """)
        folium.Marker((qlat, qlon), popup="Query center (Composite)", icon=folium.Icon(color="red")).add_to(m)
        folium.Circle((qlat, qlon), radius=rr, color="gray", weight=2, fill=False).add_to(m)

        cluster_cand = MarkerCluster(name="Candidates (radius)").add_to(m)
        for p in pts_cand[: args.max_markers]:
            circle_marker(p["lat"], p["lon"],
                          popup=f'[cand] {p["name"]} | {p["top_category"]} | {p["id"]}',
                          color="gray", radius=3, fill_opacity=0.6).add_to(cluster_cand)

        cluster_filt = MarkerCluster(name="Filtered (composite)").add_to(m)
        for p in pts_filt[: args.max_markers]:
            circle_marker(p["lat"], p["lon"],
                          popup=f'[match] {p["name"]} | {p["top_category"]} | {p["id"]}',
                          color="green", radius=4, fill_opacity=0.9).add_to(cluster_filt)

        folium.LayerControl(collapsed=False).add_to(m)

        fn = f"{theme}_composite_q{qid}.html"
        m.save(str(theme_out / fn))
        links.append((f"COMPOSITE qid={qid}", fn))

        # Write index page for this theme/backend
        write_index_html(theme_out, links)

        print(f"[OK] Wrote maps for theme={theme} to {theme_out}")

    if conn is not None:
        conn.close()

    # Root index (links per theme)
    root_links = []
    for theme in all_qs.keys():
        root_links.append((f"{theme} (open theme index)", f"{theme}/index.html"))
    write_index_html(out_root, root_links)
    print(f"[DONE] Open: {out_root / 'index.html'}")


if __name__ == "__main__":
    main()
