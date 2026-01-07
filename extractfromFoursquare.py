from __future__ import annotations

import csv
import json
import math
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG
# =========================
API_URL = "https://places-api.foursquare.com/places/search"
API_VERSION = "2025-06-17"

# Hardcoded key
FSQ_API_KEY = "___PUT___YOUR___KEY___"

HEADERS = {
    "accept": "application/json",
    "X-Places-Api-Version": API_VERSION,
    "authorization": f"Bearer {FSQ_API_KEY}",
}

# Group 3
CATEGORIES = {
    "theaters": "4bf58dd8d48988d137941735",   # Theater
    "libraries": "4bf58dd8d48988d12f941735",  # Library
    "restaurants": "4d4b7105d754a06374d81259" # Restaurant
}

FIELDS = ",".join([
    "fsq_place_id",
    "name",
    "latitude",
    "longitude",
    "categories",
    "location",
])

MILANO_BBOX = (45.38, 9.04, 45.56, 9.28)  # south, west, north, east

GRID_NX = 10
GRID_NY = 10

# Split logic
MAX_SPLIT_DEPTH = 4          # max depth split
MIN_TILE_RADIUS_M = 200      # min splited tile
MAX_PAGES_PER_TILE = 20      # safety cap pagination

SLEEP_BETWEEN_CALLS_SEC = 0.15


# =========================
# Paths
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
REPORTS_DIR = SCRIPT_DIR / "reports"
DB_PATH = DATA_DIR / "milano_places.sqlite"


# =========================
# Helpers
# =========================
@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    def split4(self) -> List["BBox"]:
        mid_lat = (self.south + self.north) / 2.0
        mid_lon = (self.west + self.east) / 2.0
        return [
            BBox(self.south, self.west, mid_lat, mid_lon),   # SW
            BBox(self.south, mid_lon, mid_lat, self.east),   # SE
            BBox(mid_lat, self.west, self.north, mid_lon),   # NW
            BBox(mid_lat, mid_lon, self.north, self.east),   # NE
        ]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b.south + b.north) / 2.0, (b.west + b.east) / 2.0)

def bbox_radius_to_cover_m(b: BBox) -> int:
    clat, clon = bbox_center(b)
    corners = [(b.south, b.west), (b.south, b.east), (b.north, b.west), (b.north, b.east)]
    return int(max(haversine_m(clat, clon, lat, lon) for lat, lon in corners) + 20)

def make_initial_grid(b: BBox, nx: int, ny: int) -> List[BBox]:
    lat_step = (b.north - b.south) / ny
    lon_step = (b.east - b.west) / nx
    tiles: List[BBox] = []
    for iy in range(ny):
        for ix in range(nx):
            south = b.south + iy * lat_step
            north = b.south + (iy + 1) * lat_step
            west = b.west + ix * lon_step
            east = b.west + (ix + 1) * lon_step
            tiles.append(BBox(south, west, north, east))
    return tiles


# =========================
# SQLite schema
# =========================
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS places (
  fsq_place_id TEXT PRIMARY KEY,
  name TEXT,
  lat REAL,
  lon REAL,
  categories_json TEXT,
  location_json TEXT,
  themes_json TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS api_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT,
  theme TEXT,
  tile_id TEXT,
  depth INTEGER,
  page INTEGER,
  url TEXT,
  params_json TEXT,
  status_code INTEGER,
  results_count INTEGER,
  raw_path TEXT,
  error TEXT
);
"""

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn

def db_upsert_place(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO places (fsq_place_id, name, lat, lon, categories_json, location_json, themes_json, raw_json)
        VALUES (:fsq_place_id, :name, :lat, :lon, :categories_json, :location_json, :themes_json, :raw_json)
        ON CONFLICT(fsq_place_id) DO UPDATE SET
          name=excluded.name,
          lat=excluded.lat,
          lon=excluded.lon,
          categories_json=excluded.categories_json,
          location_json=excluded.location_json,
          themes_json=excluded.themes_json,
          raw_json=excluded.raw_json
        """,
        row
    )

def db_log_call(conn: sqlite3.Connection, log: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO api_calls (ts_utc, theme, tile_id, depth, page, url, params_json, status_code, results_count, raw_path, error)
        VALUES (:ts_utc, :theme, :tile_id, :depth, :page, :url, :params_json, :status_code, :results_count, :raw_path, :error)
        """,
        log
    )


# =========================
# API + pagination
# =========================
def places_search(params: Dict[str, Any]) -> Tuple[Dict[str, Any], requests.Response]:
    r = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json(), r

def parse_cursor_from_link_header(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    if "<" not in link_header or ">" not in link_header:
        return None
    url = link_header.split("<", 1)[1].split(">", 1)[0]
    if "cursor=" not in url:
        return None
    cursor = url.split("cursor=", 1)[1].split("&", 1)[0].strip()
    return cursor or None


# =========================
# Core: collect theme with split-on-50
# =========================
def collect_theme(theme: str, category_id: str, bbox: BBox, conn: sqlite3.Connection) -> Dict[str, Any]:
    tiles = make_initial_grid(bbox, GRID_NX, GRID_NY)

    places_by_id: Dict[str, Dict[str, Any]] = {}

    api_csv_path = REPORTS_DIR / "api_calls.csv"
    csv_exists = api_csv_path.exists()

    with api_csv_path.open("a", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "ts_utc", "theme", "tile_id", "depth", "page", "url", "params_json",
                "status_code", "results_count", "raw_path", "error"
            ]
        )
        if not csv_exists:
            writer.writeheader()

        stack: List[Tuple[str, BBox, int]] = [(f"T{idx:04d}", t, 0) for idx, t in enumerate(tiles)]
        req_counter = 0

        while stack:
            tile_id, tile_bbox, depth = stack.pop()

            clat, clon = bbox_center(tile_bbox)
            radius = min(bbox_radius_to_cover_m(tile_bbox), 100000)

            base_params = {
                "ll": f"{clat},{clon}",
                "radius": radius,
                "limit": 50,
                "sort": "RELEVANCE",
                "fsq_category_ids": category_id,
                "fields": FIELDS,
            }

            # ---- split trigger flag for THIS tile
            tile_hit_limit = False

            page = 1
            cursor: Optional[str] = None
            pages_seen = 0

            while True:
                params = dict(base_params)
                if cursor:
                    params["cursor"] = cursor

                req_counter += 1
                err = None
                status_code = 200
                payload: Dict[str, Any] = {}
                raw_path = ""
                link = None

                try:
                    payload, resp = places_search(params)
                    status_code = resp.status_code
                    link = resp.headers.get("link")
                    cursor = parse_cursor_from_link_header(link)
                except Exception as e:
                    err = str(e)
                    status_code = getattr(getattr(e, "response", None), "status_code", 0) or 0
                    payload = {}
                    cursor = None

                results = payload.get("results", []) if isinstance(payload, dict) else []
                results_count = len(results)

                if results_count == 50:
                    tile_hit_limit = True

                pages_seen += 1

                # write raw JSON
                raw_fname = f"{theme}_{tile_id}_d{depth}_p{page:02d}_req{req_counter:06d}.json"
                raw_path = str(RAW_DIR / raw_fname)
                (RAW_DIR / raw_fname).write_text(
                    json.dumps(
                        {
                            "meta": {
                                "theme": theme,
                                "tile_id": tile_id,
                                "depth": depth,
                                "page": page,
                                "params": params,
                                "ts_utc": utc_now_iso(),
                                "status_code": status_code,
                                "link": link,
                            },
                            "data": payload,
                        },
                        ensure_ascii=False,
                        indent=2
                    ),
                    encoding="utf-8"
                )

                log_row = {
                    "ts_utc": utc_now_iso(),
                    "theme": theme,
                    "tile_id": tile_id,
                    "depth": depth,
                    "page": page,
                    "url": API_URL,
                    "params_json": json.dumps(params, ensure_ascii=False),
                    "status_code": int(status_code),
                    "results_count": int(results_count),
                    "raw_path": raw_path,
                    "error": err,
                }
                writer.writerow(log_row)
                db_log_call(conn, log_row)
                conn.commit()

                if err is None:
                    for p in results:
                        pid = p.get("fsq_place_id")
                        if not pid:
                            continue
                        existing = places_by_id.get(pid)
                        if existing is None:
                            p["_themes"] = [theme]
                            places_by_id[pid] = p
                        else:
                            themes = set(existing.get("_themes", []))
                            themes.add(theme)
                            existing["_themes"] = sorted(themes)

                # end pagination?
                if not cursor:
                    break

                page += 1

                if pages_seen >= MAX_PAGES_PER_TILE:
                    # safety: stop pagination, will split below
                    break

                time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            # ---- Split-on-50 OR "too many pages"
            if (tile_hit_limit or pages_seen >= MAX_PAGES_PER_TILE) and depth < MAX_SPLIT_DEPTH and radius > MIN_TILE_RADIUS_M:
                for i, sub in enumerate(tile_bbox.split4()):
                    stack.append((f"{tile_id}_S{i}", sub, depth + 1))

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # upsert places to DB
    for pid, p in places_by_id.items():
        row = {
            "fsq_place_id": pid,
            "name": p.get("name"),
            "lat": float(p.get("latitude")) if p.get("latitude") is not None else None,
            "lon": float(p.get("longitude")) if p.get("longitude") is not None else None,
            "categories_json": json.dumps(p.get("categories", []), ensure_ascii=False),
            "location_json": json.dumps(p.get("location", {}), ensure_ascii=False),
            "themes_json": json.dumps(p.get("_themes", [theme]), ensure_ascii=False),
            "raw_json": json.dumps(p, ensure_ascii=False),
        }
        db_upsert_place(conn, row)
    conn.commit()

    return {"theme": theme, "unique_places": len(places_by_id), "places": list(places_by_id.values())}




def main() -> None:

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = db_connect()
    bbox = BBox(*MILANO_BBOX)

    all_places: Dict[str, Dict[str, Any]] = {}
    summaries: List[Dict[str, Any]] = []

    for theme, cat_id in CATEGORIES.items():
        print(f"\n=== Collecting: {theme} ===")
        s = collect_theme(theme, cat_id, bbox, conn)
        summaries.append({"theme": theme, "unique_places": s["unique_places"]})
        for p in s["places"]:
            pid = p.get("fsq_place_id")
            if not pid:
                continue
            existing = all_places.get(pid)
            if existing is None:
                all_places[pid] = p
            else:
                themes = set(existing.get("_themes", [])) | set(p.get("_themes", []))
                existing["_themes"] = sorted(themes)

    # write clean JSONL
    clean_path = CLEAN_DIR / "places_clean.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for p in all_places.values():
            rec = {
                "fsq_place_id": p.get("fsq_place_id"),
                "name": p.get("name"),
                "lat": p.get("latitude"),
                "lon": p.get("longitude"),
                "categories": p.get("categories", []),
                "location": p.get("location", {}),
                "themes": p.get("_themes", []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = REPORTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_utc": utc_now_iso(),
                "milano_bbox": MILANO_BBOX,
                "grid": {"nx": GRID_NX, "ny": GRID_NY},
                "split": {"max_depth": MAX_SPLIT_DEPTH, "min_tile_radius_m": MIN_TILE_RADIUS_M},
                "pagination": {"max_pages_per_tile": MAX_PAGES_PER_TILE},
                "summaries": summaries,
                "total_unique_places_all_themes": len(all_places),
                "outputs": {
                    "raw_dir": str(RAW_DIR),
                    "clean_jsonl": str(clean_path),
                    "sqlite_db": str(DB_PATH),
                    "api_calls_csv": str(REPORTS_DIR / "api_calls.csv"),
                },
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    conn.close()

    print("\nDONE.")
    print("Raw JSON:", RAW_DIR)
    print("Clean JSONL:", clean_path)
    print("SQLite DB:", DB_PATH)
    print("API calls CSV:", REPORTS_DIR / "api_calls.csv")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()

