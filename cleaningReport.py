from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple



ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "milano_places.sqlite"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MILANO_BBOX = (45.38, 9.04, 45.56, 9.28)  # south, west, north, east
BBOX_MARGIN = 0.08  # degrees (safety margin due to radius search)


@dataclass(frozen=True)
class AuditResult:
    n_rows: int
    n_distinct_ids: int
    n_missing_coords: int
    n_invalid_range: int
    n_outside_milano: int
    n_missing_name: int
    n_missing_categories: int
    total_returned_rows: int
    unique_places: int


def q1(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> Any:
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


def audit(db_path: Path) -> AuditResult:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    s, w, n, e = MILANO_BBOX
    m = BBOX_MARGIN

    conn = sqlite3.connect(db_path)

    n_rows = q1(conn, "SELECT COUNT(*) FROM places")
    n_distinct = q1(conn, "SELECT COUNT(DISTINCT fsq_place_id) FROM places")

    n_missing_coords = q1(conn, "SELECT COUNT(*) FROM places WHERE lat IS NULL OR lon IS NULL")
    n_invalid_range = q1(conn, """
        SELECT COUNT(*) FROM places
        WHERE lat NOT BETWEEN -90 AND 90 OR lon NOT BETWEEN -180 AND 180
    """)

    n_outside = q1(conn, """
        SELECT COUNT(*) FROM places
        WHERE lat < ? OR lat > ? OR lon < ? OR lon > ?
    """, (s - m, n + m, w - m, e + m))

    n_missing_name = q1(conn, """
        SELECT COUNT(*) FROM places
        WHERE name IS NULL OR TRIM(name) = ''
    """)

    n_missing_categories = q1(conn, """
        SELECT COUNT(*) FROM places
        WHERE categories_json IS NULL OR categories_json = '[]'
    """)

    total_returned = q1(conn, "SELECT COALESCE(SUM(results_count),0) FROM api_calls")
    unique_places = n_rows

    conn.close()

    return AuditResult(
        n_rows=int(n_rows),
        n_distinct_ids=int(n_distinct),
        n_missing_coords=int(n_missing_coords),
        n_invalid_range=int(n_invalid_range),
        n_outside_milano=int(n_outside),
        n_missing_name=int(n_missing_name),
        n_missing_categories=int(n_missing_categories),
        total_returned_rows=int(total_returned),
        unique_places=int(unique_places),
    )


def main() -> None:
    res = audit(DB_PATH)

    decisions = {
        "deduplication": "Unique place identity enforced by fsq_place_id (primary key) and UPSERT.",
        "coordinates": [
            "Drop records with missing lat/lon when exporting clean dataset.",
            "Flag invalid coordinate ranges: lat not in [-90,90] or lon not in [-180,180].",
            f"Flag out-of-area records outside Milano bbox {MILANO_BBOX} with margin {BBOX_MARGIN}."
        ],
        "kept_fields_in_clean_jsonl": [
            "fsq_place_id", "name", "lat", "lon", "categories", "location", "themes"
        ],
        "notes": [
            "Grid + radius search introduces overlap; duplicates expected in raw API returns, removed in final places table.",
        ],
    }

    report: Dict[str, Any] = {
        "db_path": str(DB_PATH),
        "milano_bbox": MILANO_BBOX,
        "bbox_margin": BBOX_MARGIN,
        "audit": res.__dict__,
        "decisions": decisions,
    }

    out = REPORTS_DIR / "cleaning_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Cleaning audit written to:", out)
    print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
