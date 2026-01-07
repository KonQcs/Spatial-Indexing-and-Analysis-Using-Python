<img width="158" height="158" alt="UTH_greek_logo" src="https://github.com/user-attachments/assets/b1fca0da-b2ea-463b-844f-8380583ac5ac" />


# Spatial Indexing & Visualization of POIs in Milan (Foursquare Places API)


Course project for **Databases II**.  
Team 3 scope: **restaurants**, **theaters**, **libraries** in **Milan (Milano)** using the **Foursquare Places API**.

This repository contains **code only**. The folders **`data/`** and **`reports/`** are generated locally (they are not included in the repository because they can be large and may contain raw API responses).

The project implements:
- POI collection from Foursquare Places API using **adaptive spatial tiling** (grid/tiles) and raw-response storage.
- Data cleaning/preprocessing (duplicates, invalid coordinates, Milan-area checks) and production of a clean dataset.
- Indexing and benchmarking: **KD-tree**, **Quad-tree**, **R-tree** (multiple parameterizations) + comparison with **SQLite RTree**.
- Spatial queries: Nearest Neighbor, Bounding Box, Radius (500m), Composite (spatial + attribute filter).
- Interactive visualization of query results using **Folium**.

---

## Contents
- [Requirements](#requirements)
- [How to get your own Foursquare API Key](#how-to-get-your-own-foursquare-api-key)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run pipeline (from scratch)](#run-pipeline-from-scratch)
- [Generated outputs (local)](#generated-outputs-local)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Requirements
- Python 3.11+ (or any modern Python 3.x)
- Windows / Linux / macOS (tested mainly on Windows)
- Python packages:
  - `requests`, `pandas`, `numpy`, `scipy`
  - `rtree` (libspatialindex backend)
  - `pyqtree`
  - `folium`

> Note: `rtree` depends on libspatialindex. On Windows it usually works via pip wheels. See Troubleshooting if needed.

---

## How to get your own Foursquare API Key

This project uses the **Foursquare Places API**. To run the extraction script, you must generate your own API key from the Foursquare Developer Console.

1. Create / log in to your **Foursquare Developer account**.
2. Create a **new Project** in the Developer Console.
3. Generate a **Service API Key** for that project (this is the key you will use as a Bearer token).
4. Copy and securely store the key (you will typically see it only once).
5. Use the key in requests as:
   - HTTP header: `Authorization: Bearer <YOUR_KEY>`

> Security note: **Do not commit API keys to GitHub**. Use environment variables or a local `.env` that is excluded via `.gitignore`.

---

## Installation

### 1) Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Linux/macOS (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies
```bash
pip install requests pandas numpy scipy rtree pyqtree folium
```

---

## Configuration

### 1) Set the Foursquare API Key (recommended)
Windows (PowerShell):
```powershell
$env:FSQ_API_KEY="PUT_YOUR_KEY_HERE"
```

Linux/macOS (bash/zsh):
```bash
export FSQ_API_KEY="PUT_YOUR_KEY_HERE"
```

### 2) (Optional) Create the local folders (if scripts do not create them automatically)

Windows (PowerShell):
```powershell
mkdir data, reports -ErrorAction SilentlyContinue
mkdir data\raw, data\clean -ErrorAction SilentlyContinue
mkdir reports\maps -ErrorAction SilentlyContinue
```

Linux/macOS:
```bash
mkdir -p data/raw data/clean reports/maps
```

---

## Run pipeline (from scratch)

> The scripts below are the final project entry points:
> - `extractfromFoursquare.py`
> - `cleaningReport.py`
> - `indexesBenchmark.py`
> - `sqliteBenchmark.py`
> - `foliumVisualize.py`

### Step 1 — Extract data (Foursquare → raw JSON + SQLite)
```bash
python extractfromFoursquare.py
```

Expected (locally):
- raw responses under `data/raw/`
- SQLite database created/updated (e.g., `data/milano_places.sqlite`)

### Step 2 — Clean & preprocess
```bash
python cleaningReport.py
```

Expected (locally):
- `reports/cleaning_report.json`
- clean export under `data/clean/` (e.g., JSONL/CSV depending on implementation)

### Step 3 — Benchmark spatial indexes (KD/Quad/R)
Recommended with 5 repeats:
```bash
python indexesBenchmark.py --repeats 5
```

Expected (locally) under `reports/`:
- `index_build_times_allthemes.csv`
- `index_query_times_allthemes.csv`
- `index_query_summary_allthemes.csv`
- `query_sets_allthemes.json`

### Step 4 — Benchmark database (SQLite RTree)
```bash
python sqliteBenchmark.py --repeats 5
```

Expected (locally) under `reports/`:
- `db_build_times_allthemes.csv`
- `db_query_times_allthemes.csv`
- `db_query_summary_allthemes.csv`

### Step 5 — Visualize queries (Folium)
Run visualization for a specific query id (`qid` ranges from 0 to 9):
```bash
python foliumVisualize.py --qid 0
python foliumVisualize.py --qid 9
```

If your visualization script supports choosing backend:
```bash
python foliumVisualize.py --backend indexes --qid 9
python foliumVisualize.py --backend db --qid 9
```

Expected (locally):
- HTML maps under `reports/maps/...`
- Open the produced `index.html` (if generated) in a browser.

---

## Generated outputs (local)

Because `data/` and `reports/` are not committed, you will generate these locally by running the pipeline.

### `data/`
- `data/raw/` — raw API responses (JSON) per request
- `data/clean/` — cleaned dataset exports
- `data/milano_places.sqlite` — SQLite database (deliverable copy)

### `reports/`
- `reports/cleaning_report.json` — cleaning summary (duplicates, invalid coords, bbox checks)
- `reports/query_sets_allthemes.json` — 10 queries per theme per query type
- `reports/index_*_allthemes.csv` — index benchmark outputs
- `reports/db_*_allthemes.csv` — DB benchmark outputs
- `reports/maps/` — Folium HTML maps for NN/BBOX/Radius/Composite

---

## Troubleshooting

### 1) RTreeError: NearMinimumOverlapFactor / capacity mismatch
If you encounter an error similar to:
`NearMinimumOverlapFactor must be ... less than both index and leaf capacities`

Then `near_minimum_overlap_factor` must be strictly **less than** both `index_capacity` and `leaf_capacity`.
Fix by lowering the overlap factor or increasing capacities consistently (the benchmark script includes safe parameter choices).

### 2) Installing rtree / libspatialindex
- Try a clean venv: `pip install rtree`
- If it fails, use an appropriate wheel for your OS/Python version or use conda.

### 3) “I only see a few points on the map”
Folium maps visualize **results of a single query** (selected by `qid`), not the full dataset.
Change `--qid` to visualize a different location, or implement an “all points” overview map (cluster/heatmap).

---

## References
- Foursquare Places API documentation (Get Started, API keys)
- SciPy `scipy.spatial.KDTree` documentation
- `rtree` / libspatialindex documentation
- Pyqtree documentation
- SQLite RTree module documentation
- Folium documentation
