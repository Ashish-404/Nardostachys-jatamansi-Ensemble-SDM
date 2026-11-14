=====================================================================================
 Ensemble species distribution model for Nardostachys jatamansi, far-western Nepal
 (Maxent + GLM + random forest; WorldClim 2.1 baseline and CMIP6 BCC-CSM2-MR futures)
=====================================================================================


# Import LIBRARIES

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import sys
import re
import json
import time
import argparse
import platform
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import elapid as ela
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")


# STEP 1. CONFIGURATION 

BASE_DIR = Path(r"E:\maxent\Mission Jatamansi")
OUTPUT_DIR = Path(r"E:\maxent\result\OUTPUT_v31_with_Dolpa")

OCCURRENCE_CSV = BASE_DIR / "N_jatamansi_thinned_1km.csv"
LON_COL, LAT_COL = "Longitude", "Latitude"

# Study area = old six-district polygon + Dolpa. If STUDY_AREA_SHP does not exist yet, the script builds
# it from OLD_STUDY_AREA_SHP and the Nepal district layer (DISTRICTS_SHP), and saves it to STUDY_AREA_SHP.
STUDY_AREA_SHP = BASE_DIR / "Input" / "output" / "study_area" / "study_area_with_dolpa.shp"
OLD_STUDY_AREA_SHP = BASE_DIR / "Input" / "output" / "study_area" / "study_area.shp"
DISTRICTS_SHP = Path(r"E:\Arc GIS\ArcMAP\Input\districts_2023_July_05_11_51_17\districts")  # .shp, or its folder
ADD_DISTRICTS = ["Dolpa"]                      # district names added to the old study area (case-insensitive)

# Original WorldClim 2.1 30-second files (uncut). All seven sit in one folder: the unzipped historical
# "wc2.1_30s_bio" folder (wc2.1_30s_bio_1.tif ... _19.tif) and the six CMIP6 "bc" files, names unchanged.
GCM_NAME = "BCC-CSM2-MR"
WORLDCLIM_DIR = Path(r"E:\world clim")
BASELINE_CLIMATE = WORLDCLIM_DIR / "wc2.1_30s_bio"                    # folder of 19 files
FUTURE_CLIMATE = {                                                     # one 19-band file each
    f"SSP{ssp}_{p0}_{p1}": WORLDCLIM_DIR / f"wc2.1_30s_bioc_{GCM_NAME}_ssp{ssp}_{p0}-{p1}.tif"
    for ssp in ("245", "585") for p0, p1 in (("2021", "2040"), ("2061", "2080"), ("2081", "2100"))
}
# Digital elevation model covering the whole study area INCLUDING DOLPA (any CRS). If your SRTM file was
# clipped to the old six districts, download SRTM 1-arc-second or Copernicus GLO-30 tiles for the area
# and mosaic them (do not use the 1-km WorldClim elevation: slopes from it are meaningless at 90 m).
DEM_PATH = Path(r"E:\Arc GIS\Q GIS\Input\Nepal_Copernicus_DEM_30m.tif")   # a file, or a folder of tiles

# ---- Modelling choices (defaults explained in the manuscript Methods) -----------------------
RANDOM_STATE = 42
N_BACKGROUND = 10_000
CORR_THRESHOLD = 0.70            # pairwise |r| above which one variable of a pair is removed
VIF_THRESHOLD = 10.0
# Ranked priority list: when two correlated variables are both listed, the earlier one is kept.
#   bio10 = mean temperature of warmest quarter (growing-season warmth, thermal limit of alpine
#   plants); bio19 = precipitation of coldest quarter (winter snow); bio18 = precipitation of
#   warmest quarter (monsoon); bio15 = precipitation seasonality.
PRIORITY_VARS = ["bio10", "bio19", "bio18", "bio15", "bio2", "slope"]
# At least one of these temperature-level variables must survive screening.
THERMAL_VARS = ["bio1", "bio5", "bio6", "bio8", "bio9", "bio10", "bio11"]
EXCLUDE_FROM_MODEL = []              # e.g. ["elevation"] to keep the elevation classes out of the model
FORCE_KEEP = ["aspect"]              # always retained: aspect is nominal, so correlation screening does not apply.
# Add "elevation" and/or "tri" here to keep them even if |r| > 0.70 with a retained variable (e.g. temperature).
TERRAIN_RES_M = 30                   # slope, aspect and TRI are derived at this resolution (m, UTM 44N)
FLAT_SLOPE_DEG = 0.0                 # aspect class "Flat" where slope <= this value (degrees)
ELEV_BREAKS = [2500, 3000, 3500, 3900, 4300, 4700]   # elevation class limits (m)
UTM_CRS = "EPSG:32644"

N_FOLDS = 4
BLOCK_SIZE_KM = 40                   # side of square spatial blocks for cross-validation
MAXENT_FEATURES = [["linear"], ["linear", "quadratic"], ["linear", "quadratic", "hinge"]]
MAXENT_BETAS = [0.5, 1.0, 2.0, 3.0, 5.0]
GLM_C = [0.01, 0.1, 1.0, 10.0]
GLM_QUADRATIC = [False, True]
RF_MAX_DEPTH = [5, 10, None]
RF_MIN_LEAF = [1, 5, 20, 50]
RF_TREES = 500
MIN_ENSEMBLE_AUC = 0.70              # algorithms below this CV AUC are left out of the ensemble

# Maximum sensitivity + specificity (maxSSS = max TSS), computed on out-of-fold predictions; recommended for
# presence-only data (Liu et al. 2013). P10 is not used as primary: out-of-fold P10 collapses when one spatial
# fold is predicted poorly (first full run: P10 = 0.006, i.e. almost the whole study area "suitable").
PRIMARY_THRESHOLD = "MTSS"
MESS_REFERENCE = "presence"          # "presence" (observed niche) or "calibration" (presence+background)
MESS_THRESHOLD = -20.0
N_BOOTSTRAP = 100
N_PERMUTATIONS = 5                   # permutation-importance repeats per outer fold
COVERAGE_TOLERANCE = 0.005           # max share of study-area cells allowed to lack data in a layer
# options
DROP_RECORDS_CSV = None              # e.g. BASE_DIR / "records_to_drop.csv" (columns Longitude, Latitude); None = keep all
LOW_ELEVATION_FLAG_M = 2500          # presences in cells below this mean elevation are flagged for checking
UNPREDICTED_FLAG = 0.02              # presences with out-of-fold suitability below this are flagged
MESS_SENS_MIN_IMPORTANCE = 0.01      # sensitivity run: MESS only on predictors with nested importance >= this
DRAW_DISTRICTS = True                # draw district boundaries/names on maps (uses DISTRICTS_SHP if it exists)
DUPLICATE_SHARE = 0.99               # two future scenarios sharing > 99% identical cells = duplicate

SCENARIO_LABEL = {
    "Baseline_1970_2000": "Baseline 1970–2000",
    "SSP245_2021_2040": "SSP2-4.5, 2021–2040", "SSP245_2061_2080": "SSP2-4.5, 2061–2080",
    "SSP245_2081_2100": "SSP2-4.5, 2081–2100", "SSP585_2021_2040": "SSP5-8.5, 2021–2040",
    "SSP585_2061_2080": "SSP5-8.5, 2061–2080", "SSP585_2081_2100": "SSP5-8.5, 2081–2100",
}
VAR_LABEL = {
    "bio1": "Annual mean temperature (°C)", "bio2": "Mean diurnal range (°C)", "bio3": "Isothermality",
    "bio4": "Temperature seasonality", "bio5": "Max. temperature warmest month (°C)",
    "bio6": "Min. temperature coldest month (°C)", "bio7": "Temperature annual range (°C)",
    "bio8": "Mean temperature wettest quarter (°C)", "bio9": "Mean temperature driest quarter (°C)",
    "bio10": "Mean temperature warmest quarter (°C)", "bio11": "Mean temperature coldest quarter (°C)",
    "bio12": "Annual precipitation (mm)", "bio13": "Precipitation wettest month (mm)",
    "bio14": "Precipitation driest month (mm)", "bio15": "Precipitation seasonality (CV)",
    "bio16": "Precipitation wettest quarter (mm)", "bio17": "Precipitation driest quarter (mm)",
    "bio18": "Precipitation warmest quarter (mm)", "bio19": "Precipitation coldest quarter (mm)",
    "aspect": "Aspect (class)", "elevation": "Elevation class (m)", "slope": "Slope (°)",
    "tri": "Terrain ruggedness index (m)", "elevation_m": "Elevation (m)",
}
BIO = [f"bio{i}" for i in range(1, 20)]
TERRAIN = ["aspect", "elevation", "slope", "tri"]          # variables 20, 21, 22, 23
VAR_NUMBER = {**{f"bio{i}": i for i in range(1, 20)}, "aspect": 20, "elevation": 21, "slope": 22, "tri": 23}
ASPECT_LABELS = {0: "Flat", 1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW", 7: "W", 8: "NW"}
_eb = ELEV_BREAKS
ELEV_LABELS = {1: f"<{_eb[0]}", **{i + 2: f"{_eb[i]}–{_eb[i+1]}" for i in range(len(_eb) - 1)}, len(_eb) + 1: f">{_eb[-1]}"}
CATEGORICAL = {"aspect": ASPECT_LABELS, "elevation": ELEV_LABELS}   # code -> label



# STEP 2. COMMAND LINE, OUTPUT FOLDERS AND LOG

def parse_args():
    ap = argparse.ArgumentParser(description="N. jatamansi ensemble SDM")
    ap.add_argument("--check", action="store_true", help="run input checks only")
    ap.add_argument("--bootstrap", type=int, default=None, help="number of bootstrap replicates")
    ap.add_argument("--quick", action="store_true", help="small grids and few replicates (test run)")
    ap.add_argument("--drop", default=None, help="CSV of records to exclude (Longitude, Latitude columns)")
    ap.add_argument("--out", default=None, help="output folder (overrides OUTPUT_DIR)")
    return ap.parse_args()


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


def setup_output():
    dirs = {k: OUTPUT_DIR / k for k in ("rasters", "figures", "tables")}
    for d in [OUTPUT_DIR, *dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "run_log.txt")
    return dirs


def say(msg=""):
    print(msg, flush=True)


def section(title):
    say("\n" + "=" * 78 + f"\n {title}\n" + "=" * 78)


def write_versions(path):
    import sklearn, scipy, shapely, pyproj, matplotlib as mpl
    mods = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "geopandas": gpd.__version__, "rasterio": rasterio.__version__, "shapely": shapely.__version__,
            "pyproj": pyproj.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "elapid": getattr(ela, "__version__", "unknown"), "matplotlib": mpl.__version__,
            "GDAL": rasterio.__gdal_version__}
    with open(path, "w") as f:
        for k, v in mods.items():
            f.write(f"{k} {v}\n")
    say("Software: " + "; ".join(f"{k} {v}" for k, v in mods.items()))
    return mods


# STEP 3. MASTER GRID FROM THE STUDY AREA AND THE BASELINE CLIMATE

def bio_number(stem):
    m = re.search(r"bio_?(\d{1,2})$", stem.lower())
    return int(m.group(1)) if m else None


def find_bio_files(folder):
    folder = Path(folder)
    files = {}
    for p in folder.iterdir():
        if p.suffix.lower() in (".tif", ".tiff", ".asc", ".img"):
            n = bio_number(p.stem)
            if n and 1 <= n <= 19:
                files[f"bio{n}"] = p
    missing = [b for b in BIO if b not in files]
    if missing:
        raise FileNotFoundError(f"{folder}: missing {missing}")
    return files


def resolve_shp(p):
    """Accept a .shp file, a path without the extension, or a folder holding one shapefile."""
    p = Path(p)
    if p.is_file():
        return p
    if p.with_suffix(".shp").is_file():
        return p.with_suffix(".shp")
    if p.is_dir():
        shps = sorted(p.glob("*.shp"))
        if len(shps) == 1:
            return shps[0]
    return p


def check_paths():
    """List every input path from STEP 1 and stop once, naming all that are missing."""
    if Path(STUDY_AREA_SHP).is_file():
        sa_items = [("Study-area shapefile", STUDY_AREA_SHP, "file")]
    else:
        sa_items = [("Old study area (to add Dolpa)", OLD_STUDY_AREA_SHP, "file"),
                    ("Nepal district shapefile", resolve_shp(DISTRICTS_SHP), "file")]
    items = [("Occurrence CSV", OCCURRENCE_CSV, "file"), *sa_items,
             ("Baseline climate folder", BASELINE_CLIMATE, "dir"),
             *[(f"Future climate {k}", v, "any") for k, v in FUTURE_CLIMATE.items()],
             ("DEM (file or folder of tiles)", DEM_PATH, "any")]
    missing = []
    say("Input paths (STEP 1):")
    for label, path, kind in items:
        p = Path(path)
        ok = p.is_file() if kind == "file" else p.is_dir() if kind == "dir" else p.exists()
        say(f"  [{'ok' if ok else 'MISSING'}] {label:32s} {p}")
        if not ok:
            missing.append(label)
    if missing:
        raise SystemExit("\nSTOP: these inputs were not found: " + ", ".join(missing) +
                         ".\nEdit the paths in STEP 1 so they point to your actual files and folders.")


def build_study_area():
    """Old study area + the districts in ADD_DISTRICTS, merged and saved to STUDY_AREA_SHP."""
    old = gpd.read_file(OLD_STUDY_AREA_SHP)
    dist = gpd.read_file(resolve_shp(DISTRICTS_SHP))
    if old.crs is None or dist.crs is None:
        raise ValueError("The old study area or the district shapefile has no CRS (.prj missing).")
    dist = dist.to_crs(old.crs)
    wanted = {n.strip().lower() for n in ADD_DISTRICTS}
    text_cols = [c for c in dist.columns if c != "geometry" and
                 (pd.api.types.is_object_dtype(dist[c]) or pd.api.types.is_string_dtype(dist[c]))]
    name_col = next((c for c in text_cols if dist[c].astype(str).str.strip().str.lower().isin(wanted).any()), None)
    if name_col is None:
        raise SystemExit(f"STOP: no column of the district layer contains {ADD_DISTRICTS}. Text columns: {text_cols}. "
                         f"Examples: {dist[text_cols].head(3).to_dict('records')}. Fix the spelling in ADD_DISTRICTS.")
    sel = dist[dist[name_col].astype(str).str.strip().str.lower().isin(wanted)]
    merged = gpd.GeoSeries(list(old.geometry) + list(sel.geometry), crs=old.crs)
    merged = merged.union_all() if hasattr(merged, "union_all") else merged.unary_union
    eps = 50 if old.crs.is_projected else 0.0005          # close slivers between the two boundary sources
    merged = merged.buffer(eps).buffer(-eps)
    out = gpd.GeoDataFrame({"name": ["study_area_with_dolpa"]}, geometry=[merged], crs=old.crs)
    Path(STUDY_AREA_SHP).parent.mkdir(parents=True, exist_ok=True)
    out.to_file(STUDY_AREA_SHP)
    eq = "EPSG:6933"
    say(f"  built new study area: old {old.to_crs(eq).area.sum()/1e6:,.0f} km2 + {', '.join(ADD_DISTRICTS)} "
        f"{sel.to_crs(eq).area.sum()/1e6:,.0f} km2 (column '{name_col}') -> {out.to_crs(eq).area.sum()/1e6:,.0f} km2")
    if merged.geom_type == "MultiPolygon":
        say(f"  NOTE: the new study area has {len(merged.geoms)} separate parts; check it in GIS.")
    say(f"  saved to {STUDY_AREA_SHP}")


def load_study_area():
    if not Path(STUDY_AREA_SHP).is_file():
        build_study_area()
    gdf = gpd.read_file(STUDY_AREA_SHP)
    if gdf.crs is None:
        raise ValueError("Study-area shapefile has no CRS (.prj missing).")
    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    return gdf, geom


def build_master_grid(geom, baseline_files):
    """Grid = the native WorldClim 30-second grid, cropped to the study-area bounds."""
    with rasterio.open(baseline_files["bio1"]) as src:
        crs = src.crs or rasterio.crs.CRS.from_epsg(4326)
        res = src.res[0]
        minx, miny, maxx, maxy = geom.bounds
        win = rasterio.windows.from_bounds(minx - res, miny - res, maxx + res, maxy + res, transform=src.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        transform = src.window_transform(win)
        shape = (int(win.height), int(win.width))
    inside = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)
    ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    # exact cell area (km2) per row on the WGS84 sphere approximation
    R = 6371.0088
    lat_top = np.radians(transform.f + np.arange(shape[0]) * transform.e)
    lat_bot = np.radians(transform.f + (np.arange(shape[0]) + 1) * transform.e)
    row_area = (R ** 2) * np.radians(abs(transform.a)) * np.abs(np.sin(lat_top) - np.sin(lat_bot))
    cell_area = np.repeat(row_area[:, None], shape[1], axis=1)
    return {"transform": transform, "shape": shape, "crs": crs, "inside": inside,
            "lon": xs, "lat": ys, "cell_area": cell_area,
            "bounds": (transform.c, transform.f + shape[0] * transform.e,
                       transform.c + shape[1] * transform.a, transform.f)}


def read_to_grid(path, grid, band=1, resampling=Resampling.nearest):
    """Read one band of any raster onto the master grid (only the needed window is read)."""
    with rasterio.open(path) as src:
        src_crs = src.crs or rasterio.crs.CRS.from_epsg(4326)
        with WarpedVRT(src, src_crs=src_crs, crs=grid["crs"], transform=grid["transform"],
                       width=grid["shape"][1], height=grid["shape"][0], resampling=resampling) as vrt:
            a = vrt.read(band, masked=True).astype("float64").filled(np.nan)
    a[(a < -1e30) | (a == -9999) | (a == -3.4e38)] = np.nan
    return a


def load_climate(source, grid, name):
    """Baseline: folder of 19 files. Future: one 19-band file (WorldClim CMIP6) or a folder."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{name}: not found -> {source}")
    out = {}
    if source.is_dir():
        files = find_bio_files(source)
        for b in BIO:
            out[b] = read_to_grid(files[b], grid)
    else:
        with rasterio.open(source) as src:
            if src.count != 19:
                raise ValueError(f"{name}: {source.name} has {src.count} bands, expected 19")
        for i, b in enumerate(BIO, start=1):
            out[b] = read_to_grid(source, grid, band=i)
    return out


# STEP 4. TERRAIN VARIABLES FROM THE DEM
#   20 aspect (Flat, N, NE, E, SE, S, SW, W, NW), 21 elevation class, 22 slope, 23 TRI

def classify_elevation(elev_m):
    cls = np.full(elev_m.shape, np.nan)
    ok = np.isfinite(elev_m)
    cls[ok] = np.digitize(elev_m[ok], ELEV_BREAKS) + 1          # 1 = below the first break
    return cls


def terrain_from_dem(dem_path, grid, res_m, work_dir):
    """Derive slope, aspect class and TRI at res_m in UTM 44N, block by block (low memory), save them as
    GeoTIFFs, then aggregate to the 30-second grid: mean for elevation/slope/TRI, majority for aspect."""
    from rasterio.windows import Window
    minx, miny, maxx, maxy = grid["bounds"]
    ux0, uy0, ux1, uy1 = transform_bounds("EPSG:4326", UTM_CRS, minx, miny, maxx, maxy, densify_pts=50)
    pad = 3 * res_m
    ux0, uy0, ux1, uy1 = ux0 - pad, uy0 - pad, ux1 + pad, uy1 + pad
    w, h = int(np.ceil((ux1 - ux0) / res_m)), int(np.ceil((uy1 - uy0) / res_m))
    utm_tr = from_origin(ux0, uy1, res_m, res_m)
    dem_path = Path(dem_path)
    tiles = ([dem_path] if dem_path.is_file() else
             sorted(p for p in dem_path.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".hgt", ".img", ".asc")))
    if not tiles:
        raise FileNotFoundError(f"No DEM raster found in {dem_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: work_dir / f"terrain_{res_m}m_{k}.tif" for k in ("elevation_m", "slope", "aspect", "tri")}
    base_prof = {"driver": "GTiff", "height": h, "width": w, "count": 1, "crs": UTM_CRS, "transform": utm_tr,
                 "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"}
    dsts = {k: rasterio.open(p, "w", **base_prof, **({"dtype": "uint8", "nodata": 255} if k == "aspect"
                                                       else {"dtype": "float32", "nodata": -9999}))
            for k, p in paths.items()}
    srcs = [rasterio.open(t) for t in tiles]
    rs = Resampling.bilinear if res_m <= 45 else Resampling.average
    vrts = [WarpedVRT(src, src_crs=src.crs or rasterio.crs.CRS.from_epsg(4326), crs=UTM_CRS, transform=utm_tr,
                      width=w, height=h, resampling=rs) for src in srcs]
    say(f"  deriving terrain at {res_m} m ({h} x {w} cells) from {len(tiles)} DEM file(s) ...")
    B = 512
    try:
        for r0 in range(0, h, B):
            n = min(B, h - r0)
            top, bot = max(r0 - 1, 0), min(r0 + n + 1, h)
            z = np.full((bot - top, w), np.nan, dtype="float32")
            for vrt in vrts:
                zt = vrt.read(1, window=Window(0, top, w, bot - top), masked=True).astype("float32").filled(np.nan)
                z = np.where(np.isnan(z), zt, z)
            z[(z < -500) | (z > 9000)] = np.nan
            zp = np.pad(z, ((1 if top == r0 else 0, 1 if bot == r0 + n else 0), (1, 1)), mode="edge")
            a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
            d, e, f = zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:]
            g, hh, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
            gx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res_m)          # dz/d(east), Horn (1981)
            gy = ((a + 2 * b + c) - (g + 2 * hh + i)) / (8 * res_m)         # dz/d(north)
            slope = np.degrees(np.arctan(np.hypot(gx, gy)))
            bearing = (np.degrees(np.arctan2(-gx, -gy)) + 360) % 360      # downslope direction, clockwise from N
            asp = (np.floor(((bearing + 22.5) % 360) / 45) + 1).astype("float32")   # 1 N, 2 NE ... 8 NW
            asp[slope <= FLAT_SLOPE_DEG + 1e-6] = 0                         # 0 Flat
            tri = np.sqrt(sum((nb - e) ** 2 for nb in (a, b, c, d, f, g, hh, i)))   # Riley et al. (1999)
            bad = np.isnan(e) | np.isnan(slope)
            win = Window(0, r0, w, n)
            dsts["elevation_m"].write(np.where(np.isnan(e), -9999, e).astype("float32"), 1, window=win)
            dsts["slope"].write(np.where(bad, -9999, slope).astype("float32"), 1, window=win)
            dsts["tri"].write(np.where(bad | np.isnan(tri), -9999, tri).astype("float32"), 1, window=win)
            dsts["aspect"].write(np.where(bad, 255, asp).astype("uint8"), 1, window=win)
    finally:
        for v in vrts:
            v.close()
        for src in srcs:
            src.close()
        for dst in dsts.values():
            dst.close()
    out = {"elevation_m": read_to_grid(paths["elevation_m"], grid, resampling=Resampling.average),
           "slope": read_to_grid(paths["slope"], grid, resampling=Resampling.average),
           "tri": read_to_grid(paths["tri"], grid, resampling=Resampling.average),
           "aspect": read_to_grid(paths["aspect"], grid, resampling=Resampling.mode)}
    out["elevation"] = classify_elevation(out["elevation_m"])
    say(f"  {res_m}-m terrain rasters saved in {work_dir}")
    return out



# STEP 5. INPUT CHECKS (coverage, duplicated scenarios, warming signal, occurrences)

def coverage_report(layers, grid, name):
    inside = grid["inside"]
    n = inside.sum()
    bad = {}
    for v, a in layers.items():
        miss = np.sum(inside & ~np.isfinite(a))
        if miss:
            bad[v] = miss
    worst = max(bad.values()) if bad else 0
    say(f"  {name:22s} study-area cells {n:6d}   cells without data (worst layer) {worst:5d} ({100*worst/n:.2f}%)")
    if worst / n > COVERAGE_TOLERANCE:
        raise RuntimeError(
            f"{name} does not cover the study area ({100*worst/n:.1f}% of cells have no data). "
            "If Dolpa was added, the input was probably pre-clipped to the old six districts: "
            "use the original (uncut) WorldClim / DEM file.")
    return bad


def duplicate_check(future, mask):
    rows, dups = [], []
    for a, b in itertools.combinations(future, 2):
        same = tot = 0
        maxd = 0.0
        for v in BIO:
            x, y = future[a][v][mask], future[b][v][mask]
            same += np.sum(np.abs(x - y) < 1e-6); tot += x.size
            maxd = max(maxd, float(np.nanmax(np.abs(x - y))))
        share = same / tot
        rows.append({"scenario_a": a, "scenario_b": b, "identical_share": share, "max_abs_difference": maxd})
        if share > DUPLICATE_SHARE:
            dups.append((a, b))
    return pd.DataFrame(rows), dups


def climate_change_table(base, future, mask, extra_masks=None):
    rows = []
    for s, lay in future.items():
        row = {"scenario": s}
        for v in BIO:
            row[f"d_{v}"] = float(np.nanmean(lay[v][mask]) - np.nanmean(base[v][mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def warming_checks(cc):
    d = dict(zip(cc["scenario"], cc["d_bio1"]))
    msgs = []
    for ssp in ("SSP245", "SSP585"):
        seq = [d.get(f"{ssp}_{p}") for p in ("2021_2040", "2061_2080", "2081_2100")]
        if None not in seq and not (seq[0] < seq[1] < seq[2]):
            msgs.append(f"{ssp}: annual mean temperature does not rise through time (" + ", ".join(f"{v:+.2f}" for v in seq) + " °C)")
    for p in ("2061_2080", "2081_2100"):
        a, b = d.get(f"SSP245_{p}"), d.get(f"SSP585_{p}")
        if a is not None and b is not None and b <= a:
            msgs.append(f"{p}: SSP5-8.5 is not warmer than SSP2-4.5 ({b:.2f} vs {a:.2f} °C)")
    a, b = d.get("SSP245_2021_2040"), d.get("SSP585_2021_2040")
    if a is not None and b is not None and abs(a - b) > 0.6:
        msgs.append(f"2021-2040: SSP2-4.5 and SSP5-8.5 differ by {abs(a-b):.2f} °C (expected < 0.6)")
    return msgs


def load_occurrences(geom, grid, valid):
    df = pd.read_csv(OCCURRENCE_CSV)
    if LON_COL not in df or LAT_COL not in df:
        raise KeyError(f"Columns {LON_COL}/{LAT_COL} not found in {OCCURRENCE_CSV.name}: {list(df.columns)}")
    n0 = len(df)
    df["record_id"] = np.arange(1, n0 + 1)
    df = df.dropna(subset=[LON_COL, LAT_COL]).copy()
    n_dropped_list = 0
    if DROP_RECORDS_CSV is not None:
        drop = pd.read_csv(DROP_RECORDS_CSV)
        key = lambda a, b: pd.Series(np.round(a, 5)).astype(str) + "_" + pd.Series(np.round(b, 5)).astype(str)
        dk = set(key(drop[LON_COL].values, drop[LAT_COL].values))
        hit = key(df[LON_COL].values, df[LAT_COL].values).isin(dk).values
        n_dropped_list = int(hit.sum())
        if n_dropped_list < len(drop):
            say(f"  WARNING: {len(drop) - n_dropped_list} rows of {Path(DROP_RECORDS_CSV).name} matched no record "
                "(coordinates must match to 5 decimals)")
        df = df[~hit].copy()
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]), crs="EPSG:4326")
    inside_poly = pts.within(geom).values
    tr = grid["transform"]
    col = np.floor((df[LON_COL].values - tr.c) / tr.a).astype(int)
    row = np.floor((df[LAT_COL].values - tr.f) / tr.e).astype(int)
    on_grid = (row >= 0) & (row < grid["shape"][0]) & (col >= 0) & (col < grid["shape"][1])
    ok = np.zeros(len(df), bool)
    ok[on_grid] = valid[row[on_grid], col[on_grid]]
    df["row"], df["col"] = row, col
    df["inside_study_area"], df["has_predictors"] = inside_poly, ok
    outside = df[~inside_poly]
    cnt = df[inside_poly & ok].groupby(["row", "col"]).size().rename("n_records_in_cell")
    kept = df[inside_poly & ok].drop_duplicates(subset=["row", "col"]).copy()
    kept = kept.join(cnt, on=["row", "col"])
    report = {"records_in_file": n0, "dropped_by_list": n_dropped_list,
              "missing_coordinates": n0 - n_dropped_list - len(df),
              "outside_study_area": int((~inside_poly).sum()),
              "inside_but_no_predictor_data": int((inside_poly & ~ok).sum()),
              "duplicates_in_same_cell": int((inside_poly & ok).sum() - len(kept)),
              "presences_used": len(kept)}
    return kept, outside, report



# STEP 6. PREDICTOR SCREENING (Pearson with ranked priorities, then VIF)

def pearson_screen(df, threshold, priority):
    corr = df.corr().abs()
    rank = {v: i for i, v in enumerate(priority)}
    pairs = sorted(((corr.loc[a, b], a, b) for a, b in itertools.combinations(df.columns, 2)
                    if corr.loc[a, b] > threshold), reverse=True)
    dropped, log = set(), []
    for r, a, b in pairs:
        if a in dropped or b in dropped:
            continue
        ra, rb = rank.get(a, 999), rank.get(b, 999)
        if ra != rb:
            drop = b if ra < rb else a
            why = "priority"
        else:
            drop = a if corr[a].mean() > corr[b].mean() else b
            why = "higher mean |r|"
        keep = b if drop == a else a
        dropped.add(drop)
        log.append({"dropped": drop, "kept": keep, "abs_r": round(r, 3), "rule": why})
    return [c for c in df.columns if c not in dropped], pd.DataFrame(log)


def vif(df):
    out = {}
    for c in df.columns:
        others = df.drop(columns=c)
        r2 = LinearRegression().fit(others, df[c]).score(others, df[c]) if others.shape[1] else 0.0
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out).sort_values(ascending=False)


def vif_screen(df, threshold, priority):
    rank = {v: i for i, v in enumerate(priority)}
    cols, log = list(df.columns), []
    while len(cols) > 2:
        v = vif(df[cols])
        if v.iloc[0] < threshold:
            break
        over = [c for c in v.index if v[c] >= threshold]
        drop = max(over, key=lambda c: (rank.get(c, 999), v[c]))   # drop least-preferred high-VIF variable
        log.append({"dropped": drop, "vif": round(float(v[drop]), 2)})
        cols.remove(drop)
    return cols, vif(df[cols]), pd.DataFrame(log)


# STEP 7. SPATIAL BLOCKS FOR CROSS-VALIDATION

def to_utm(lon, lat):
    t = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    return t.transform(lon, lat)


def make_spatial_folds(x_m, y_m, y, k, block_km, seed):
    size = block_km * 1000
    bx = np.floor((x_m - x_m.min()) / size).astype(int)
    by = np.floor((y_m - y_m.min()) / size).astype(int)
    block = bx * 10_000 + by
    ids = pd.DataFrame({"block": block, "y": y}).groupby("block")["y"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    ids = ids.sample(frac=1, random_state=seed).sort_values("sum", ascending=False, kind="stable")
    pres = np.zeros(k); tot = np.zeros(k); assign = {}
    for blk, r in ids.iterrows():
        if r["sum"] > 0:
            f = int(np.lexsort((tot, pres))[0])
        else:
            f = int(np.argmin(tot))
        assign[blk] = f; pres[f] += r["sum"]; tot[f] += r["count"]
    folds = np.array([assign[b] for b in block])
    return folds, pd.DataFrame({"fold": range(k), "presences": pres.astype(int), "points": tot.astype(int)})



# STEP 8. THE THREE ALGORITHMS (uniform fit / predict interface)

class Member:
    """One algorithm with a common fit/predict interface. Aspect and elevation classes are one-hot
    encoded (dummy variables); continuous predictors are clamped to the training range, as Maxent does."""

    def __init__(self, algo, params, seed=RANDOM_STATE):
        self.algo, self.params, self.seed = algo, dict(params), seed

    def _dummies(self, X):
        cols = {}
        for c in self.cat:
            v = X[c].values
            for k in CATEGORICAL[c]:
                cols[f"{c}_{k}"] = (v == k).astype(float)
        return pd.DataFrame(cols, index=X.index)

    def _design(self, X, fitting=False):
        Xn = X[self.num].astype(float)
        if not fitting:
            Xn = Xn.clip(self.lo, self.hi, axis=1)
        D = self._dummies(X)
        if fitting:
            self.dcols = [c for c in D.columns if D[c].nunique() > 1]   # categories seen in training
        return Xn, D[self.dcols]

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.cat = [c for c in self.cols if c in CATEGORICAL]
        self.num = [c for c in self.cols if c not in CATEGORICAL]
        self.lo, self.hi = X[self.num].min(), X[self.num].max()
        Xn, D = self._design(X, fitting=True)
        y = np.asarray(y)
        p = self.params
        if self.algo == "maxent":
            self.m = ela.MaxentModel(feature_types=p["features"], beta_multiplier=p["beta"],
                                     transform="cloglog", clamp=True, random_state=self.seed)
            self.m.fit(pd.concat([Xn, D], axis=1), y)
        elif self.algo == "glm":
            self.mu, self.sd = Xn.mean(), Xn.std().replace(0, 1)
            self.m = LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000).fit(self._glm(Xn, D), y)
        elif self.algo == "rf":
            self.m = RandomForestClassifier(n_estimators=RF_TREES, max_depth=p["max_depth"],
                                            min_samples_leaf=p["min_leaf"], max_features="sqrt",
                                            class_weight="balanced", n_jobs=-1,
                                            random_state=self.seed).fit(np.hstack([Xn.values, D.values]), y)
        return self

    def _glm(self, Xn, D):
        Z = ((Xn - self.mu) / self.sd).values
        if self.params["quadratic"]:
            Z = np.hstack([Z, Z ** 2])
        return np.hstack([Z, D.values])

    def predict(self, X):
        Xn, D = self._design(X[self.cols])
        if self.algo == "maxent":
            return np.asarray(self.m.predict(pd.concat([Xn, D], axis=1)), dtype=float).ravel()
        if self.algo == "glm":
            return self.m.predict_proba(self._glm(Xn, D))[:, 1]
        return self.m.predict_proba(np.hstack([Xn.values, D.values]))[:, 1]


def grid_for(algo):
    if algo == "maxent":
        return [{"features": f, "beta": b} for f in MAXENT_FEATURES for b in MAXENT_BETAS]
    if algo == "glm":
        return [{"C": c, "quadratic": q} for q in GLM_QUADRATIC for c in GLM_C]
    return [{"max_depth": d, "min_leaf": l} for d in RF_MAX_DEPTH for l in RF_MIN_LEAF]


def params_str(p):
    return ", ".join(f"{k}={'+'.join(v) if isinstance(v, list) else v}" for k, v in p.items())


ALGOS = ["maxent", "glm", "rf"]



# STEP 9. EVALUATION HELPERS (AUC, thresholds, continuous Boyce, Moran's I)

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def cv_score(algo, params, X, y, folds, use_folds):
    aucs = []
    for f in use_folds:
        tr = np.isin(folds, use_folds) & (folds != f)
        te = folds == f
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = Member(algo, params).fit(X[tr], y[tr])
        aucs.append(safe_auc(y[te], m.predict(X[te])))
    return float(np.mean(aucs)) if aucs else np.nan, float(np.std(aucs)) if aucs else np.nan, aucs


def tune(algo, X, y, folds, use_folds):
    rows = []
    for p in grid_for(algo):
        m, s, a = cv_score(algo, p, X, y, folds, use_folds)
        rows.append({"algorithm": algo, "params": params_str(p), "cv_auc": m, "cv_sd": s,
                     "fold_aucs": ";".join(f"{v:.3f}" for v in a), "_p": p})
    tab = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    return tab.loc[0, "_p"], tab


def ensemble_weights(cv_aucs):
    keep = {a: v for a, v in cv_aucs.items() if np.isfinite(v) and v >= MIN_ENSEMBLE_AUC}
    if not keep:
        best = max(cv_aucs, key=lambda a: cv_aucs[a])
        keep = {best: cv_aucs[best]}
    s = sum(keep.values())
    return {a: v / s for a, v in keep.items()}


def ensemble_predict(members, weights, X):
    return sum(w * members[a].predict(X) for a, w in weights.items())


def thresholds_from(p_pres, p_bg):
    cand = np.linspace(0.001, 0.999, 999)
    sens = np.array([(p_pres >= t).mean() for t in cand])
    spec = np.array([(p_bg < t).mean() for t in cand])
    th = {"MTSS": float(cand[np.argmax(sens + spec)]),
          "ETS": float(cand[np.argmin(np.abs(sens - spec))]),
          "P10": float(np.percentile(p_pres, 10)),
          "P5": float(np.percentile(p_pres, 5))}
    rows = []
    for k, t in th.items():
        se, sp = (p_pres >= t).mean(), (p_bg < t).mean()
        rows.append({"threshold": k, "cutoff": t, "sensitivity": se, "specificity": sp, "TSS": se + sp - 1})
    return th, pd.DataFrame(rows)


def boyce_continuous(p_pres, p_bg, n_points=100, window=0.1):
    """Continuous Boyce index (Hirzel et al. 2006): moving window of width 1/10 of the range."""
    lo, hi = min(p_bg.min(), p_pres.min()), max(p_bg.max(), p_pres.max())
    w = window * (hi - lo)
    mids = np.linspace(lo + w / 2, hi - w / 2, n_points)
    ratios = []
    for m in mids:
        pi = np.mean((p_pres >= m - w / 2) & (p_pres <= m + w / 2))
        ei = np.mean((p_bg >= m - w / 2) & (p_bg <= m + w / 2))
        ratios.append(pi / ei if ei > 0 else np.nan)
    ratios = np.array(ratios)
    ok = np.isfinite(ratios)
    rho = spearmanr(mids[ok], ratios[ok])[0] if ok.sum() > 3 else np.nan
    return float(rho), mids, ratios


def morans_i(resid, x, y, k=8, n_perm=199, seed=RANDOM_STATE):
    xy = np.column_stack([x, y])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    z = resid - resid.mean()
    def stat(zz):
        return (len(zz) / (len(zz) * k)) * np.sum(zz * zz[nn].sum(1)) / np.sum(zz ** 2)
    obs = stat(z)
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    return float(obs), float(p)



# STEP 10. PROJECTION, NOVELTY (MESS + most dissimilar variable) AND RANGE CHANGE

def stack_frame(layers, variables, mask):
    return pd.DataFrame({v: layers[v][mask] for v in variables})


def predict_map(members, weights, layers, variables, mask, chunk=200_000):
    X = stack_frame(layers, variables, mask)
    out = {a: np.full(mask.shape, np.nan) for a in weights}
    ens = np.full(mask.shape, np.nan)
    vals = {a: np.empty(len(X)) for a in weights}
    for s in range(0, len(X), chunk):
        xs = X.iloc[s:s + chunk]
        for a in weights:
            vals[a][s:s + chunk] = members[a].predict(xs)
    e = np.zeros(len(X))
    for a, w in weights.items():
        out[a][mask] = vals[a]; e += w * vals[a]
    ens[mask] = e
    return ens, out


def mess(layers, ref, variables, mask):
    sims = []
    for v in variables:
        r = np.sort(ref[v].values)
        rmin, rmax = r[0], r[-1]
        p = layers[v][mask]
        f = 100 * np.searchsorted(r, p, side="left") / len(r)
        rng_ = (rmax - rmin) if rmax > rmin else 1.0
        s = np.where(f == 0, 100 * (p - rmin) / rng_,
            np.where(f <= 50, 2 * f, np.where(f < 100, 2 * (100 - f), 100 * (rmax - p) / rng_)))
        sims.append(s)
    sims = np.vstack(sims)
    m = np.full(mask.shape, np.nan); mod = np.full(mask.shape, -1, dtype=int)
    m[mask] = sims.min(0); mod[mask] = sims.argmin(0)
    return m, mod


def bearing_label(deg):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int(((deg % 360) + 11.25) // 22.5) % 16]


def range_change(base_suit, base_known, fut_suit, fut_known, grid, ux, uy, elev):
    A = grid["cell_area"]
    stable = base_suit & fut_suit
    lost = base_suit & ~fut_suit
    lost_novel = base_suit & ~fut_known
    lost_suit = lost & fut_known
    gained = fut_suit & ~base_suit
    gained_base_unknown = gained & ~base_known
    area = lambda m: float(A[m].sum())
    b, f = area(base_suit), area(fut_suit)
    res = {"baseline_km2": b, "future_km2": f, "stable_km2": area(stable), "lost_km2": area(lost),
           "lost_novel_km2": area(lost_novel), "lost_suitability_km2": area(lost_suit),
           "gained_km2": area(gained), "gained_where_baseline_novel_km2": area(gained_base_unknown),
           "change_pct": 100 * (f - b) / b if b else np.nan,
           "turnover_pct": 100 * (area(lost) + area(gained)) / (b + area(gained)) if b else np.nan,
           "share_of_loss_from_novelty_pct": 100 * area(lost_novel) / area(lost) if area(lost) else np.nan}
    if base_suit.any() and fut_suit.any():
        bx, by = np.average(ux[base_suit], weights=A[base_suit]), np.average(uy[base_suit], weights=A[base_suit])
        fx, fy = np.average(ux[fut_suit], weights=A[fut_suit]), np.average(uy[fut_suit], weights=A[fut_suit])
        dist = np.hypot(fx - bx, fy - by) / 1000
        brg = (np.degrees(np.arctan2(fx - bx, fy - by)) + 360) % 360
        res.update({"centroid_shift_km": dist, "centroid_bearing_deg": brg, "centroid_direction": bearing_label(brg)})
    if elev is not None and base_suit.any() and fut_suit.any():
        res.update({"median_elev_baseline_m": float(np.nanmedian(elev[base_suit])),
                    "median_elev_future_m": float(np.nanmedian(elev[fut_suit]))})
        res["median_elev_shift_m"] = res["median_elev_future_m"] - res["median_elev_baseline_m"]
    return res


def save_raster(path, arr, grid, dtype="float32", nodata=-9999):
    a = np.where(np.isfinite(arr), arr, nodata).astype(dtype)
    prof = {"driver": "GTiff", "height": grid["shape"][0], "width": grid["shape"][1], "count": 1,
            "dtype": dtype, "crs": grid["crs"], "transform": grid["transform"], "nodata": nodata,
            "compress": "deflate"}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(a, 1)


# STEP 10c. RECORD CHECKS AND DISTRICTS
=
def load_districts():
    """District polygons (WGS84) with a 'district' name column, or None if the layer is not available."""
    if not DRAW_DISTRICTS:
        return None
    try:
        p = resolve_shp(DISTRICTS_SHP)
        if not Path(p).is_file():
            return None
        d = gpd.read_file(p)
        d = d.set_crs("EPSG:4326") if d.crs is None else d.to_crs("EPSG:4326")
        text = [c for c in d.columns if c != "geometry" and
                (pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]))]
        wanted = {n.strip().lower() for n in ADD_DISTRICTS}
        col = next((c for c in text if d[c].astype(str).str.strip().str.lower().isin(wanted).any()), None)
        col = col or (text[0] if text else None)
        d["district"] = d[col].astype(str).str.strip().str.title() if col else ""
        return d[["district", "geometry"]]
    except Exception as e:                      # maps are drawn without districts if the layer cannot be read
        say(f"  (district layer not used: {e})")
        return None


def mess_points(Xq, Xref, variables):
    """MESS (Elith et al. 2010) of query points relative to reference points, and the most dissimilar variable."""
    sims = []
    for v in variables:
        r = np.sort(Xref[v].values)
        rmin, rmax = r[0], r[-1]
        p = Xq[v].values
        f = 100 * np.searchsorted(r, p, side="left") / len(r)
        rng_ = (rmax - rmin) if rmax > rmin else 1.0
        sims.append(np.where(f == 0, 100 * (p - rmin) / rng_,
                    np.where(f <= 50, 2 * f, np.where(f < 100, 2 * (100 - f), 100 * (rmax - p) / rng_))))
    sims = np.vstack(sims)
    return sims.min(0), np.array(variables)[sims.argmin(0)]


def records_audit(pres, pf, oof_pres, Xp, cont, elev_cell, districts, dem30_path):
    """One row per presence cell with the information needed to check doubtful records."""
    a = pres[["record_id", LON_COL, LAT_COL, "n_records_in_cell"]].copy().reset_index(drop=True)
    if districts is not None:
        pts = gpd.GeoDataFrame(a, geometry=gpd.points_from_xy(a[LON_COL], a[LAT_COL]), crs="EPSG:4326")
        j = gpd.sjoin(pts, districts, how="left", predicate="within")
        a["district"] = j.groupby(level=0)["district"].first().reindex(a.index).values
    a["cell_mean_elevation_m"] = np.round(elev_cell, 0)
    try:
        with rasterio.open(dem30_path) as src:
            xs, ys = to_utm(a[LON_COL].values, a[LAT_COL].values)
            v = np.array([z[0] for z in src.sample(zip(xs, ys))], dtype=float)
            v[v == src.nodata] = np.nan
            a["point_elevation_m_30m_dem"] = np.round(v, 0)
    except Exception:
        a["point_elevation_m_30m_dem"] = np.nan
    a["fold"] = pf
    a["oof_suitability"] = np.round(oof_pres, 4)
    m = np.full(len(a), np.nan); mod = np.full(len(a), "", dtype=object)
    for f in np.unique(pf):
        q, r = Xp.loc[pf == f, cont], Xp.loc[pf != f, cont]
        m[pf == f], mod[pf == f] = mess_points(q, r, cont)
    a["mess_vs_other_folds"] = np.round(m, 1)
    a["most_dissimilar_vs_other_folds"] = mod
    for v in cont:
        a[v] = np.round(Xp[v].values, 2)
    flags = []
    for _, r in a.iterrows():
        f = []
        if r["cell_mean_elevation_m"] < LOW_ELEVATION_FLAG_M:
            f.append(f"cell below {LOW_ELEVATION_FLAG_M} m")
        if r["oof_suitability"] < UNPREDICTED_FLAG:
            f.append("not predicted when its fold was withheld")
        if r["mess_vs_other_folds"] < 0:
            f.append(f"climate outside other folds' presences ({r['most_dissimilar_vs_other_folds']})")
        flags.append("; ".join(f))
    a["check"] = flags
    a["n_flags"] = [0 if not f else f.count(";") + 1 for f in flags]
    return a.sort_values(["n_flags", "oof_suitability"], ascending=[False, True])


def overlay(ax, grid, geom, districts, labels=False, lw=0.35):
    if districts is not None:
        x0, x1, y0, y1 = extent(grid)
        sub = districts.cx[x0:x1, y0:y1]
        sub.boundary.plot(ax=ax, color="0.25", lw=lw)
        if labels:
            for _, r in sub.iterrows():
                part = r.geometry.intersection(geom)
                if part.is_empty or part.area < 0.02:
                    continue
                pt = part.representative_point()
                ax.text(pt.x, pt.y, r["district"].upper(), fontsize=7, fontweight="bold", ha="center", va="center",
                        color="0.1", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(ax=ax, color="k", lw=lw * 2.2)
    x0, x1, y0, y1 = extent(grid)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel(""); ax.set_ylabel("")


def fig_study_area_v3(elev, grid, pres, geom, districts, path):
    """Fig. 1: elevation with hillshade, district boundaries and names, occurrences, scale bar, north arrow, inset."""
    from matplotlib.colors import LightSource, Normalize
    import matplotlib.cm as mcm
    inside = grid["inside"]
    z = np.where(np.isfinite(elev), elev, np.nanmin(elev))
    norm = Normalize(vmin=np.nanpercentile(elev[inside], 0.5), vmax=np.nanpercentile(elev[inside], 99.5))
    cmap = plt.get_cmap("terrain")
    ls = LightSource(azdeg=315, altdeg=40)
    dx = abs(grid["transform"].a) * 111320 * np.cos(np.radians(np.mean(grid["lat"])))
    rgb = ls.shade(z, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=1.5, dx=dx, dy=abs(grid["transform"].e) * 110570)
    rgb[~inside, :3] = 1 - 0.35 * (1 - rgb[~inside, :3])          # fade outside the study area
    rgb[~np.isfinite(elev), :3] = 1.0                               # no DEM (outside Nepal): white
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.imshow(rgb, extent=extent(grid), interpolation="nearest")
    overlay(ax, grid, geom, districts, labels=True, lw=0.45)
    ax.scatter(pres[LON_COL], pres[LAT_COL], s=14, c="#d7301f", edgecolor="k", lw=0.4, zorder=5,
               label=f"Occurrences (n = {len(pres)} grid cells)")
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02, label="Elevation (m a.s.l.)")
    x0, x1, y0, y1 = extent(grid)
    km_per_deg = 111.32 * np.cos(np.radians((y0 + y1) / 2))
    L = 50 / km_per_deg
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.plot([bx, bx + L], [by, by], color="k", lw=3, solid_capstyle="butt")
    ax.plot([bx, bx + L / 2], [by, by], color="w", lw=1.4, solid_capstyle="butt")
    ax.text(bx + L / 2, by + 0.03 * (y1 - y0), "50 km", ha="center", fontsize=8)
    nr, nc = inside.shape
    share = {"left": inside[: int(0.32 * nr), : int(0.32 * nc)].mean(), "right": inside[: int(0.32 * nr), -int(0.32 * nc):].mean()}
    inset_side = min(share, key=share.get)
    ax_x = 0.05 if inset_side == "right" else 0.95
    ax.annotate("N", xy=(ax_x, 0.93), xytext=(ax_x, 0.80), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    if districts is not None and len(districts) > 2:
        ins = ax.inset_axes([0.67, 0.68, 0.32, 0.31] if inset_side == "right" else [0.01, 0.68, 0.32, 0.31])
        nep = districts.dissolve()
        nep.plot(ax=ins, color="#f0f0f0", edgecolor="0.4", lw=0.4)
        districts.boundary.plot(ax=ins, color="0.7", lw=0.15)
        gpd.GeoSeries([geom], crs="EPSG:4326").plot(ax=ins, color="#d7301f", alpha=0.6)
        ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white"); ins.set_xlabel(""); ins.set_ylabel("")
        ins.text(0.03, 0.95, "Nepal", transform=ins.transAxes, fontsize=7, va="top")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_records_check(audit, grid, geom, districts, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    overlay(ax, grid, geom, districts, labels=True)
    # drawn from bottom to top; the "outside" ring is hollow so the symbols beneath stay visible
    cats = [("No flag", audit["n_flags"] == 0, dict(s=18, c="#4daf4a", marker="o", edgecolor="k", lw=0.3)),
            ("Climate outside other folds' presences", audit["check"].str.contains("outside"),
             dict(s=70, facecolors="none", edgecolors="#ff7f00", marker="o", lw=1.1)),
            ("Not predicted when withheld", audit["check"].str.contains("not predicted"),
             dict(s=20, c="#e41a1c", marker="s", edgecolor="k", lw=0.3)),
            (f"Cell below {LOW_ELEVATION_FLAG_M} m", audit["check"].str.contains("below"),
             dict(s=46, c="#7b3294", marker="^", edgecolor="w", lw=0.6))]
    for z, (lab, m, kw) in enumerate(cats):
        if m.any():
            ax.scatter(audit.loc[m, LON_COL], audit.loc[m, LAT_COL], label=f"{lab} (n = {int(m.sum())})",
                       zorder=5 + z, **kw)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Presence records to check (see records_to_check.csv)", fontsize=9)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_flagged_climate(audit, bg_vals, cont, path):
    """Elevation and predictor values of presences not predicted when withheld vs other presences and background."""
    flag = audit["check"].str.contains("not predicted")
    if not flag.any():
        return
    cols = ["cell_mean_elevation_m"] + list(cont)
    labels = {"cell_mean_elevation_m": "Elevation (m)", **{v: VAR_LABEL.get(v, v) for v in cont}}
    n = len(cols)
    ncol = 4
    fig, axs = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(3.0 * ncol, 2.6 * np.ceil(n / ncol)), squeeze=False)
    groups = [(f"Not predicted\nwhen withheld\n(n = {int(flag.sum())})", audit.loc[flag], "#e41a1c"),
              (f"Other presences\n(n = {int((~flag).sum())})", audit.loc[~flag], "#4daf4a"),
              ("Background", bg_vals, "#bdbdbd")]
    for ax, c in zip(axs.flat, cols):
        data = [g[c].dropna().values for _, g, _ in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, (_, _, col) in zip(bp["boxes"], groups):
            patch.set_facecolor(col); patch.set_alpha(0.8)
        for med in bp["medians"]:
            med.set_color("k")
        for k, d in enumerate(data[:2], start=1):
            ax.scatter(np.random.default_rng(1).normal(k, 0.06, len(d)), d, s=5, c="k", alpha=0.5, zorder=3)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
        ax.set_title(labels[c], fontsize=8, loc="left"); ax.tick_params(axis="y", labelsize=7)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)



# STEP 11. FIGURES

def extent(grid):
    x0, y0, x1, y1 = grid["bounds"]
    return [x0, x1, y0, y1]


def panel_axes(n, ncol=3, w=4.2, h=3.0):
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow + 0.6), squeeze=False)
    axs = list(axs.flat)
    for ax in axs[n:]:
        ax.axis("off")
    return fig, axs


def scenario_axes(keys, w=3.7, h=2.35):
    """Baseline top-left, SSP2-4.5 in the top row and SSP5-8.5 in the bottom row (periods in columns).
    Returns the figure, the axes in key order and a spare axis (bottom-left) for a colour bar or legend."""
    periods = ["2021_2040", "2061_2080", "2081_2100"]
    has_base = keys[0] == "Baseline_1970_2000"
    fut = keys[1:] if has_base else keys
    if not all(k[:6] in ("SSP245", "SSP585") and k[-9:] in periods for k in fut):
        fig, axs = panel_axes(len(keys))
        return fig, axs[:len(keys)], None
    off = 1 if has_base else 0
    fig, grid_ax = plt.subplots(2, 3 + off, figsize=(w * (3 + off), h * 2 + 0.4), squeeze=False)
    amap = {}
    if has_base:
        amap[keys[0]] = grid_ax[0, 0]
    for k in fut:
        amap[k] = grid_ax[0 if k.startswith("SSP245") else 1, off + periods.index(k[-9:])]
    used = [amap[k] for k in keys]
    spare = grid_ax[1, 0] if has_base else None
    for ax in grid_ax.flat:
        if all(ax is not u for u in used):
            ax.axis("off")
    return fig, used, spare


def fig_continuous(maps, novel, grid, path, geom=None, districts=None):
    keys = list(maps)
    fig, axs, spare = scenario_axes(keys)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        im = ax.imshow(maps[k], extent=extent(grid), vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        if novel.get(k) is not None:
            ov = np.where(novel[k], 1.0, np.nan)
            ax.imshow(ov, extent=extent(grid), cmap=ListedColormap(["#bdbdbd"]), alpha=0.75, interpolation="nearest")
        if geom is not None:
            overlay(ax, grid, geom, districts)
        ax.set_title(f"({chr(97+i)}) {SCENARIO_LABEL[k]}", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    if spare is not None:
        cax = spare.inset_axes([0.08, 0.68, 0.84, 0.09])
        fig.colorbar(im, cax=cax, orientation="horizontal", label="Ensemble suitability")
        spare.text(0.08, 0.22, f"Grey: climate novel (MESS < {MESS_THRESHOLD:g})", transform=spare.transAxes, fontsize=8)
    else:
        fig.colorbar(im, ax=axs, shrink=0.6, label="Ensemble suitability (grey: MESS < threshold)")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_range_change(cats, grid, path, geom=None, districts=None):
    cols = ["#e3e3e3", "#2c7bb6", "#d7191c", "#fdae61", "#1a9641"]
    cmap = ListedColormap(cols)
    keys = list(cats)
    fig, axs, _ = scenario_axes(keys)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        ax.imshow(cats[k], extent=extent(grid), cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
        if geom is not None:
            overlay(ax, grid, geom, districts)
        ax.set_title(f"({chr(97+i)}) {SCENARIO_LABEL[k]}", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    lab = ["Unsuitable in both periods", "Stable", "Loss (suitability below threshold)",
           "Loss (future climate novel)", "Gain"]
    fig.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(cols, lab)] +
               [mpatches.Patch(facecolor="white", edgecolor="0.5", label="Novel climate or outside study area")],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_mess(mess_maps, mod_maps, variables, grid, path_mess, path_mod, geom=None, districts=None):
    keys = list(mess_maps)
    fig, axs, spare = scenario_axes(keys)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        im = ax.imshow(np.clip(mess_maps[k], -100, 100), extent=extent(grid), cmap="RdYlGn", vmin=-100, vmax=100,
                       interpolation="nearest")
        if geom is not None:
            overlay(ax, grid, geom, districts)
        ax.set_title(f"({chr(97+i)}) {SCENARIO_LABEL[k]}", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    if spare is not None:
        cax = spare.inset_axes([0.08, 0.55, 0.84, 0.09])
        fig.colorbar(im, cax=cax, orientation="horizontal", label="MESS (negative = novel)")
    else:
        fig.colorbar(im, ax=axs, shrink=0.6, label="MESS (negative = novel)")
    fig.tight_layout(); fig.savefig(path_mess, dpi=300, bbox_inches="tight"); plt.close(fig)
    palette = plt.get_cmap("tab10").colors
    cmap = ListedColormap(palette[:len(variables)])
    fig, axs, spare = scenario_axes(keys)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        show = np.where((mod_maps[k] >= 0) & (mess_maps[k] < 0), mod_maps[k], np.nan).astype(float)
        ax.imshow(show, extent=extent(grid), cmap=cmap, vmin=-0.5, vmax=len(variables) - 0.5, interpolation="nearest")
        if geom is not None:
            overlay(ax, grid, geom, districts)
        ax.set_title(f"({chr(97+i)}) {SCENARIO_LABEL[k]}", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    handles = [mpatches.Patch(color=palette[j], label=VAR_LABEL.get(v, v)) for j, v in enumerate(variables)]
    if spare is not None:
        spare.legend(handles=handles, loc="center", frameon=False, fontsize=7.5, title="Most dissimilar variable",
                     title_fontsize=8)
        fig.tight_layout()
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.subplots_adjust(bottom=0.16)
    fig.savefig(path_mod, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_response(members, weights, Xbg, variables, path, Xpres=None):
    """Marginal responses: other continuous predictors at their background median, other categorical
    predictors at the class most frequent among presences (e.g. the elevation band where the species occurs)."""
    n = len(variables)
    fig, axs = panel_axes(n, ncol=3, w=3.6, h=2.8)
    ref_cat = Xpres if Xpres is not None else Xbg
    med = {c: (ref_cat[c].mode().iloc[0] if c in CATEGORICAL else Xbg[c].median()) for c in variables}
    colors = {"maxent": "#1b9e77", "glm": "#7570b3", "rf": "#d95f02"}
    for i, (ax, v) in enumerate(zip(axs, variables)):
        if v in CATEGORICAL:
            codes = [k for k in CATEGORICAL[v] if (Xbg[v] == k).any()]
            grid_v = np.array(codes, dtype=float)
        else:
            grid_v = np.linspace(Xbg[v].quantile(0.01), Xbg[v].quantile(0.99), 100)
        Xg = pd.DataFrame({c: np.repeat(med[c], len(grid_v)) for c in variables}); Xg[v] = grid_v
        xs = np.arange(len(grid_v)) if v in CATEGORICAL else grid_v
        style = dict(marker="o", ls="none") if v in CATEGORICAL else {}
        for a in weights:
            ax.plot(xs, members[a].predict(Xg), color=colors[a], lw=0.9, alpha=0.7, label=a.upper(), **style)
        ax.plot(xs, ensemble_predict(members, weights, Xg), color="k", lw=2, label="Ensemble",
                **({"marker": "s", "ls": "none", "ms": 7} if v in CATEGORICAL else {}))
        if v in CATEGORICAL:
            ax.set_xticks(xs); ax.set_xticklabels([CATEGORICAL[v][int(k)] for k in grid_v], rotation=45, fontsize=7)
        ax.set_ylim(0, 1); ax.set_xlabel(f"{VAR_NUMBER.get(v, '')} {VAR_LABEL.get(v, v)}", fontsize=8)
        ax.set_title(f"({chr(97+i)})", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
        if i % 3 == 0:
            ax.set_ylabel("Suitability", fontsize=8)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_importance(imp, path):
    imp = imp.sort_values("mean_auc_drop")
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(imp) + 1))
    ax.barh([VAR_LABEL.get(v, v) for v in imp["variable"]], imp["mean_auc_drop"], xerr=imp["sd_across_folds"],
            color="#5b8db8", edgecolor="k", lw=0.5, capsize=3)
    ax.set_xlabel("Mean decrease in AUC on outer folds (ensemble)")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_folds(lon, lat, folds, y, geom, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(ax=ax, color="k", lw=0.7)
    sc = ax.scatter(lon[y == 0], lat[y == 0], c=folds[y == 0], cmap="Set2", s=1, alpha=0.4)
    ax.scatter(lon[y == 1], lat[y == 1], c=folds[y == 1], cmap="Set2", s=18, edgecolor="k", lw=0.5)
    ax.set_title("Spatial cross-validation folds (large points: presences)", fontsize=9)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_study_area(elev, grid, pres, geom, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(elev, extent=extent(grid), cmap="terrain", interpolation="nearest")
    gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(ax=ax, color="k", lw=0.8)
    ax.scatter(pres[LON_COL], pres[LAT_COL], s=12, c="#e41a1c", edgecolor="k", lw=0.4, label="Occurrences")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Elevation (m)")
    ax.legend(loc="lower left", fontsize=8); ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_roc(y, p, path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="navy", lw=2, label=f"Out-of-fold ensemble (AUC = {safe_auc(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_corr(df, path, title):
    c = df.corr()
    fig, ax = plt.subplots(figsize=(0.45 * len(c) + 2, 0.45 * len(c) + 1.5))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns, fontsize=7)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)



# STEP 12. MAIN WORKFLOW

def main():
    global N_BOOTSTRAP, MAXENT_BETAS, RF_MAX_DEPTH, RF_MIN_LEAF, GLM_C, RF_TREES, MAXENT_FEATURES, OUTPUT_DIR, DROP_RECORDS_CSV
    args = parse_args()
    if args.out:
        OUTPUT_DIR = Path(args.out)
    if args.drop:
        DROP_RECORDS_CSV = Path(args.drop)
        if not DROP_RECORDS_CSV.is_file():
            raise SystemExit(f"STOP: --drop file not found: {DROP_RECORDS_CSV}")
    if args.bootstrap is not None:
        N_BOOTSTRAP = args.bootstrap
    if args.quick:
        N_BOOTSTRAP = min(N_BOOTSTRAP, 5)
        MAXENT_BETAS, MAXENT_FEATURES = [1.0, 3.0], [["linear"], ["linear", "quadratic"]]
        GLM_C, RF_MAX_DEPTH, RF_MIN_LEAF, RF_TREES = [0.1, 1.0], [10], [5, 20], 200
    dirs = setup_output()
    T0 = time.time()
    rng = np.random.default_rng(RANDOM_STATE)
    say(f"\nRun started {time.strftime('%Y-%m-%d %H:%M')}  ({'CHECK ONLY' if args.check else 'full run'})")
    versions = write_versions(OUTPUT_DIR / "software_versions.txt")

    #  12.1 Study area and master grid 
    section("STEP 1. Checking input paths")
    check_paths()
    section("STEP 3. Study area and master grid")
    gdf, geom = load_study_area()
    districts = load_districts()
    base_files = find_bio_files(BASELINE_CLIMATE)
    grid = build_master_grid(geom, base_files)
    say(f"  grid {grid['shape'][0]} x {grid['shape'][1]} cells, bounds {np.round(grid['bounds'], 4)}")
    say(f"  study-area cells {grid['inside'].sum()}, area {grid['cell_area'][grid['inside']].sum():,.0f} km2")

    #  12.2 Climate and terrain 
    section("STEP 3-4. Loading climate and terrain onto the master grid")
    base = load_climate(BASELINE_CLIMATE, grid, "Baseline_1970_2000")
    future = {s: load_climate(p, grid, s) for s, p in FUTURE_CLIMATE.items()}
    terrain = terrain_from_dem(DEM_PATH, grid, TERRAIN_RES_M, OUTPUT_DIR / "rasters" / f"terrain_{TERRAIN_RES_M}m")

    section("STEP 5. Input checks")
    say("Coverage of the study area:")
    coverage_report(base, grid, "Baseline_1970_2000")
    for s in future:
        coverage_report(future[s], grid, s)
    coverage_report(terrain, grid, "DEM-derived terrain")
    valid = grid["inside"].copy()
    for lay in [base, terrain, *future.values()]:
        for a in lay.values():
            valid &= np.isfinite(a)
    say(f"  cells with complete data in every layer: {valid.sum()} "
        f"({grid['cell_area'][valid].sum():,.0f} km2)")

    dup_tab, dups = duplicate_check(future, valid)
    dup_tab.to_csv(dirs["tables"] / "check_duplicate_scenarios.csv", index=False)
    say("\nDuplicate check of future scenarios (share of identical cells, 19 variables):")
    for _, r in dup_tab.iterrows():
        flag = "   <-- DUPLICATE" if r["identical_share"] > DUPLICATE_SHARE else ""
        say(f"  {r['scenario_a']:17s} vs {r['scenario_b']:17s} {100*r['identical_share']:6.2f}%{flag}")

    cc = climate_change_table(base, future, valid)
    cc.to_csv(dirs["tables"] / "climate_change_study_area.csv", index=False)
    say("\nMean change from baseline over the study area:")
    show = cc[["scenario", "d_bio1", "d_bio10", "d_bio2", "d_bio12", "d_bio19"]].round(2)
    say(show.to_string(index=False))
    warn = warming_checks(cc)
    for w in warn:
        say(f"  WARNING: {w}")

    say("\nOccurrences:")
    pres, outside, occ_rep = load_occurrences(geom, grid, valid)
    for k, v in occ_rep.items():
        say(f"  {k:32s} {v}")
    if len(outside):
        outside.to_csv(dirs["tables"] / "check_occurrences_outside_study_area.csv", index=False)
        say(f"  WARNING: {len(outside)} records lie outside the study-area polygon (saved to "
            "check_occurrences_outside_study_area.csv). Did you add Dolpa to the shapefile?")
    say(f"  occurrence extent: lon {pres[LON_COL].min():.3f}-{pres[LON_COL].max():.3f}, "
        f"lat {pres[LAT_COL].min():.3f}-{pres[LAT_COL].max():.3f}")
    elev_pres = terrain["elevation_m"][pres["row"].values, pres["col"].values]
    pres.assign(cell_mean_elevation_m=np.round(elev_pres, 0)).drop(columns=["row", "col"]).to_csv(
        dirs["tables"] / "occurrences_used.csv", index=False)
    n_low = int(np.sum(elev_pres < LOW_ELEVATION_FLAG_M))
    if n_low:
        say(f"  NOTE: {n_low} presence cells lie below {LOW_ELEVATION_FLAG_M} m (listed in records_to_check.csv)")
    say(f"  elevation of occurrences: median {np.nanmedian(elev_pres):.0f} m, range "
        f"{np.nanmin(elev_pres):.0f}-{np.nanmax(elev_pres):.0f} m")

    if dups:
        raise SystemExit("\nSTOP: duplicated future scenarios " + ", ".join(f"{a} = {b}" for a, b in dups) +
                         ". Re-download the correct WorldClim file and run again.")
    if args.check:
        say(f"\nInput checks finished in {(time.time()-T0)/60:.1f} min. No duplicated scenarios found."
            + (" See warnings above." if warn or len(outside) else ""))
        return

    #  12.3 Background points and predictor table 
    section("STEP 6. Background points and predictor screening")
    layers_base = {**base, **terrain}
    cells = np.argwhere(valid)
    pres_cells = set(zip(pres["row"], pres["col"]))
    nbg = min(N_BACKGROUND, len(cells))
    pick = cells[rng.choice(len(cells), nbg, replace=False)]
    bg = pd.DataFrame({"row": pick[:, 0], "col": pick[:, 1]})
    bg[LON_COL] = grid["lon"][bg["col"]]; bg[LAT_COL] = grid["lat"][bg["row"]]
    candidates = [v for v in BIO + TERRAIN if v not in EXCLUDE_FROM_MODEL]
    Xp = pd.DataFrame({v: layers_base[v][pres["row"], pres["col"]] for v in candidates})
    Xb = pd.DataFrame({v: layers_base[v][bg["row"], bg["col"]] for v in candidates})
    say(f"  presences {len(Xp)}, background points {len(Xb)} ({len(pres_cells & set(zip(bg.row, bg.col)))} shared cells)")

    screen = [v for v in candidates if v not in FORCE_KEEP]          # aspect is nominal: not screened
    fig_corr(Xb[screen], dirs["figures"] / "S_correlation_all_candidates.png", "Candidate predictors (background)")
    kept, pear_log = pearson_screen(Xb[screen], CORR_THRESHOLD, PRIORITY_VARS)
    pear_log.to_csv(dirs["tables"] / "screening_pearson_log.csv", index=False)
    kept, vif_final, vif_log = vif_screen(Xb[kept], VIF_THRESHOLD, PRIORITY_VARS)
    vif_log.to_csv(dirs["tables"] / "screening_vif_log.csv", index=False)
    vif_final.rename("VIF").to_csv(dirs["tables"] / "screening_vif_final.csv")
    if not any(v in kept for v in THERMAL_VARS):
        raise SystemExit("STOP: no temperature-level variable survived screening; put one first in PRIORITY_VARS.")
    VARS = [v for v in candidates if v in kept or v in FORCE_KEEP]   # in the order 1-23
    CONT = [v for v in VARS if v not in CATEGORICAL]                 # continuous (used for MESS)
    say(f"  retained {len(VARS)} predictors: " + ", ".join(f"{VAR_NUMBER[v]} {v}" for v in VARS))
    say(f"  max VIF {vif_final.max():.2f}" + (f"; always kept: {', '.join(FORCE_KEEP)}" if FORCE_KEEP else ""))
    dropped_by = {r["dropped"]: f"Pearson (r = {r['abs_r']} with {r['kept']})" for _, r in pear_log.iterrows()}
    dropped_by.update({r["dropped"]: f"VIF = {r['vif']}" for _, r in vif_log.iterrows()})
    pd.DataFrame({"number": [VAR_NUMBER[v] for v in candidates], "variable": candidates,
                  "label": [VAR_LABEL.get(v, v) for v in candidates],
                  "type": ["categorical" if v in CATEGORICAL else "continuous" for v in candidates],
                  "retained": [v in VARS for v in candidates],
                  "outcome": ["Retained (always kept)" if v in FORCE_KEEP else "Retained" if v in VARS
                              else f"Dropped: {dropped_by.get(v, '')}" for v in candidates]}
                 ).to_csv(dirs["tables"] / "screening_outcome.csv", index=False)
    fig_corr(Xb[[v for v in VARS if v in screen]], dirs["figures"] / "S_correlation_retained.png", "Retained predictors")
    for c in CATEGORICAL:
        if c in candidates:
            tab = pd.DataFrame({"code": list(CATEGORICAL[c]), "class": list(CATEGORICAL[c].values()),
                                "presences": [int((Xp[c] == k).sum()) for k in CATEGORICAL[c]],
                                "background": [int((Xb[c] == k).sum()) for k in CATEGORICAL[c]]})
            tab.to_csv(dirs["tables"] / f"classes_{c}.csv", index=False)

    X = pd.concat([Xp[VARS], Xb[VARS]], ignore_index=True)
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xb))].astype(int)
    lon = np.r_[pres[LON_COL].values, bg[LON_COL].values]
    lat = np.r_[pres[LAT_COL].values, bg[LAT_COL].values]
    ux, uy = to_utm(lon, lat)

    #  12.4 Spatial folds -
    section("STEP 7. Spatial blocks")
    folds, fold_tab = make_spatial_folds(np.asarray(ux), np.asarray(uy), y, N_FOLDS, BLOCK_SIZE_KM, RANDOM_STATE)
    pf = folds[:len(Xp)]                                                   # fold of each presence
    for col, vals in (("pres_lon_min", lambda f: pres[LON_COL].values[pf == f].min()),
                      ("pres_lon_max", lambda f: pres[LON_COL].values[pf == f].max()),
                      ("pres_lat_min", lambda f: pres[LAT_COL].values[pf == f].min()),
                      ("pres_lat_max", lambda f: pres[LAT_COL].values[pf == f].max()),
                      ("pres_median_elev_m", lambda f: np.nanmedian(elev_pres[pf == f]))):
        fold_tab[col] = [round(float(vals(f)), 3) if (pf == f).any() else np.nan for f in fold_tab["fold"]]
    fold_tab.to_csv(dirs["tables"] / "cv_folds.csv", index=False)
    say(fold_tab.to_string(index=False))
    if (fold_tab["presences"] < 0.1 * len(Xp)).any():
        say("  WARNING: a fold holds < 10% of presences; consider changing BLOCK_SIZE_KM.")
    fig_folds(lon, lat, folds, y, geom, dirs["figures"] / "S_cv_folds.png")
    all_folds = list(range(N_FOLDS))

    # - 12.5 Nested spatial cross-validation  
    section("STEP 8-9. Nested spatial cross-validation")
    oof = {a: np.full(len(y), np.nan) for a in ALGOS}
    oof_ens = np.full(len(y), np.nan)
    nested_rows, perm_rows = [], []
    for k in all_folds:
        inner = [f for f in all_folds if f != k]
        tr, te = folds != k, folds == k
        members, inner_auc, chosen = {}, {}, {}
        for a in ALGOS:
            p, tab = tune(a, X, y, folds, inner)
            inner_auc[a] = float(tab.loc[0, "cv_auc"]); chosen[a] = params_str(p)
            members[a] = Member(a, p).fit(X[tr], y[tr])
            oof[a][te] = members[a].predict(X[te])
        w = ensemble_weights(inner_auc)
        oof_ens[te] = ensemble_predict(members, w, X[te])
        row = {"outer_fold": k, "n_presences_test": int(y[te].sum())}
        for a in ALGOS:
            row[f"{a}_params"] = chosen[a]; row[f"{a}_inner_cv_auc"] = inner_auc[a]
            row[f"{a}_outer_auc"] = safe_auc(y[te], oof[a][te]); row[f"{a}_weight"] = w.get(a, 0.0)
        row["ensemble_outer_auc"] = safe_auc(y[te], oof_ens[te])
        row["median_oof_suitability_test_presences"] = float(np.median(oof_ens[te & (y == 1)]))
        row["median_oof_suitability_test_background"] = float(np.median(oof_ens[te & (y == 0)]))
        nested_rows.append(row)
        say(f"  outer fold {k}: " + ", ".join(f"{a} {row[f'{a}_outer_auc']:.3f}" for a in ALGOS) +
            f", ensemble {row['ensemble_outer_auc']:.3f}  (ensemble members: {', '.join(w)})")
        base_auc = row["ensemble_outer_auc"]
        Xte = X[te].reset_index(drop=True)
        for v in VARS:
            drops = []
            for r in range(N_PERMUTATIONS):
                Xperm = Xte.copy(); Xperm[v] = rng.permutation(Xperm[v].values)
                drops.append(base_auc - safe_auc(y[te], ensemble_predict(members, w, Xperm)))
            perm_rows.append({"outer_fold": k, "variable": v, "auc_drop": float(np.mean(drops))})
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(dirs["tables"] / "nested_cv_by_fold.csv", index=False)
    imp = (pd.DataFrame(perm_rows).groupby("variable")["auc_drop"].agg(["mean", "std"])
           .rename(columns={"mean": "mean_auc_drop", "std": "sd_across_folds"}).reset_index()
           .sort_values("mean_auc_drop", ascending=False))
    imp.to_csv(dirs["tables"] / "permutation_importance_nested.csv", index=False)
    fig_importance(imp, dirs["figures"] / "Fig_permutation_importance.png")

    # out-of-fold thresholds, Boyce, Moran's I
    th, th_tab = thresholds_from(oof_ens[y == 1], oof_ens[y == 0])
    th_tab.to_csv(dirs["tables"] / "thresholds_out_of_fold.csv", index=False)
    boyce, b_mid, b_ratio = boyce_continuous(oof_ens[y == 1], oof_ens[y == 0])
    mi, mp = morans_i(y - oof_ens, np.asarray(ux), np.asarray(uy))
    fig_roc(y, oof_ens, dirs["figures"] / "S_roc_out_of_fold.png")
    pt_rows = np.r_[pres["row"].values, bg["row"].values]; pt_cols = np.r_[pres["col"].values, bg["col"].values]
    pd.DataFrame({"x": lon, "y": lat, "presence": y, "fold": folds,
                  "elevation_m": terrain["elevation_m"][pt_rows, pt_cols].round(0),
                  **{v: X[v].values for v in VARS}, **{f"oof_{a}": oof[a] for a in ALGOS},
                  "oof_ensemble": oof_ens}).to_csv(dirs["tables"] / "points_folds_oof_predictions.csv", index=False)
    say("\n" + th_tab.round(3).to_string(index=False))
    say(f"  pooled out-of-fold AUC: ensemble {safe_auc(y, oof_ens):.3f}; "
        + ", ".join(f"{a} {safe_auc(y, oof[a]):.3f}" for a in ALGOS))
    say(f"  continuous Boyce index (out-of-fold) {boyce:.3f};  Moran's I of OOF residuals {mi:.3f} (p = {mp:.3f})")
    uxa, uya = np.asarray(ux), np.asarray(uy)
    mi_p, mp_p = morans_i((y - oof_ens)[y == 1], uxa[y == 1], uya[y == 1])
    mi_b, mp_b = morans_i((y - oof_ens)[y == 0], uxa[y == 0], uya[y == 0])
    say(f"  Moran's I of OOF residuals: presences only {mi_p:.3f} (p = {mp_p:.3f}); background only {mi_b:.3f} (p = {mp_b:.3f})")
    dem30 = OUTPUT_DIR / "rasters" / f"terrain_{TERRAIN_RES_M}m" / f"terrain_{TERRAIN_RES_M}m_elevation_m.tif"
    audit = records_audit(pres, pf, oof_ens[y == 1], Xp.reset_index(drop=True), CONT, elev_pres, districts, dem30)
    audit.to_csv(dirs["tables"] / "records_to_check.csv", index=False)
    nflag = int((audit["n_flags"] > 0).sum())
    say(f"  records_to_check.csv: {nflag} of {len(audit)} presence cells flagged "
        f"({int(audit['check'].str.contains('below').sum())} low elevation, "
        f"{int(audit['check'].str.contains('not predicted').sum())} not predicted when withheld, "
        f"{int(audit['check'].str.contains('outside').sum())} climate outside other folds)")
    if districts is not None and "district" in audit:
        bad = audit[audit["check"].str.contains("not predicted")].groupby("district").size()
        if len(bad):
            say("  presences not predicted when withheld, by district: " + ", ".join(f"{k} {v}" for k, v in bad.items()))

    #  12.6 Final models on all data 
    section("STEP 8. Final models (tuned on all four folds, fitted to all data)")
    final, cv_auc, tune_tabs = {}, {}, []
    for a in ALGOS:
        p, tab = tune(a, X, y, folds, all_folds)
        tune_tabs.append(tab.drop(columns="_p"))
        cv_auc[a] = float(tab.loc[0, "cv_auc"])
        final[a] = Member(a, p).fit(X, y)
        say(f"  {a:6s} best: {params_str(p)}   CV AUC {tab.loc[0, 'cv_auc']:.3f} ± {tab.loc[0, 'cv_sd']:.3f}")
    pd.concat(tune_tabs).to_csv(dirs["tables"] / "tuning_all_settings.csv", index=False)
    W = ensemble_weights(cv_auc)
    say(f"  ensemble weights: {', '.join(f'{a} {w:.3f}' for a, w in W.items())}"
        + ("" if len(W) == 3 else f"   (excluded: {[a for a in ALGOS if a not in W]}, CV AUC < {MIN_ENSEMBLE_AUC})"))
    perf = pd.DataFrame([{
        "model": a, "cv_auc": cv_auc[a],
        "cv_sd": float(pd.concat(tune_tabs).query("algorithm == @a").iloc[0]["cv_sd"]),
        "nested_outer_auc_mean": nested[f"{a}_outer_auc"].mean(),
        "nested_outer_auc_sd": nested[f"{a}_outer_auc"].std(ddof=0),
        "pooled_oof_auc": safe_auc(y, oof[a]),
        "in_sample_auc": safe_auc(y, final[a].predict(X)), "weight": W.get(a, 0.0)} for a in ALGOS] + [{
        "model": "ensemble", "cv_auc": np.nan, "cv_sd": np.nan,
        "nested_outer_auc_mean": nested["ensemble_outer_auc"].mean(),
        "nested_outer_auc_sd": nested["ensemble_outer_auc"].std(ddof=0),
        "pooled_oof_auc": safe_auc(y, oof_ens),
        "in_sample_auc": safe_auc(y, ensemble_predict(final, W, X)), "weight": 1.0}])
    perf.to_csv(dirs["tables"] / "Table1_model_performance.csv", index=False)
    say("\n" + perf.round(3).to_string(index=False))
    fig_response(final, W, Xb[VARS], VARS, dirs["figures"] / "Fig_response_curves.png", Xpres=Xp[VARS])

    #  12.7 Projection and novelty 
    section("STEP 10. Projection, novelty and range change")
    scen_layers = {"Baseline_1970_2000": layers_base, **{s: {**future[s], **terrain} for s in future}}
    ref = Xp[CONT] if MESS_REFERENCE == "presence" else pd.concat([Xp[CONT], Xb[CONT]])
    ens_maps, mess_maps, mod_maps = {}, {}, {}
    nov_rows = []
    for s, lay in scen_layers.items():
        ens, mem = predict_map(final, W, lay, VARS, valid)
        m, mod = mess(lay, ref, CONT, valid)
        m_cal, _ = mess(lay, pd.concat([Xp[CONT], Xb[CONT]]), CONT, valid)
        ens_maps[s], mess_maps[s], mod_maps[s] = ens, m, mod
        save_raster(dirs["rasters"] / f"{s}_ensemble.tif", ens, grid)
        for a, arr in mem.items():
            save_raster(dirs["rasters"] / f"{s}_{a}.tif", arr, grid)
        save_raster(dirs["rasters"] / f"{s}_mess_{MESS_REFERENCE}.tif", m, grid)
        save_raster(dirs["rasters"] / f"{s}_mod.tif", np.where(valid, mod, -1).astype(float), grid, "int16", -1)
        A = grid["cell_area"][valid]
        row = {"scenario": s,
               "pct_area_mess_lt0": 100 * A[m[valid] < 0].sum() / A.sum(),
               f"pct_area_mess_lt{int(abs(MESS_THRESHOLD))}": 100 * A[m[valid] < MESS_THRESHOLD].sum() / A.sum(),
               "mean_mess": float(np.mean(m[valid])),
               "pct_area_mess_lt0_calibration_ref": 100 * A[m_cal[valid] < 0].sum() / A.sum()}
        for j, v in enumerate(CONT):
            row[f"mod_share_{v}"] = 100 * A[(mod[valid] == j) & (m[valid] < 0)].sum() / max(A[m[valid] < 0].sum(), 1e-9)
        nov_rows.append(row)
    nov = pd.DataFrame(nov_rows)
    nov.to_csv(dirs["tables"] / "novelty_summary.csv", index=False)
    say(nov.iloc[:, :4].round(1).to_string(index=False))

    #  12.8 Range change (masked and unmasked, all thresholds) 
    UX = np.repeat(grid["lon"][None, :], grid["shape"][0], 0); UY = np.repeat(grid["lat"][:, None], grid["shape"][1], 1)
    UXm, UYm = to_utm(UX.ravel(), UY.ravel())
    UXm, UYm = np.asarray(UXm).reshape(UX.shape), np.asarray(UYm).reshape(UY.shape)
    elev = terrain["elevation_m"]
    rc_rows, cats = [], {}
    b_ens, b_mess = ens_maps["Baseline_1970_2000"], mess_maps["Baseline_1970_2000"]
    for tname, t in th.items():
        for masked in (True, False):
            b_known = valid & ((b_mess >= MESS_THRESHOLD) if masked else True)
            b_suit = b_known & (b_ens >= t)
            if tname == PRIMARY_THRESHOLD:
                save_raster(dirs["rasters"] / f"Baseline_1970_2000_binary_{tname}_{'masked' if masked else 'unmasked'}.tif",
                            np.where(valid, b_suit, np.nan), grid, "uint8", 255)
            for s in FUTURE_CLIMATE:
                f_known = valid & ((mess_maps[s] >= MESS_THRESHOLD) if masked else True)
                f_suit = f_known & (ens_maps[s] >= t)
                r = range_change(b_suit, b_known, f_suit, f_known, grid, UXm, UYm, elev)
                rc_rows.append({"threshold": tname, "cutoff": t, "novelty_mask": masked, "scenario": s, **r})
                if tname == PRIMARY_THRESHOLD:
                    save_raster(dirs["rasters"] / f"{s}_binary_{tname}_{'masked' if masked else 'unmasked'}.tif",
                                np.where(valid, f_suit, np.nan), grid, "uint8", 255)
                    if masked:
                        c = np.full(grid["shape"], np.nan)
                        c[valid & b_known & f_known] = 0
                        c[b_suit & f_suit] = 1
                        c[b_suit & ~f_suit & f_known] = 2
                        c[b_suit & ~f_known] = 3
                        c[f_suit & ~b_suit & b_known] = 4
                        cats[s] = c
    rc = pd.DataFrame(rc_rows)
    rc.to_csv(dirs["tables"] / "range_change_all_thresholds.csv", index=False)
    A_valid = grid["cell_area"][valid].sum()
    tchk = pd.DataFrame([{"threshold": k, "cutoff": t,
                          "baseline_suitable_pct_of_area_unmasked": 100 * grid["cell_area"][valid & (b_ens >= t)].sum() / A_valid}
                         for k, t in th.items()])
    tchk = tchk.merge(th_tab[["threshold", "sensitivity", "specificity", "TSS"]], on="threshold")
    tchk.to_csv(dirs["tables"] / "threshold_area_check.csv", index=False)
    say("\nShare of the modelled area classed suitable at baseline (unmasked), by threshold:")
    say(tchk.round(3).to_string(index=False))
    share = float(tchk.loc[tchk.threshold == PRIMARY_THRESHOLD, "baseline_suitable_pct_of_area_unmasked"].iloc[0])
    if share > 70 or share < 2:
        say(f"  WARNING: {PRIMARY_THRESHOLD} classes {share:.0f}% of the study area as suitable at baseline; "
            "binary range-change results will not be meaningful. Choose another PRIMARY_THRESHOLD.")
    #  MESS sensitivity: threshold (0, -20, -40) and variables (only predictors that matter) 
    impv = imp.set_index("variable")["mean_auc_drop"]
    sens_vars = [v for v in CONT if v in impv and impv[v] >= MESS_SENS_MIN_IMPORTANCE]
    variants = [("all continuous predictors", CONT, mess_maps, thr) for thr in (0.0, MESS_THRESHOLD, -40.0)]
    if 2 <= len(sens_vars) < len(CONT):
        mess_s = {s: mess(lay, ref[sens_vars], sens_vars, valid)[0] for s, lay in scen_layers.items()}
        variants.append(("predictors with importance >= %.2f" % MESS_SENS_MIN_IMPORTANCE, sens_vars, mess_s, MESS_THRESHOLD))
    t = th[PRIMARY_THRESHOLD]
    A_v = grid["cell_area"][valid]
    ms_rows = []
    for label, vars_, mm, thr in variants:
        bk = valid & (mm["Baseline_1970_2000"] >= thr)
        bs = bk & (b_ens >= t)
        for s in FUTURE_CLIMATE:
            fk = valid & (mm[s] >= thr)
            r = range_change(bs, bk, fk & (ens_maps[s] >= t), fk, grid, UXm, UYm, elev)
            ms_rows.append({"mess_variables": label, "variables": "+".join(vars_), "mess_threshold": thr,
                            "threshold": PRIMARY_THRESHOLD, "scenario": s,
                            "pct_area_novel_baseline": 100 * A_v[mm["Baseline_1970_2000"][valid] < thr].sum() / A_v.sum(),
                            "pct_area_novel_future": 100 * A_v[mm[s][valid] < thr].sum() / A_v.sum(), **r})
    mess_sens = pd.DataFrame(ms_rows)
    mess_sens.to_csv(dirs["tables"] / "range_change_mess_sensitivity.csv", index=False)
    say("\nMasked range change (%) by MESS threshold and MESS variables:")
    piv = mess_sens.assign(variant=mess_sens["mess_threshold"].map(lambda v: f"MESS<{v:g}") +
                           np.where(mess_sens["mess_variables"].str.startswith("all"), "", " (important vars)"))
    say(piv.pivot(index="scenario", columns="variant", values="change_pct").round(1).to_string())
    prim = rc[rc.threshold == PRIMARY_THRESHOLD]
    say(f"\nRange change at {PRIMARY_THRESHOLD} = {th[PRIMARY_THRESHOLD]:.3f}:")
    cols_show = ["scenario", "novelty_mask", "baseline_km2", "future_km2", "change_pct", "lost_km2",
                 "share_of_loss_from_novelty_pct", "gained_km2", "turnover_pct", "centroid_shift_km",
                 "centroid_direction", "median_elev_shift_m"]
    say(prim[cols_show].round(1).to_string(index=False))

    fig_continuous(ens_maps, {s: valid & (mess_maps[s] < MESS_THRESHOLD) for s in ens_maps}, grid,
                   dirs["figures"] / "S_continuous_suitability.png", geom, districts)
    fig_range_change(cats, grid, dirs["figures"] / "Fig_range_change.png", geom, districts)
    fig_mess(mess_maps, mod_maps, CONT, grid, dirs["figures"] / "S_mess.png",
             dirs["figures"] / "S_mess_most_dissimilar_variable.png", geom, districts)
    fig_study_area_v3(elev, grid, pres, geom, districts, dirs["figures"] / "Fig1_study_area.png")
    fig_records_check(audit, grid, geom, districts, dirs["figures"] / "S_records_to_check.png")
    bg_vals = Xb[CONT].assign(cell_mean_elevation_m=terrain["elevation_m"][bg["row"].values, bg["col"].values])
    fig_flagged_climate(audit, bg_vals, CONT, dirs["figures"] / "S_flagged_presences_climate.png")

    # climate change over the baseline-suitable area (helps the Discussion)
    b_suit_un = valid & (b_ens >= th[PRIMARY_THRESHOLD])
    cc_suit = climate_change_table(base, future, b_suit_un)
    cc_suit.to_csv(dirs["tables"] / "climate_change_baseline_suitable_area.csv", index=False)

    # gained-area novelty for every scenario
    gain_rows = []
    for s in FUTURE_CLIMATE:
        g = (valid & (ens_maps[s] >= th[PRIMARY_THRESHOLD])) & ~b_suit_un
        if g.any():
            gain_rows.append({"scenario": s, "gained_cells_unmasked": int(g.sum()),
                              "pct_gained_mess_lt0": 100 * np.mean(mess_maps[s][g] < 0),
                              "pct_gained_mess_lt_thr": 100 * np.mean(mess_maps[s][g] < MESS_THRESHOLD),
                              "mean_mess_gained": float(np.mean(mess_maps[s][g]))})
    pd.DataFrame(gain_rows).to_csv(dirs["tables"] / "novelty_of_gained_area.csv", index=False)

    #  12.9 Bootstrap of presences 
    section(f"STEP 10b. Bootstrap of presences ({N_BOOTSTRAP} replicates)")
    best = {a: final[a].params for a in W}
    t_fixed = th[PRIMARY_THRESHOLD]
    pres_idx, bg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    stacks = {s: stack_frame(lay, VARS, valid) for s, lay in scen_layers.items()}
    A = grid["cell_area"][valid]
    mess_known = {s: mess_maps[s][valid] >= MESS_THRESHOLD for s in scen_layers}
    boot = []
    for b in range(N_BOOTSTRAP):
        idx = np.r_[rng.choice(pres_idx, len(pres_idx), replace=True), bg_idx]
        Xb_, yb_ = X.iloc[idx].reset_index(drop=True), y[idx]
        mem = {a: Member(a, best[a], seed=b).fit(Xb_, yb_) for a in W}
        pred = {s: sum(w * mem[a].predict(stacks[s]) for a, w in W.items()) for s in stacks}
        for masked in (True, False):
            bk = mess_known["Baseline_1970_2000"] if masked else np.ones(len(A), bool)
            base_area = A[bk & (pred["Baseline_1970_2000"] >= t_fixed)].sum()
            for s in FUTURE_CLIMATE:
                fk = mess_known[s] if masked else np.ones(len(A), bool)
                fa = A[fk & (pred[s] >= t_fixed)].sum()
                boot.append({"replicate": b, "novelty_mask": masked, "scenario": s,
                             "change_pct": 100 * (fa - base_area) / base_area if base_area else np.nan})
        if (b + 1) % 10 == 0 or b == N_BOOTSTRAP - 1:
            say(f"  replicate {b+1}/{N_BOOTSTRAP}")
    boot = pd.DataFrame(boot)
    boot.to_csv(dirs["tables"] / "bootstrap_replicates.csv", index=False)
    bsum = (boot.groupby(["scenario", "novelty_mask"])["change_pct"]
            .agg(median="median", lo=lambda v: np.nanpercentile(v, 2.5), hi=lambda v: np.nanpercentile(v, 97.5),
                 share_positive=lambda v: 100 * np.mean(v > 0)).reset_index())
    bsum.to_csv(dirs["tables"] / "bootstrap_summary.csv", index=False)
    say(bsum.round(1).to_string(index=False))

    #  12.10 Key numbers for the manuscript -
    section("STEP 12. Summary")
    key = {"software": versions, "gcm": GCM_NAME, "occurrences": occ_rep, "n_background": int(len(Xb)),
           "retained_predictors": VARS, "max_vif": float(vif_final.max()),
           "folds": fold_tab.to_dict("records"), "thresholds": th,
           "boyce_out_of_fold": boyce, "morans_i_oof": {"I": mi, "p": mp},
           "morans_i_oof_presences_only": {"I": mi_p, "p": mp_p}, "morans_i_oof_background_only": {"I": mi_b, "p": mp_b},
           "records_flagged": int((audit["n_flags"] > 0).sum()),
           "mess_sensitivity_variables": sens_vars,
           "ensemble_weights": W, "performance": perf.to_dict("records"),
           "study_area_km2": float(grid["cell_area"][grid["inside"]].sum()),
           "modelled_area_km2": float(grid["cell_area"][valid].sum()),
           "elevation_occurrences_m": {"median": float(np.nanmedian(elev_pres)), "min": float(np.nanmin(elev_pres)),
                                       "max": float(np.nanmax(elev_pres))},
           "warnings": warn}
    with open(OUTPUT_DIR / "key_results.json", "w") as f:
        json.dump(key, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    say(f"Finished in {(time.time()-T0)/60:.1f} min. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
