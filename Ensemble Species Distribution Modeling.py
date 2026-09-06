
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import matplotlib
matplotlib.use('Agg')  # Forces non-interactive plotting to prevent Tkinter thread errors
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
import elapid as ela

from sklearn.metrics import (
    roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve,
    average_precision_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


import matplotlib
matplotlib.use('Agg') 


import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
import elapid as ela

from sklearn.metrics import (
    roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve,
    average_precision_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')



BASE_DIR = Path(r"E:\maxent\Mission Jatamansi")
OUTPUT_DIR = Path(r"E:\maxent\result\OUTPUT_PY_result_2_TrueEnsemble")

OCCURRENCE_CSV = BASE_DIR / "N_jatamansi_thinned_1km.csv"
STUDY_AREA_SHP = BASE_DIR / "Input" / "output" / "study_area" / "study_area.shp"

LAT_COL = "Latitude"
LON_COL = "Longitude"

STATIC_RASTER_FOLDER = BASE_DIR / "topographic"
STATIC_VAR_NAMES = ["slope", "aspect", "elevation", "tri"]

SCENARIOS = {
    
    "Baseline_1970_2000": BASE_DIR / "1970_2000",
    "SSP45_2021_2040":    BASE_DIR / "4.5 2021_2040",
    "SSP45_2061_2080":    BASE_DIR / "4.5 61_81",
    "SSP45_2081_2100":    BASE_DIR / "4.5 81-2100",
    "SSP85_2021_2040":    BASE_DIR / "8.5_2021-2040",
    "SSP85_2061_2080":    BASE_DIR / "8.5 61_81",
    "SSP85_2081_2100":    BASE_DIR / "8.5 81-2100",
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CORR_THRESHOLD = 0.70
N_BLOCKS_X, N_BLOCKS_Y = 3, 3
BETA_CANDIDATES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
FEATURE_CANDIDATES = [['linear'], ['linear', 'quadratic']]
MESS_NOVELTY_THRESHOLD = -20.0
ENABLE_ENSEMBLE = False


CONTINUOUS_DIR = OUTPUT_DIR / "Continuous_Probability_Maps"
BINARY_DIR = OUTPUT_DIR / "Binary_Suitability_Maps"
MESS_DIR = OUTPUT_DIR / "MESS_Novelty_Maps"
FIGURES_DIR = OUTPUT_DIR / "Evaluation_Plots"
DIAGNOSTICS_DIR = OUTPUT_DIR / "Diagnostics"

for folder in [OUTPUT_DIR, CONTINUOUS_DIR, BINARY_DIR, MESS_DIR, FIGURES_DIR, DIAGNOSTICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def get_raster_list(folder_path, static_folder=None, static_names=None):
    raster_files = []
    if folder_path and folder_path.exists():
        for ext in ["*.asc", "*.tif"]:
            raster_files.extend(sorted([str(p) for p in folder_path.glob(ext)]))

    if static_names:
        if not static_folder:
            raise FileNotFoundError("STATIC_VAR_NAMES is set but STATIC_RASTER_FOLDER is None.")
        static_folder = Path(static_folder)
        if not static_folder.exists():
            raise FileNotFoundError(f"STATIC_RASTER_FOLDER does not exist: {static_folder}")
        
        all_static_files = []
        for ext in ["*.asc", "*.tif"]:
            all_static_files.extend(sorted(static_folder.glob(ext)))
        if not all_static_files:
            raise FileNotFoundError(f"STATIC_RASTER_FOLDER exists but contains no .asc/.tif files: {static_folder}")

        for name in static_names:
            key = name.lower()
            matches = [p for p in all_static_files if p.stem.lower() == key or p.stem.lower().startswith(key + "_")]
            if len(matches) == 0:
                available = sorted(p.stem for p in all_static_files)
                raise FileNotFoundError(f"No raster found for static variable '{name}' in {static_folder}. Available: {available}")
            if len(matches) > 1:
                raise ValueError(f"Ambiguous match for static variable '{name}' in {static_folder}: {[p.name for p in matches]}")
            raster_files.append(str(matches[0]))

    return sorted(list(set(raster_files)))


def resolve_retained_raster_names(baseline_raster_list, retained_columns):
    stems_by_position = [Path(p).stem for p in baseline_raster_list]
    resolved = {}
    for col in retained_columns:
        m = re.match(r"^b(\d+)$", col)
        if not m:
            raise ValueError(f"Unexpected column name '{col}'.")
        idx = int(m.group(1)) - 1
        if idx >= len(stems_by_position):
            raise ValueError(f"Column '{col}' refers to raster index {idx}, but only {len(stems_by_position)} baseline rasters were found.")
        resolved[col] = stems_by_position[idx]
    return resolved


def select_rasters_by_name(raster_list, variable_names):
    name_to_path = {Path(p).stem.lower(): p for p in raster_list}
    selected, missing = [], []
    for var in variable_names:
        key = var.lower()
        if key in name_to_path:
            selected.append(name_to_path[key])
        else:
            missing.append(var)
    if missing:
        raise ValueError(f"Could not find raster(s) for variable(s) {missing}. Available: {sorted(name_to_path.keys())}")
    return selected


def align_raster_to_reference(src_raster_path, ref_meta, dst_raster_path, resampling=Resampling.bilinear):
    fill_value = ref_meta.get('nodata')
    if fill_value is None:
        fill_value = -9999
    dst_dtype = ref_meta.get('dtype') or np.float32
    with rio.open(src_raster_path) as src:
        destination = np.full((ref_meta['height'], ref_meta['width']), fill_value, dtype=dst_dtype)
        reproject(source=rio.band(src, 1), destination=destination,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
                  resampling=resampling)
    with rio.open(dst_raster_path, 'w', **ref_meta) as dst:
        dst.write(destination, 1)


def _align_array_to_grid(src_arr, src_meta, ref_meta, resampling=Resampling.bilinear, fill_value=None):
    """In-memory equivalent of align_raster_to_reference for a numpy array that
    is already in memory (avoids an extra disk round-trip). Used to align the
    per-scenario ENSEMBLE array onto a common MASTER_META grid (captured from
    the baseline scenario) before MESS/masking/binarization, since different
    scenario climate rasters can have different extents/resolutions (confirmed
    by the 'Grid shape mismatch: baseline (181, 297) vs SSP45 (179, 295)' error
    this fixes).
    """
    if fill_value is None:
        fill_value = ref_meta.get('nodata')
        if fill_value is None:
            fill_value = -9999
    destination = np.full((ref_meta['height'], ref_meta['width']), fill_value, dtype=np.float32)
    reproject(source=src_arr, destination=destination,
              src_transform=src_meta['transform'], src_crs=src_meta['crs'],
              dst_transform=ref_meta['transform'], dst_crs=ref_meta['crs'],
              resampling=resampling, src_nodata=src_meta.get('nodata'), dst_nodata=fill_value)
    return destination


def compute_thresholds(y_true, y_prob):
    thresholds = np.linspace(0.001, 0.999, 1000)
    best_mtss, max_sum = 0.0, -1.0
    best_mtss_sens, best_mtss_spec = 0.0, 0.0
    for th in thresholds:
        y_bin = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if sens + spec > max_sum:
            max_sum = sens + spec
            best_mtss = th
            best_mtss_sens, best_mtss_spec = sens, spec
    tss_scores = []
    for th in thresholds:
        y_bin = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss_scores.append(sens + spec - 1)
    best_tss = thresholds[np.argmax(tss_scores)]
    presence_probs = y_prob[y_true == 1]
    p10 = np.percentile(presence_probs, 10) if len(presence_probs) > 0 else 0.1
    p5 = np.percentile(presence_probs, 5) if len(presence_probs) > 0 else 0.05
    best_ets, min_diff = 0.0, float('inf')
    for th in thresholds:
        y_bin = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        diff = abs(sens - spec)
        if diff < min_diff:
            min_diff = diff
            best_ets = th
    return {'MTSS': best_mtss, 'TSS': best_tss, 'P10': p10, 'P5': p5, 'ETS': best_ets,
            'MTSS_sens': best_mtss_sens, 'MTSS_spec': best_mtss_spec}


def evaluate_model(y_true, y_prob, threshold):
    y_bin = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {'AUC': roc_auc_score(y_true, y_prob), 'AUC_PR': average_precision_score(y_true, y_prob),
            'TSS': sens + spec - 1, 'Kappa': cohen_kappa_score(y_true, y_bin),
            'Sensitivity': sens, 'Specificity': spec, 'Threshold': threshold}


def compute_vif(df):
    """ADDED: Variance Inflation Factor for each column, computed as 1/(1-R^2)
    from regressing that column on all other columns. Pairwise Pearson |r|
    screening (already used elsewhere) can miss higher-order collinearity where
    three or more variables are jointly redundant even though no single pair
    exceeds the threshold; VIF catches that. No pair with |r| > 0.8-0.9ish is
    needed for VIF to flag a problem \u2014 it's a complementary check, not a
    replacement for the existing pairwise screening.
    """
    from sklearn.linear_model import LinearRegression
    cols = df.columns.tolist()
    rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            rows.append({'Variable': col, 'VIF': np.nan})
            continue
        reg = LinearRegression().fit(df[others].values, df[col].values)
        r2 = reg.score(df[others].values, df[col].values)
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append({'Variable': col, 'VIF': vif})
    return pd.DataFrame(rows).sort_values('VIF', ascending=False)


def iterative_vif_reduction(df, threshold=10.0, priority_vars=None):
    """ADDED: Iteratively drop the single highest-VIF variable and recompute,
    until every remaining variable has VIF below `threshold` (or only one
    variable remains). Pairwise Pearson correlation (already used upstream in
    remove_collinear_predictors) can miss higher-order collinearity -- the
    canonical WorldClim example is Bio3 (isothermality), which is DEFINED as
    100 x Bio2/Bio7, so all three can each have pairwise |r| < 0.7 with one
    another while still being near-perfectly linearly dependent as a triplet.
    Only a multi-variable check like VIF catches that.

    priority_vars (optional): variables to avoid dropping if possible -- if the
    single highest-VIF variable is a priority variable, the next-highest
    non-priority variable is dropped instead; a priority variable is only
    dropped if every remaining variable is a priority variable.

    Returns (reduced_df, dropped_variable_list, iteration_history_df).
    """
    current = df.copy()
    dropped = []
    history = []
    priority_vars = priority_vars or []
    while current.shape[1] > 1:
        vif_df = compute_vif(current).replace(np.inf, 1e12)
        max_vif = vif_df['VIF'].max()
        history.append({'n_remaining': current.shape[1], 'max_vif': max_vif,
                        'worst_variable': vif_df.iloc[0]['Variable']})
        if max_vif <= threshold:
            break
        candidates = vif_df.sort_values('VIF', ascending=False)
        to_drop = next((row['Variable'] for _, row in candidates.iterrows()
                       if row['Variable'] not in priority_vars), candidates.iloc[0]['Variable'])
        current = current.drop(columns=[to_drop])
        dropped.append(to_drop)
    return current, dropped, pd.DataFrame(history)


def compute_boyce_index(pred_all, pred_presence, n_bins=10):
    """ADDED: Continuous Boyce Index (Hirzel et al. 2006). AUC is known to be an
    imperfect metric for presence-background models (sensitive to the arbitrary
    background sample, doesn't assess calibration). The Boyce index instead checks
    whether presence points fall disproportionately in HIGH-suitability bins
    relative to what's expected by chance across the full predicted surface.
    pred_all = predicted suitability across presence+background (the 'expected'
    distribution); pred_presence = predicted suitability at presence points only.
    Returns a value in [-1, 1]; strongly positive = good; near zero = no better
    than random; negative = model favors low-suitability areas (a red flag).
    """
    from scipy.stats import spearmanr
    mini, maxi = pred_all.min(), pred_all.max()
    bin_edges = np.linspace(mini, maxi, n_bins + 1)
    bin_mids, f_ratios = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin_all = np.sum((pred_all >= lo) & (pred_all <= hi))
        in_bin_pres = np.sum((pred_presence >= lo) & (pred_presence <= hi))
        if in_bin_all == 0:
            continue
        expected = in_bin_all / len(pred_all)
        observed = in_bin_pres / len(pred_presence)
        if expected > 0:
            f_ratios.append(observed / expected)
            bin_mids.append((lo + hi) / 2)
    if len(f_ratios) < 3:
        return np.nan
    rho, _ = spearmanr(bin_mids, f_ratios)
    return rho


def compute_morans_i(residuals, x_coords, y_coords, k=8, n_permutations=199, random_state=42):
    """ADDED: Global Moran's I on model residuals (observed class - predicted
    probability), using a k-nearest-neighbor row-standardized spatial weight
    matrix. This directly checks whether spatial block CV (already used for
    hyperparameter tuning) actually achieved its purpose -- if residuals still
    show significant positive spatial autocorrelation, nearby points still share
    unexplained similarity, meaning the reported CV AUC may still be optimistic.
    Returns (observed_I, p_value); p_value from a permutation test (residuals
    randomly reshuffled across locations n_permutations times).
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(residuals)
    coords = np.column_stack([x_coords, y_coords])
    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    idx = idx[:, 1:]  # drop the point itself

    def moran_stat(z_vals):
        num = sum(z_vals[i] * np.sum(z_vals[idx[i]]) / k_eff for i in range(n))
        denom = np.sum(z_vals ** 2)
        return (n / (n * k_eff)) * (num / denom) if denom > 0 else 0.0

    z = residuals - residuals.mean()
    observed_I = moran_stat(z)
    rng = np.random.RandomState(random_state)
    perm_Is = np.array([moran_stat(rng.permutation(z)) for _ in range(n_permutations)])
    p_value = (np.sum(np.abs(perm_Is) >= np.abs(observed_I)) + 1) / (n_permutations + 1)
    return observed_I, p_value


def remove_collinear_predictors(df, threshold=0.70, priority_vars=None):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if val > threshold:
                pairs.append((val, row, col))
    pairs.sort(reverse=True)
    dropped = set()
    for _, var_a, var_b in pairs:
        if var_a in dropped or var_b in dropped:
            continue
        if priority_vars:
            a_priority = var_a in priority_vars
            b_priority = var_b in priority_vars
            if a_priority and not b_priority:
                drop = var_b
            elif b_priority and not a_priority:
                drop = var_a
            else:
                drop = var_a if corr[var_a].mean() > corr[var_b].mean() else var_b
        else:
            drop = var_a if corr[var_a].mean() > corr[var_b].mean() else var_b
        dropped.add(drop)
    retained = [c for c in df.columns if c not in dropped]
    return df[retained], list(dropped)


def compute_raster_mess(reference_df, scenario_raster_paths, output_mess_path, master_meta):
    """
    BUG FIX: the original version read every raster in scenario_raster_paths at
    its NATIVE resolution/shape and passed the results straight to np.stack(),
    which requires every array to have an identical shape. This works fine when
    all retained predictors happen to come from the same source (e.g. the
    all-topographic degenerate run), but breaks as soon as predictors are mixed
    from different native grids — which is exactly the real case here: WorldClim
    bioclim rasters (~1 km) and SRTM-derived topographic rasters (finer, and
    possibly a different extent) do not share a pixel grid.

    Fix: reproject/resample every input raster onto master_meta's grid (the same
    grid the ensemble prediction rasters already share, since it comes from the
    already-projected maxent_out for this scenario) BEFORE reading it into the
    stack, using the same align_raster_to_reference() helper already used
    elsewhere in this script. Temporary aligned copies are cleaned up after use.
    """
    raster_arrays = []
    col_names = reference_df.columns.tolist()
    tmp_aligned_paths = []
    try:
        for p in scenario_raster_paths:
            p = Path(p)
            with rio.open(p) as src:
                same_grid = (
                    src.height == master_meta['height'] and
                    src.width == master_meta['width'] and
                    src.transform == master_meta['transform'] and
                    src.crs == master_meta['crs']
                )
            if same_grid:
                read_path = p
            else:
                aligned_p = output_mess_path.parent / f"_tmp_mess_align_{p.stem}.tif"
                align_raster_to_reference(p, master_meta, aligned_p)
                tmp_aligned_paths.append(aligned_p)
                read_path = aligned_p

            with rio.open(read_path) as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                raster_arrays.append(data)

        stacked = np.stack(raster_arrays, axis=0)
    finally:
        for tp in tmp_aligned_paths:
            if tp.exists():
                os.remove(tp)

    n_vars, height, width = stacked.shape
    mess_map = np.full((height, width), -9999.0, dtype=np.float32)
    valid_mask = ~np.isnan(stacked).any(axis=0)
    if not np.any(valid_mask):
        return
    valid_pixels = stacked[:, valid_mask].T
    proj_df = pd.DataFrame(valid_pixels, columns=col_names)
    min_sim_per_var = []
    for col in col_names:
        ref_vals = reference_df[col].values
        proj_vals = proj_df[col].values
        f = np.searchsorted(np.sort(ref_vals), proj_vals, side='right') / len(ref_vals) * 100
        ref_min, ref_max = ref_vals.min(), ref_vals.max()
        ref_range = ref_max - ref_min if (ref_max - ref_min) > 0 else 1e-6
        sim = np.where(f == 0, (proj_vals - ref_min) / ref_range * 100,
                       np.where(f <= 50, 2 * f, np.where(f < 100, 2 * (100 - f),
                                        (ref_max - proj_vals) / ref_range * 100)))
        min_sim_per_var.append(sim)
    mess_values = np.min(np.array(min_sim_per_var), axis=0)
    mess_map[valid_mask] = mess_values
    tmp_mess = output_mess_path.parent / f"_tmp_{output_mess_path.name}"
    mess_meta = master_meta.copy()
    mess_meta.update(dtype=rio.float32, nodata=-9999.0)
    with rio.open(tmp_mess, 'w', **mess_meta) as dst:
        dst.write(mess_map.astype(np.float32), 1)
    align_raster_to_reference(tmp_mess, master_meta, output_mess_path)
    if os.path.exists(tmp_mess):
        os.remove(tmp_mess)


class _RasterProbaWrapper:
    """Wraps scikit-learn models to mimic elapid.MaxentModel's .predict(df) interface."""
    def __init__(self, sk_model, feature_order, scaler=None):
        self.sk_model = sk_model
        self.feature_order = list(feature_order)
        self.scaler = scaler

    def predict(self, df):
        # Handle DataFrame vs 2D NumPy Array passed by elapid raster tools
        if isinstance(df, pd.DataFrame):
            X = df[self.feature_order].values
        else:
            X = np.asarray(df)

        # Scale parameters if model requires it (e.g., Logistic Regression)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.sk_model.predict_proba(X)[:, 1]



if __name__ == "__main__":
    print("=" * 70)
    print("  SDM PIPELINE: Nardostachys jatamansi (TRUE ENSEMBLE PATCH)")
    print("=" * 70)

    #  Load occurrence data
    print("\n[INFO] Loading occurrence data...")
    df_occ = pd.read_csv(OCCURRENCE_CSV)
    df_occ.columns = df_occ.columns.str.strip()
    missing = [c for c in (LON_COL, LAT_COL) if c not in df_occ.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    presence_gdf = ela.xy_to_geoseries(df_occ[LON_COL], df_occ[LAT_COL], crs="EPSG:4326")
    presence_gdf = gpd.GeoDataFrame(geometry=presence_gdf)
    n_presence = len(presence_gdf)
    print(f"[INFO] Presence points: {n_presence}")
    if n_presence < 15:
        raise ValueError(f"Only {n_presence} points. Minimum 15 required.")
    if n_presence >= 40:
        print(f"[INFO] n={n_presence} >= 40. Ensemble models will be enabled.")
        ENABLE_ENSEMBLE = True

    #  Load & annotate baseline rasters
    baseline_rasters = get_raster_list(SCENARIOS["Baseline_1970_2000"],
                                        static_folder=STATIC_RASTER_FOLDER, static_names=STATIC_VAR_NAMES)
    if not baseline_rasters:
        raise FileNotFoundError("No baseline rasters found.")
    print(f"[INFO] Baseline rasters: {len(baseline_rasters)}")
    print(f"[INFO] Baseline raster files: {[Path(p).name for p in baseline_rasters]}")
    
    n_static_expected = len(STATIC_VAR_NAMES) if STATIC_VAR_NAMES else 0
    if len(baseline_rasters) <= n_static_expected:
        raise RuntimeError(
            f"Only {len(baseline_rasters)} baseline raster(s) found, which is <= the "
            f"{n_static_expected} static topographic variable(s) alone. This means the "
            f"dynamic bioclim folder ({SCENARIOS['Baseline_1970_2000']}) contributed ZERO "
            f"rasters — almost certainly a wrong path. Check that this folder exists and "
            f"contains .asc/.tif files before proceeding; otherwise every future scenario "
            f"will silently come out identical to baseline."
        )

    with rio.open(baseline_rasters[0]) as _src:
        raster_crs = _src.crs
    if raster_crs is not None and str(raster_crs) != "EPSG:4326":
        presence_gdf = presence_gdf.to_crs(raster_crs)

    study_area_gdf = gpd.read_file(STUDY_AREA_SHP)
    if raster_crs is not None and study_area_gdf.crs != raster_crs:
        study_area_gdf = study_area_gdf.to_crs(raster_crs)

    clipped_path = OUTPUT_DIR / "_tmp_baseline_clipped.tif"
    with rio.open(baseline_rasters[0]) as src:
        clipped_data, clipped_transform = rio_mask(src, study_area_gdf.geometry, crop=True,
                                                   nodata=src.nodata if src.nodata is not None else -9999)
        clipped_meta = src.meta.copy()
        clipped_meta.update({"height": clipped_data.shape[1], "width": clipped_data.shape[2],
                              "transform": clipped_transform,
                              "nodata": src.nodata if src.nodata is not None else -9999})
    with rio.open(clipped_path, "w", **clipped_meta) as dst:
        dst.write(clipped_data)

    print("[INFO] Generating 10,000 background points...")
    background_gdf = ela.sample_raster(str(clipped_path), count=10_000)
    if os.path.exists(clipped_path):
        os.remove(clipped_path)

    merged_gdf = ela.stack_geodataframes(presence_gdf, background_gdf, add_class_label=True)
    annotated = ela.annotate(merged_gdf, baseline_rasters, drop_na=True, quiet=True)
    if len(annotated) == 0:
        raise RuntimeError("No points survived annotation.")

    X_raw = annotated.drop(columns=['class', 'geometry'])
    y = annotated['class']
    geometries = annotated['geometry']

    # Collinearity screening
    print(f"\n[INFO] Screening multicollinearity (|r| < {CORR_THRESHOLD})...")

    # correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_raw.corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=0.5, annot_kws={"size": 6})
    plt.title('Predictor Correlation (Pre-Removal)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_correlation_matrix_raw.png", dpi=300)
    plt.close()

    actual_cols_lower = {c.lower(): c for c in X_raw.columns}
    priority_vars_mapped = [actual_cols_lower.get(v.lower(), v) for v in
                            ['bio18', 'bio9', 'bio19', 'bio12', 'bio15', 'elevation', 'slope']
                            if v.lower() in actual_cols_lower]
    X_pearson, dropped_vars_pearson = remove_collinear_predictors(X_raw, threshold=CORR_THRESHOLD, priority_vars=priority_vars_mapped)
    print(f"[INFO] Retained {X_pearson.shape[1]} variables after Pearson screening: {list(X_pearson.columns)}")

    # post-Pearson-removal correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_pearson.corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=0.5)
    plt.title('Predictor Correlation (Post Pearson-Removal)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_correlation_matrix_clean.png", dpi=300)
    plt.close()

    
    print("[INFO] Computing VIF (pre-Pearson and post-Pearson)...")
    vif_before = compute_vif(X_raw)
    vif_before.to_csv(DIAGNOSTICS_DIR / "vif_before_screening.csv", index=False)
    vif_after_pearson = compute_vif(X_pearson)
    vif_after_pearson.to_csv(DIAGNOSTICS_DIR / "vif_after_pearson_screening.csv", index=False)
    max_vif_after_pearson = vif_after_pearson['VIF'].replace(np.inf, np.nan).max()
    print(f"[INFO] Max VIF among the {X_pearson.shape[1]} Pearson-survivors: {max_vif_after_pearson:.2f} "
          f"(values > 5-10 conventionally flag remaining concern)")

    VIF_THRESHOLD = 10.0
    if max_vif_after_pearson > VIF_THRESHOLD:
        print(f"[WARN] Max VIF ({max_vif_after_pearson:.2f}) exceeds threshold ({VIF_THRESHOLD}) -- "
              f"running iterative VIF-based reduction...")
        X, dropped_vars_vif, vif_iteration_history = iterative_vif_reduction(
            X_pearson, threshold=VIF_THRESHOLD, priority_vars=priority_vars_mapped)
        vif_iteration_history.to_csv(DIAGNOSTICS_DIR / "vif_iterative_reduction_history.csv", index=False)
        print(f"[INFO] VIF-based reduction dropped: {dropped_vars_vif}")
        print(f"[INFO] Final retained set ({X.shape[1]} variables): {list(X.columns)}")
    else:
        X = X_pearson
        dropped_vars_vif = []

    vif_after_final = compute_vif(X)
    vif_after_final.to_csv(DIAGNOSTICS_DIR / "vif_after_screening.csv", index=False)
    dropped_vars = dropped_vars_pearson + dropped_vars_vif

    COLUMN_TO_VARNAME = resolve_retained_raster_names(baseline_rasters, list(X.columns))
    RETAINED_VARIABLE_NAMES = list(COLUMN_TO_VARNAME.values())
    baseline_rasters_filtered = select_rasters_by_name(baseline_rasters, RETAINED_VARIABLE_NAMES)

    # collinearity screening + column mapping to CSV,
    
    # distinguishes Pearson-stage drops from VIF-stage drops.
    pd.DataFrame({'All_Variables': list(X_raw.columns),
                  'Retained': [c in X.columns for c in X_raw.columns],
                  'Dropped_By': ['Pearson' if c in dropped_vars_pearson else
                                 'VIF' if c in dropped_vars_vif else '' for c in X_raw.columns]}).to_csv(
        DIAGNOSTICS_DIR / "collinearity_report.csv", index=False)
    pd.DataFrame({'Column_Label': list(COLUMN_TO_VARNAME.keys()),
                  'Resolved_Variable': list(COLUMN_TO_VARNAME.values())}).to_csv(
        DIAGNOSTICS_DIR / "column_to_variable_mapping.csv", index=False)

    #  Spatial block CV + beta & feature tuning for MaxEnt
    print("\n[INFO] Spatial block CV + hyperparameter tuning (MaxEnt)...")
    bounds = geometries.total_bounds
    x_bins = np.linspace(bounds[0], bounds[2], N_BLOCKS_X + 1)
    y_bins = np.linspace(bounds[1], bounds[3], N_BLOCKS_Y + 1)
    annotated['x'] = geometries.x
    annotated['y'] = geometries.y
    annotated['block_x'] = np.digitize(annotated['x'], x_bins) - 1
    annotated['block_y'] = np.digitize(annotated['y'], y_bins) - 1
    annotated['spatial_fold'] = (annotated['block_x'] + annotated['block_y'] * N_BLOCKS_X) % 4
    folds = sorted(annotated['spatial_fold'].unique())

    tuning_results = []
    for beta in BETA_CANDIDATES:
        for feats in FEATURE_CANDIDATES:
            fold_aucs, fold_tss = [], []
            for fold in folds:
                train_mask = annotated['spatial_fold'] != fold
                test_mask = annotated['spatial_fold'] == fold
                X_tr, y_tr = X[train_mask], y[train_mask]
                X_te, y_te = X[test_mask], y[test_mask]
                if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                    continue
                m = ela.MaxentModel(transform='cloglog', beta_multiplier=beta, feature_types=feats)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                fold_aucs.append(roc_auc_score(y_te, preds))
                y_bin = (preds >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_te, y_bin, labels=[0, 1]).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                fold_tss.append(sens + spec - 1)
            if fold_aucs:
                tuning_results.append({'beta': beta, 'features': '+'.join(feats),
                                        'cv_auc_mean': np.mean(fold_aucs), 'cv_auc_std': np.std(fold_aucs),
                                        'cv_tss_mean': np.mean(fold_tss), 'n_folds': len(fold_aucs)})

    tuning_df = pd.DataFrame(tuning_results).sort_values('cv_auc_mean', ascending=False)
    tuning_df.to_csv(DIAGNOSTICS_DIR / "hyperparameter_tuning_results.csv", index=False)  # ADDED: was computed but never saved
    best_row = tuning_df.iloc[0]
    BEST_BETA = best_row['beta']
    BEST_FEATURES = best_row['features'].split('+')
    print(f"[INFO] Best MaxEnt: Beta={BEST_BETA}, Features={best_row['features']}, "
          f"CV AUC={best_row['cv_auc_mean']:.4f}")

    # Final MaxEnt model
    print("\n[INFO] Fitting final MaxEnt model...")
    final_maxent = ela.MaxentModel(transform='cloglog', beta_multiplier=BEST_BETA, feature_types=BEST_FEATURES)
    final_maxent.fit(X, y)
    y_pred_maxent = final_maxent.predict(X)
    auc_maxent_insample = roc_auc_score(y, y_pred_maxent)

    
    print("\n[INFO] Spatially-blocked CV hyperparameter tuning (GLM and RF)...")

    GLM_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
    RF_MAX_DEPTH_CANDIDATES = [5, 10, 20, None]
    RF_MIN_LEAF_CANDIDATES = [1, 3, 5]

    glm_tuning_results = []
    for C in GLM_C_CANDIDATES:
        fold_aucs = []
        for fold in folds:
            train_mask = annotated['spatial_fold'] != fold
            test_mask = annotated['spatial_fold'] == fold
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask], y[test_mask]
            if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                continue
            fold_scaler = StandardScaler().fit(X_tr.values)
            X_tr_s = fold_scaler.transform(X_tr.values)
            X_te_s = fold_scaler.transform(X_te.values)
            glm_cv = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
            glm_cv.fit(X_tr_s, y_tr)
            fold_aucs.append(roc_auc_score(y_te, glm_cv.predict_proba(X_te_s)[:, 1]))
        if fold_aucs:
            glm_tuning_results.append({'C': C, 'cv_auc_mean': np.mean(fold_aucs), 'cv_auc_std': np.std(fold_aucs)})
    glm_tuning_df = pd.DataFrame(glm_tuning_results).sort_values('cv_auc_mean', ascending=False)
    glm_tuning_df.to_csv(DIAGNOSTICS_DIR / "glm_hyperparameter_tuning_results.csv", index=False)
    BEST_GLM_C = glm_tuning_df.iloc[0]['C']
    auc_glm_cv_mean, auc_glm_cv_std = glm_tuning_df.iloc[0]['cv_auc_mean'], glm_tuning_df.iloc[0]['cv_auc_std']
    print(f"[INFO] Best GLM: C={BEST_GLM_C}, CV AUC={auc_glm_cv_mean:.4f}")

    rf_tuning_results = []
    for max_depth in RF_MAX_DEPTH_CANDIDATES:
        for min_leaf in RF_MIN_LEAF_CANDIDATES:
            fold_aucs = []
            for fold in folds:
                train_mask = annotated['spatial_fold'] != fold
                test_mask = annotated['spatial_fold'] == fold
                X_tr, y_tr = X[train_mask], y[train_mask]
                X_te, y_te = X[test_mask], y[test_mask]
                if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                    continue
                rf_cv = RandomForestClassifier(n_estimators=500, max_depth=max_depth, min_samples_leaf=min_leaf,
                                               random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
                rf_cv.fit(X_tr, y_tr)
                fold_aucs.append(roc_auc_score(y_te, rf_cv.predict_proba(X_te)[:, 1]))
            if fold_aucs:
                rf_tuning_results.append({'max_depth': max_depth, 'min_samples_leaf': min_leaf,
                                          'cv_auc_mean': np.mean(fold_aucs), 'cv_auc_std': np.std(fold_aucs)})
    rf_tuning_df = pd.DataFrame(rf_tuning_results).sort_values('cv_auc_mean', ascending=False)
    rf_tuning_df.to_csv(DIAGNOSTICS_DIR / "rf_hyperparameter_tuning_results.csv", index=False)
    BEST_RF_MAX_DEPTH = rf_tuning_df.iloc[0]['max_depth']
    
    if pd.isna(BEST_RF_MAX_DEPTH):
        BEST_RF_MAX_DEPTH = None
    else:
        BEST_RF_MAX_DEPTH = int(BEST_RF_MAX_DEPTH)
    BEST_RF_MIN_LEAF = int(rf_tuning_df.iloc[0]['min_samples_leaf'])
    auc_rf_cv_mean, auc_rf_cv_std = rf_tuning_df.iloc[0]['cv_auc_mean'], rf_tuning_df.iloc[0]['cv_auc_std']
    print(f"[INFO] Best RF: max_depth={BEST_RF_MAX_DEPTH}, min_samples_leaf={BEST_RF_MIN_LEAF}, "
          f"CV AUC={auc_rf_cv_mean:.4f}")

    print(f"[METRIC] MaxEnt CV AUC: {best_row['cv_auc_mean']:.4f}")
    print(f"[METRIC] GLM CV AUC:    {auc_glm_cv_mean:.4f}")
    print(f"[METRIC] RF CV AUC:     {auc_rf_cv_mean:.4f}")

    # ROC curve for the tuned MaxEnt model
    print("\n[INFO] Generating ROC curve (MaxEnt)...")
    fpr, tpr, _ = roc_curve(y, y_pred_maxent)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkblue', lw=2.5, label=f'MaxEnt (AUC = {auc_maxent_insample:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold'); plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_roc_curve.png", dpi=300)
    plt.close()

    # permutation importance for MaxEnt
    print("[INFO] Computing permutation importance (MaxEnt only)...")
    base_auc_for_perm = roc_auc_score(y, y_pred_maxent)
    perm_results = []
    rng = np.random.RandomState(RANDOM_STATE)
    for col in X.columns:
        auc_drops = []
        for _ in range(10):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            auc_drops.append(base_auc_for_perm - roc_auc_score(y, final_maxent.predict(X_perm)))
        perm_results.append({'Variable': col, 'Mean_Importance': np.mean(auc_drops), 'Std_Dev': np.std(auc_drops)})
    importance_df = pd.DataFrame(perm_results).sort_values('Mean_Importance', ascending=False)
    importance_df.to_csv(DIAGNOSTICS_DIR / "permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['Variable'][::-1], importance_df['Mean_Importance'][::-1],
             color='steelblue', edgecolor='black', alpha=0.85)
    plt.xlabel('Permutation Importance (AUC Drop)')
    plt.title('Variable Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_variable_importance.png", dpi=300)
    plt.close()

    # Variable importance & Jackknife for MaxEnt
    print("\n[INFO] Computing Jackknife analysis (MaxEnt only)...")
    jackknife = []
    for feat in X.columns:
        m_w = ela.MaxentModel(transform='cloglog', beta_multiplier=BEST_BETA, feature_types=BEST_FEATURES)
        m_w.fit(X.drop(columns=[feat]), y)
        auc_w = roc_auc_score(y, m_w.predict(X.drop(columns=[feat])))
        
        m_o = ela.MaxentModel(transform='cloglog', beta_multiplier=BEST_BETA, feature_types=BEST_FEATURES)
        m_o.fit(X[[feat]], y)
        auc_o = roc_auc_score(y, m_o.predict(X[[feat]]))
        
        jackknife.append({'Variable': feat, 'Without_Variable_AUC': auc_w, 'With_Only_Variable_AUC': auc_o})

    jackknife_df = pd.DataFrame(jackknife).sort_values('With_Only_Variable_AUC', ascending=False)
    jackknife_df.to_csv(DIAGNOSTICS_DIR / "jackknife_analysis.csv", index=False)

    plt.figure(figsize=(10, 6))
    x_indices = np.arange(len(jackknife_df))
    bar_width = 0.35
    plt.barh(x_indices - bar_width/2, jackknife_df['Without_Variable_AUC'], bar_width, label='Without Variable', color='skyblue')
    plt.barh(x_indices + bar_width/2, jackknife_df['With_Only_Variable_AUC'], bar_width, label='With Only Variable', color='salmon')
    plt.yticks(x_indices, jackknife_df['Variable'])
    plt.axvline(x=auc_maxent_insample, color='black', linestyle='--', label='Full Model AUC')
    plt.xlabel('AUC Score')
    plt.title('MaxEnt Jackknife Variable Importance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_jackknife_test.png", dpi=300)
    plt.close()

    # Response Curves (MaxEnt)
    print("\n[INFO] Generating response curves (MaxEnt only)...")
    n_cols_plot = 3
    n_rows_plot = int(np.ceil(len(X.columns) / n_cols_plot))
    fig, axes = plt.subplots(nrows=n_rows_plot, ncols=n_cols_plot, figsize=(15, 4 * n_rows_plot))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(X.columns):
        val_min, val_max = X[col].min(), X[col].max()
        val_seq = np.linspace(val_min, val_max, 100)
        synthetic_df = pd.DataFrame(np.tile(X.mean().values, (100, 1)), columns=X.columns)
        synthetic_df[col] = val_seq

        preds = final_maxent.predict(synthetic_df)

        axes[idx].plot(val_seq, preds, color='darkgreen', linewidth=2)
        axes[idx].set_title(f'Response Curve: {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Cloglog Probability')
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, linestyle='--', alpha=0.6)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_response_curves.png", dpi=300)
    plt.close()

    # Fit Full Models & Compute Fair CV Ensemble Weights
    print("\n[INFO] Fitting full GLM and RF models...")
    full_scaler = StandardScaler().fit(X.values)
    X_scaled = full_scaler.transform(X.values)

    
    final_glm_model = LogisticRegression(C=BEST_GLM_C, max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    final_glm_model.fit(X_scaled, y)
    final_glm = _RasterProbaWrapper(final_glm_model, X.columns, scaler=full_scaler)

    final_rf_model = RandomForestClassifier(n_estimators=500, max_depth=BEST_RF_MAX_DEPTH, min_samples_leaf=BEST_RF_MIN_LEAF,
                                             random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    final_rf_model.fit(X, y)
    final_rf = _RasterProbaWrapper(final_rf_model, X.columns)

    cv_scores = {
        'maxent': max(best_row['cv_auc_mean'], 0.5),
        'glm': max(auc_glm_cv_mean, 0.5),
        'rf': max(auc_rf_cv_mean, 0.5)
    }
    total_cv_score = sum(cv_scores.values())
    weights = {k: v / total_cv_score for k, v in cv_scores.items()}

    print("[INFO] Fair CV Ensemble Weights:")
    for m_name, w_val in weights.items():
        print(f"       * {m_name.upper()}: {w_val:.4f} (CV AUC = {cv_scores[m_name]:.4f})")

    # Calculate Ensemble Thresholds on In-Sample Ensemble Predictions
    p_maxent_in = y_pred_maxent
    p_glm_in = final_glm.predict(X)
    p_rf_in = final_rf.predict(X)

    y_pred_ensemble = (weights['maxent'] * p_maxent_in +
                       weights['glm'] * p_glm_in +
                       weights['rf'] * p_rf_in)

    thresholds = compute_thresholds(y, y_pred_ensemble)
    print("\n[INFO] Calculated Ensemble Thresholds:")
    for th_name, th_val in thresholds.items():
        print(f"       * {th_name}: {th_val:.4f}")

    pd.DataFrame([thresholds]).to_csv(DIAGNOSTICS_DIR / "ensemble_thresholds.csv", index=False)

    
    auc_ens_insample = roc_auc_score(y, y_pred_ensemble)
    ensemble_weights_df = pd.DataFrame({
        'Model': ['MaxEnt', 'GLM', 'RF', 'Ensemble'],
        'CV_AUC': [best_row['cv_auc_mean'], auc_glm_cv_mean, auc_rf_cv_mean, np.nan],
        'InSample_AUC': [roc_auc_score(y, p_maxent_in), roc_auc_score(y, p_glm_in),
                         roc_auc_score(y, p_rf_in), auc_ens_insample],
        'Weight_used_for_projection': [weights['maxent'], weights['glm'], weights['rf'], 1.0],
    })
    ensemble_weights_df.to_csv(DIAGNOSTICS_DIR / "ensemble_weights.csv", index=False)
    print(f"[INFO] Ensemble in-sample AUC: {auc_ens_insample:.4f}")

    # Continuous Boyce Index for the ensemble (complements AUC, which is
    # known to have limitations for presence-background models).
    print("[INFO] Computing continuous Boyce index...")
    boyce = compute_boyce_index(y_pred_ensemble, y_pred_ensemble[y.values == 1])
    print(f"[METRIC] Boyce index (ensemble, in-sample): {boyce:.4f}")
    pd.DataFrame([{'Metric': 'Boyce_Index_Ensemble_InSample', 'Value': boyce}]).to_csv(
        DIAGNOSTICS_DIR / "boyce_index.csv", index=False)

    # ADDED: Moran's I on ensemble residuals - checks whether spatial block CV
    # actually removed spatial structure from what's left unexplained.
    print("[INFO] Computing Moran's I on ensemble residuals...")
    residuals = y.values.astype(float) - y_pred_ensemble
    morans_i, morans_p = compute_morans_i(residuals, annotated['x'].values, annotated['y'].values)
    print(f"[METRIC] Moran's I on residuals: {morans_i:.4f} (p = {morans_p:.4f}; "
          f"p < 0.05 with positive I suggests remaining spatial autocorrelation)")
    pd.DataFrame([{'Morans_I': morans_i, 'p_value': morans_p}]).to_csv(
        DIAGNOSTICS_DIR / "morans_i_residuals.csv", index=False)

   
    print("\n[INFO] Nested held-out validation (one fold fully excluded from fitting)...")
    outer_fold = max(folds, key=lambda f: (annotated['spatial_fold'] == f).sum())
    inner_mask = annotated['spatial_fold'] != outer_fold
    outer_mask = annotated['spatial_fold'] == outer_fold
    X_inner, y_inner = X[inner_mask], y[inner_mask]
    X_outer, y_outer = X[outer_mask], y[outer_mask]

    if len(np.unique(y_outer)) < 2 or len(np.unique(y_inner)) < 2:
        print("[WARN] Held-out fold lacks both classes; skipping nested validation.")
    else:
        m_nested = ela.MaxentModel(transform='cloglog', beta_multiplier=BEST_BETA, feature_types=BEST_FEATURES)
        m_nested.fit(X_inner, y_inner)
        p_maxent_outer = m_nested.predict(X_outer)

        scaler_nested = StandardScaler().fit(X_inner.values)
        glm_nested = LogisticRegression(C=BEST_GLM_C, max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
        glm_nested.fit(scaler_nested.transform(X_inner.values), y_inner)
        p_glm_outer = glm_nested.predict_proba(scaler_nested.transform(X_outer.values))[:, 1]

        rf_nested = RandomForestClassifier(n_estimators=500, max_depth=BEST_RF_MAX_DEPTH,
                                           min_samples_leaf=BEST_RF_MIN_LEAF, random_state=RANDOM_STATE,
                                           class_weight='balanced', n_jobs=-1)
        rf_nested.fit(X_inner, y_inner)
        p_rf_outer = rf_nested.predict_proba(X_outer)[:, 1]

        p_ensemble_outer = (weights['maxent'] * p_maxent_outer +
                            weights['glm'] * p_glm_outer +
                            weights['rf'] * p_rf_outer)

        nested_results = pd.DataFrame({
            'Model': ['MaxEnt', 'GLM', 'RF', 'Ensemble'],
            'Nested_HeldOut_AUC': [
                roc_auc_score(y_outer, p_maxent_outer),
                roc_auc_score(y_outer, p_glm_outer),
                roc_auc_score(y_outer, p_rf_outer),
                roc_auc_score(y_outer, p_ensemble_outer),
            ]
        })
        nested_results.to_csv(DIAGNOSTICS_DIR / "nested_holdout_validation.csv", index=False)
        print(nested_results.to_string(index=False))
        print(f"[INFO] Compare these to the main CV AUCs above -- a large drop here would "
              f"suggest the main spatial-block CV is still somewhat optimistic.")

    # Spatial Projection & Ensembling
    print("\n[INFO] Starting spatial projections across scenarios...")
    reference_df = X[y == 1]
    MASTER_META = None  

    for scenario_name, scenario_folder in SCENARIOS.items():
        print(f"\n[PROJECTION] Scenario: {scenario_name}")
        scenario_folder = Path(scenario_folder)

        all_scen_rasters = get_raster_list(scenario_folder, static_folder=STATIC_RASTER_FOLDER, static_names=STATIC_VAR_NAMES)
        if len(all_scen_rasters) <= n_static_expected:
            raise RuntimeError(
                f"Only {len(all_scen_rasters)} raster(s) found for scenario '{scenario_name}' "
                f"(folder: {scenario_folder}), which is <= the {n_static_expected} static "
                f"topographic variables alone. The bioclim folder for this scenario "
                f"contributed ZERO rasters — check this specific path before continuing."
            )
        scen_rasters_filtered = select_rasters_by_name(all_scen_rasters, RETAINED_VARIABLE_NAMES)

        maxent_out = CONTINUOUS_DIR / f"{scenario_name}_maxent_only.tif"
        glm_out = DIAGNOSTICS_DIR / f"_tmp_{scenario_name}_glm.tif"
        rf_out = DIAGNOSTICS_DIR / f"_tmp_{scenario_name}_rf.tif"
        ensemble_out = CONTINUOUS_DIR / f"{scenario_name}_ensemble.tif"
        mess_out = MESS_DIR / f"{scenario_name}_mess.tif"
        masked_ensemble_out = CONTINUOUS_DIR / f"{scenario_name}_ensemble_mess_masked.tif"

        # Apply each model
        ela.apply_model_to_rasters(final_maxent, scen_rasters_filtered, maxent_out, quiet=True)
        ela.apply_model_to_rasters(final_glm, scen_rasters_filtered, glm_out, quiet=True)
        ela.apply_model_to_rasters(final_rf, scen_rasters_filtered, rf_out, quiet=True)

        # Weighted combination
        with rio.open(maxent_out) as m_src, rio.open(glm_out) as g_src, rio.open(rf_out) as r_src:
            meta = m_src.meta.copy()
            m_arr = m_src.read(1)
            g_arr = g_src.read(1)
            r_arr = r_src.read(1)

            nodata_val = m_src.nodata if m_src.nodata is not None else -9999.0
            valid_mask = ((m_arr != nodata_val) & (g_arr != nodata_val) & (r_arr != nodata_val) &
                          ~np.isnan(m_arr) & ~np.isnan(g_arr) & ~np.isnan(r_arr))

            ens_arr = np.full(m_arr.shape, nodata_val, dtype=np.float32)
            ens_arr[valid_mask] = (
                weights['maxent'] * m_arr[valid_mask] +
                weights['glm'] * g_arr[valid_mask] +
                weights['rf'] * r_arr[valid_mask]
            )

            meta.update(dtype=rio.float32, nodata=nodata_val)

        
        if MASTER_META is None:
            MASTER_META = meta.copy()
        else:
            same_grid = (
                meta['height'] == MASTER_META['height'] and
                meta['width'] == MASTER_META['width'] and
                meta['transform'] == MASTER_META['transform'] and
                meta['crs'] == MASTER_META['crs']
            )
            if not same_grid:
                print(f"       [ALIGN] {scenario_name} grid {(meta['height'], meta['width'])} "
                      f"!= baseline grid {(MASTER_META['height'], MASTER_META['width'])} "
                      f"-> reprojecting onto baseline grid.")
                ens_arr = _align_array_to_grid(ens_arr, meta, MASTER_META,
                                                resampling=Resampling.bilinear, fill_value=nodata_val)
                meta = MASTER_META.copy()
                meta.update(dtype=rio.float32, nodata=nodata_val)

        with rio.open(ensemble_out, 'w', **meta) as dst:
            dst.write(ens_arr.astype(np.float32), 1)

        # Remove temporary intermediate files
        for tmp_file in [glm_out, rf_out]:
            if tmp_file.exists():
                os.remove(tmp_file)

        # Compute MESS
        compute_raster_mess(reference_df, scen_rasters_filtered, mess_out, meta)

        # Mask Ensemble with MESS
        with rio.open(ensemble_out) as ens_src, rio.open(mess_out) as mess_src:
            ens_data = ens_src.read(1)
            mess_data = mess_src.read(1)

            masked_data = ens_data.copy()
            masked_data[(mess_data < MESS_NOVELTY_THRESHOLD) & (mess_data != mess_src.nodata)] = nodata_val

            with rio.open(masked_ensemble_out, 'w', **meta) as dst:
                dst.write(masked_data.astype(np.float32), 1)

       
        binary_mask = (masked_data != nodata_val) & ~np.isnan(masked_data)
        for th_name, th_val in thresholds.items():
            if th_name.endswith('_sens') or th_name.endswith('_spec'):
                continue  # these two keys are diagnostic scalars, not thresholds
            binary_arr = np.zeros(ens_arr.shape, dtype=np.uint8)
            binary_arr[binary_mask] = (masked_data[binary_mask] >= th_val).astype(np.uint8)
            binary_arr[~binary_mask] = 255

            bin_meta = meta.copy()
            bin_meta.update(dtype=rio.uint8, nodata=255)
            binary_out_th = BINARY_DIR / f"{scenario_name}_binary_{th_name}.tif"
            with rio.open(binary_out_th, 'w', **bin_meta) as dst:
                dst.write(binary_arr, 1)

        print(f"       -> Complete for {scenario_name}")

   
    print("\n[INFO] Performing Range-Shift and Centroid Dynamics Analysis...")
    PRIMARY_THRESHOLD = 'P10'
    range_summary = []
    centroid_summary = []

    baseline_bin_path = BINARY_DIR / f"Baseline_1970_2000_binary_{PRIMARY_THRESHOLD}.tif"
    with rio.open(baseline_bin_path) as b_src:
        base_bin = b_src.read(1)
        transform = b_src.transform
        nodata_bin = b_src.nodata

    base_suitable = (base_bin == 1)
    base_suitable_count = np.sum(base_suitable)

    y_coords, x_coords = np.where(base_suitable)
    if len(x_coords) > 0:
        base_x_center, base_y_center = rio.transform.xy(transform, y_coords, x_coords)
        base_cx_mean, base_cy_mean = np.mean(base_x_center), np.mean(base_y_center)
    else:
        base_cx_mean, base_cy_mean = np.nan, np.nan

    centroid_summary.append({
        'Scenario': 'Baseline_1970_2000',
        'Centroid_X': base_cx_mean,
        'Centroid_Y': base_cy_mean,
        'Shift_Distance': 0.0
    })

    for scenario_name in SCENARIOS.keys():
        scen_bin_path = BINARY_DIR / f"{scenario_name}_binary_{PRIMARY_THRESHOLD}.tif"
        with rio.open(scen_bin_path) as s_src:
            scen_bin = s_src.read(1)

        if scen_bin.shape != base_bin.shape:
            raise ValueError(
                f"Grid shape mismatch: baseline binary map is {base_bin.shape} but "
                f"{scenario_name} binary map is {scen_bin.shape}. This pipeline does not "
                f"reproject/align scenario outputs to the baseline grid before comparing "
                f"them pixel-by-pixel, so retained/lost/gained calculations would be "
                f"silently wrong (comparing unrelated pixels) rather than erroring out on "
                f"their own. If your bioclim source rasters for different scenarios have "
                f"different extents/resolutions, add an explicit reprojection step (see "
                f"align_raster_to_reference) before this comparison."
            )

        scen_suitable = (scen_bin == 1)
        scen_suitable_count = np.sum(scen_suitable)

        stable = base_suitable & scen_suitable
        expansion = (~base_suitable) & scen_suitable & (scen_bin != nodata_bin)
        contraction = base_suitable & (~scen_suitable)

        net_change = scen_suitable_count - base_suitable_count
        pct_change = (net_change / base_suitable_count * 100) if base_suitable_count > 0 else 0.0

        range_summary.append({
            'Scenario': scenario_name,
            'Baseline_Suitable_Cells': base_suitable_count,
            'Future_Suitable_Cells': scen_suitable_count,
            'Stable_Cells': np.sum(stable),
            'Expansion_Cells': np.sum(expansion),
            'Contraction_Cells': np.sum(contraction),
            'Net_Change_Cells': net_change,
            'Percentage_Change (%)': pct_change
        })

        sy_coords, sx_coords = np.where(scen_suitable)
        if len(sx_coords) > 0:
            scen_x_center, scen_y_center = rio.transform.xy(transform, sy_coords, sx_coords)
            scen_cx_mean, scen_cy_mean = np.mean(scen_x_center), np.mean(scen_y_center)
            dist_shift = np.sqrt((scen_cx_mean - base_cx_mean)**2 + (scen_cy_mean - base_cy_mean)**2)
        else:
            scen_cx_mean, scen_cy_mean = np.nan, np.nan
            dist_shift = np.nan

        if scenario_name != 'Baseline_1970_2000':
            centroid_summary.append({
                'Scenario': scenario_name,
                'Centroid_X': scen_cx_mean,
                'Centroid_Y': scen_cy_mean,
                'Shift_Distance': dist_shift
            })

    pd.DataFrame(range_summary).to_csv(DIAGNOSTICS_DIR / "range_shift_summary.csv", index=False)
    pd.DataFrame(centroid_summary).to_csv(DIAGNOSTICS_DIR / "centroid_shift_summary.csv", index=False)

   
    RUN_BOOTSTRAP_UNCERTAINTY = True
    N_BOOTSTRAP = 15
    BOOTSTRAP_SCENARIOS = ['SSP45_2081_2100', 'SSP85_2021_2040']  # the two headline results

    if RUN_BOOTSTRAP_UNCERTAINTY:
        print(f"\n[INFO] Bootstrap uncertainty ({N_BOOTSTRAP} resamples) on "
              f"{BOOTSTRAP_SCENARIOS}...")
        rng_boot = np.random.RandomState(RANDOM_STATE)
        presence_idx = np.where(y.values == 1)[0]
        background_idx = np.where(y.values == 0)[0]
        p10_threshold_fixed = thresholds['P10']

        with rio.open(BINARY_DIR / f"Baseline_1970_2000_binary_{PRIMARY_THRESHOLD}.tif") as bsrc:
            base_bin_boot = bsrc.read(1)
        base_area_cells_fixed = np.sum(base_bin_boot == 1)

        bootstrap_rows = []
        for b in range(N_BOOTSTRAP):
            boot_presence_idx = rng_boot.choice(presence_idx, size=len(presence_idx), replace=True)
            boot_idx = np.concatenate([boot_presence_idx, background_idx])
            X_boot, y_boot = X.iloc[boot_idx].reset_index(drop=True), y.iloc[boot_idx].reset_index(drop=True)

            try:
                m_boot = ela.MaxentModel(transform='cloglog', beta_multiplier=BEST_BETA, feature_types=BEST_FEATURES)
                m_boot.fit(X_boot, y_boot)

                scaler_boot = StandardScaler().fit(X_boot.values)
                glm_boot = LogisticRegression(C=BEST_GLM_C, max_iter=1000, random_state=b, class_weight='balanced')
                glm_boot.fit(scaler_boot.transform(X_boot.values), y_boot)
                glm_boot_wrapped = _RasterProbaWrapper(glm_boot, X.columns, scaler=scaler_boot)

                rf_boot = RandomForestClassifier(n_estimators=500, max_depth=BEST_RF_MAX_DEPTH,
                                                 min_samples_leaf=BEST_RF_MIN_LEAF, random_state=b,
                                                 class_weight='balanced', n_jobs=-1)
                rf_boot.fit(X_boot, y_boot)
                rf_boot_wrapped = _RasterProbaWrapper(rf_boot, X.columns)

                for scen_name in BOOTSTRAP_SCENARIOS:
                    scen_folder = Path(SCENARIOS[scen_name])
                    scen_rasters_all_b = get_raster_list(scen_folder, static_folder=STATIC_RASTER_FOLDER,
                                                          static_names=STATIC_VAR_NAMES)
                    scen_rasters_b = select_rasters_by_name(scen_rasters_all_b, RETAINED_VARIABLE_NAMES)

                    mx_tmp = DIAGNOSTICS_DIR / f"_tmp_boot{b}_{scen_name}_mx.tif"
                    gl_tmp = DIAGNOSTICS_DIR / f"_tmp_boot{b}_{scen_name}_gl.tif"
                    rf_tmp = DIAGNOSTICS_DIR / f"_tmp_boot{b}_{scen_name}_rf.tif"
                    ela.apply_model_to_rasters(m_boot, scen_rasters_b, mx_tmp, quiet=True)
                    ela.apply_model_to_rasters(glm_boot_wrapped, scen_rasters_b, gl_tmp, quiet=True)
                    ela.apply_model_to_rasters(rf_boot_wrapped, scen_rasters_b, rf_tmp, quiet=True)

                    with rio.open(mx_tmp) as ms, rio.open(gl_tmp) as gs, rio.open(rf_tmp) as rs:
                        b_meta = ms.meta.copy()
                        m_a, g_a, r_a = ms.read(1), gs.read(1), rs.read(1)
                        nod = ms.nodata if ms.nodata is not None else -9999.0
                        valid = (m_a != nod) & (g_a != nod) & (r_a != nod)
                        ens_a = weights['maxent'] * m_a + weights['glm'] * g_a + weights['rf'] * r_a

                    if b_meta['height'] != MASTER_META['height'] or b_meta['width'] != MASTER_META['width']:
                        ens_a = _align_array_to_grid(ens_a, b_meta, MASTER_META, fill_value=nod)
                        valid = _align_array_to_grid(valid.astype(np.float32), b_meta, MASTER_META,
                                                      resampling=Resampling.nearest, fill_value=0) > 0.5

                    suitable_cells = np.sum(valid & (ens_a >= p10_threshold_fixed))
                    pct_change = ((suitable_cells - base_area_cells_fixed) / base_area_cells_fixed * 100
                                  if base_area_cells_fixed > 0 else np.nan)
                    bootstrap_rows.append({'Bootstrap_Iter': b, 'Scenario': scen_name,
                                           'Suitable_Cells': int(suitable_cells), 'Percentage_Change': pct_change})

                    for tmp_f in [mx_tmp, gl_tmp, rf_tmp]:
                        if tmp_f.exists():
                            os.remove(tmp_f)
            except Exception as e:
                print(f"       [WARN] Bootstrap iteration {b} failed: {e}. Skipping.")
                continue

        if bootstrap_rows:
            boot_df = pd.DataFrame(bootstrap_rows)
            boot_df.to_csv(DIAGNOSTICS_DIR / "bootstrap_range_shift_raw.csv", index=False)
            summary_rows = []
            for scen_name in BOOTSTRAP_SCENARIOS:
                vals = boot_df.loc[boot_df['Scenario'] == scen_name, 'Percentage_Change'].dropna().values
                if len(vals) > 0:
                    summary_rows.append({
                        'Scenario': scen_name, 'N_successful_bootstraps': len(vals),
                        'Median_Pct_Change': np.median(vals),
                        'CI_2.5': np.percentile(vals, 2.5), 'CI_97.5': np.percentile(vals, 97.5),
                    })
            boot_summary_df = pd.DataFrame(summary_rows)
            boot_summary_df.to_csv(DIAGNOSTICS_DIR / "bootstrap_range_shift_summary.csv", index=False)
            print("[RESULT] Bootstrap uncertainty summary:")
            print(boot_summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  SDM PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
