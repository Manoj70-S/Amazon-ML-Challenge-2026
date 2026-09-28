"""
train_matcher.py  (v2 – fast lazy normalization)
=================================================
Key fix: run blocking FIRST, then normalize ONLY the ~500k candidate records
referenced in candidate pairs — not all 13M records.

Steps:
  1. Load training data.
  2. Run TF-IDF blocking on TRAIN split -> candidate_dict.
  3. Collect unique referenced S1 + candidate IDs from blocking.
  4. Normalize ONLY those records (fast: ~500k instead of 13M).
  5. Extract 28 features for every candidate pair.
  6. Label pairs using ground truth.
  7. Train LightGBM (class-weighted, early stopping).
  8. Tune threshold on VAL split -> maximise macro F₀.₅.
  9. Save model + threshold.

Usage:
    python code/business_entity_resolution/src/train_matcher.py
"""

# Force stdout flush on every print so log files update immediately
import builtins as _builtins
_orig_print = _builtins.print
def print(*args, **kwargs):  # noqa: A001
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)
_builtins.print = print

import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional

# ── ensure local imports work regardless of CWD ──────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address
from blocking import TFIDFBlocker, export_candidate_pairs_tsv
from features import FEATURE_COLUMNS, extract_features_batch

import lightgbm as lgb

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_ROOT, "dataset", "train")
SPLIT_DIR  = os.path.join(PROJECT_ROOT, "data_split")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR  = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BLOCKER_PARAMS = dict(
    analyzer='word',          # word bigrams: fast top-N term retrieval
    ngram_range=(1, 2),       # unigrams + bigrams
    min_sim=0.05,             # minimum score threshold (effectively 1/N_TERMS match)
    top_k=50,                 # 2x candidates -> better blocking recall
    max_features=200_000,
    min_df=3,
    max_df=0.5,
    progress_every=5_000,
)

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=2000,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

MAX_TRAIN_S1 = 120_000   # training entities for blocking
MAX_VAL_S1   = 30_000    # val entities for threshold tuning


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading training data …")
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    train_ids = pd.read_csv(os.path.join(SPLIT_DIR, "train_s1_ids.tsv"), sep="\t", dtype=str)["entity_id"].tolist()
    val_ids   = pd.read_csv(os.path.join(SPLIT_DIR, "val_s1_ids.tsv"),   sep="\t", dtype=str)["entity_id"].tolist()
    print(f"  Loaded in {time.time()-t0:.1f}s | S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    return s1, s2, s3, gt, train_ids, val_ids


def build_gt_lookup(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build GT lookup: s1_entity_id -> set of matching S2/S3 entity_ids.
    GT column 'matched_entity_ids' contains comma-separated S2+S3 IDs.
    Uses numpy array access instead of iterrows for speed (~8s for 2.2M rows).
    """
    s1_ids = gt["source1_entity_id"].values
    raws   = gt["matched_entity_ids"].values
    lookup: Dict[str, Set[str]] = {}
    for i in range(len(s1_ids)):
        raw = raws[i]
        if raw and raw != "nan" and raw != "":
            lookup[s1_ids[i]] = set(str(raw).split(","))
        else:
            lookup[s1_ids[i]] = set()   # singleton
    return lookup


def normalize_subset(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Normalize a SMALL subset of records (those referenced by candidate pairs).
    Per-row normalization is fine here because subset is ~500k, not 13M.
    """
    names = df["business_name"].map(normalize_name)
    addrs = df["business_address"].map(normalize_address)
    return dict(zip(df["entity_id"], zip(names, addrs)))


def build_lazy_norm_dicts(
    candidate_dict: Dict[str, List[Tuple[str, float]]],
    s1: pd.DataFrame,
    cand_df: pd.DataFrame,   # combined S2+S3
) -> Tuple[Dict, Dict]:
    """
    KEY OPTIMIZATION: only normalize the S1 entities and candidate records
    that actually appear in candidate pairs — not all 13M records.
    """
    t0 = time.time()
    # Collect referenced IDs
    ref_s1_ids:   Set[str] = set(candidate_dict.keys())
    ref_cand_ids: Set[str] = set()
    for cands in candidate_dict.values():
        for cid, _ in cands:
            ref_cand_ids.add(cid)

    print(f"  Referenced: {len(ref_s1_ids):,} S1 ids, {len(ref_cand_ids):,} cand ids")

    # Filter to referenced subsets
    s1_sub   = s1[s1["entity_id"].isin(ref_s1_ids)]
    cand_sub = cand_df[cand_df["entity_id"].isin(ref_cand_ids)]

    # Normalize only referenced records
    s1_dict   = normalize_subset(s1_sub)
    cand_dict = normalize_subset(cand_sub)
    print(f"  Lazy normalization done in {time.time()-t0:.1f}s")
    return s1_dict, cand_dict


def build_and_label(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    cand_df: pd.DataFrame,
    gt_lookup: Dict[str, Set[str]],
    s1_ids_subset: List[str],
    blocker: TFIDFBlocker,
    split_name: str,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict, List[str], List[str]]:
    s1_sub = s1[s1["entity_id"].isin(set(s1_ids_subset))].copy()
    print(f"\n[{split_name}] Blocking {len(s1_sub):,} S1 entities …")

    candidate_dict = blocker.generate_candidates(s1_sub, s2, s3, verbose=True, cache_dir=cache_dir)

    # Lazy normalization — only referenced records
    print(f"[{split_name}] Building normalised dicts (lazy) …")
    s1_dict, c_dict = build_lazy_norm_dicts(candidate_dict, s1, cand_df)

    # Feature extraction
    print(f"[{split_name}] Extracting features …")
    t0 = time.time()
    s1_ids_out, cand_ids_out, X = extract_features_batch(candidate_dict, s1_dict, c_dict)
    print(f"  Features: {len(s1_ids_out):,} pairs in {time.time()-t0:.1f}s")

    # Label
    y = np.array([
        1 if cid in gt_lookup.get(sid, set()) else 0
        for sid, cid in zip(s1_ids_out, cand_ids_out)
    ], dtype=np.int8)

    pos = int(y.sum())
    print(f"  Labels: {pos:,} positives / {len(y)-pos:,} negatives (ratio 1:{(len(y)-pos)//max(1,pos):.0f})")
    return X, y, candidate_dict, s1_ids_out, cand_ids_out


# ─────────────────────────────────────────────────────────────────────────────
# Macro F₀.₅ threshold evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_macro_f05(
    candidate_dict: Dict,
    gt_lookup: Dict,
    s1_ids_out: List[str],
    cand_ids_out: List[str],
    proba: np.ndarray,
    threshold: float,
) -> float:
    pair_preds: Dict[str, Dict[str, int]] = {}
    for sid, cid, p in zip(s1_ids_out, cand_ids_out, proba):
        if sid not in pair_preds:
            pair_preds[sid] = {}
        pair_preds[sid][cid] = int(p >= threshold)

    scores = []
    beta = 0.5
    for sid, preds in pair_preds.items():
        gt_set   = gt_lookup.get(sid, set())
        pred_pos = {c for c, lbl in preds.items() if lbl == 1}
        if not gt_set:
            scores.append(1.0 if not pred_pos else 0.0)
        else:
            tp = len(pred_pos & gt_set)
            fp = len(pred_pos - gt_set)
            fn = len(gt_set - pred_pos)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = (beta**2) * prec + rec
            scores.append(((1 + beta**2) * prec * rec / denom) if denom > 0 else 0.0)

    # Entities with no candidates at all
    for sid in set(candidate_dict.keys()) - set(pair_preds.keys()):
        gt_set = gt_lookup.get(sid, set())
        scores.append(1.0 if not gt_set else 0.0)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    s1, s2, s3, gt, train_ids, val_ids = load_data()

    # Combined candidate DataFrame (used for lazy normalization lookups)
    print("Concatenating S2+S3 for candidate lookup …")
    cand_df = pd.concat([s2, s3], ignore_index=True)
    print(f"  Candidate pool: {len(cand_df):,} records")

    gt_lookup = build_gt_lookup(gt)
    print(f"  GT lookup built: {len(gt_lookup):,} entries")

    # Subsample training split
    rng = np.random.default_rng(42)
    if len(train_ids) > MAX_TRAIN_S1:
        train_ids = rng.choice(train_ids, MAX_TRAIN_S1, replace=False).tolist()
        print(f"  Subsampled train to {len(train_ids):,}")
    if len(val_ids) > MAX_VAL_S1:
        val_ids = val_ids[:MAX_VAL_S1]
        print(f"  Capped val to {len(val_ids):,}")

    blocker = TFIDFBlocker(**BLOCKER_PARAMS)

    # ── Build TRAIN dataset (builds + caches TF-IDF index per country) ────────
    X_train, y_train, _, _, _ = build_and_label(
        s1, s2, s3, cand_df, gt_lookup, train_ids, blocker, "TRAIN",
        cache_dir=CACHE_DIR,
    )

    # ── Build VAL dataset (loads cached TF-IDF index -> fast!) ─────────────────
    X_val, y_val, val_cand_dict, val_s1_ids, val_cand_ids = build_and_label(
        s1, s2, s3, cand_df, gt_lookup, val_ids, blocker, "VAL",
        cache_dir=CACHE_DIR,
    )

    # ── Train LightGBM ────────────────────────────────────────────────────────
    print("\n=== Training LightGBM ===")
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    spw = neg_count / max(1, pos_count)
    print(f"  Pairs: {len(y_train):,}  |  pos={pos_count:,}  neg={neg_count:,}  scale_pos_weight={spw:.1f}")

    params = dict(LGB_PARAMS, scale_pos_weight=spw)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # Feature importances
    imps = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1])
    print("  Top-10 features:")
    for fname, imp in imps[:10]:
        print(f"    {fname}: {imp:.1f}")

    # ── Threshold tuning ───────────────────────────────────────────────────────
    print("\n=== Threshold Tuning (macro F₀.₅) ===")
    val_proba = model.predict_proba(X_val)[:, 1]
    best_thr, best_f05 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.025):
        f05 = compute_macro_f05(val_cand_dict, gt_lookup, val_s1_ids, val_cand_ids, val_proba, thr)
        print(f"  thr={thr:.3f}  F₀.₅={f05:.5f}")
        if f05 > best_f05:
            best_f05, best_thr = f05, float(thr)

    print(f"\n  [OK] Best threshold: {best_thr:.3f}  ->  macro-F₀.₅ = {best_f05:.5f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "lgbm_matcher.pkl")
    meta_path  = os.path.join(MODEL_DIR, "model_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "threshold": best_thr,
        "best_val_f05": best_f05,
        "feature_columns": FEATURE_COLUMNS,
        "blocker_params": BLOCKER_PARAMS,
        "lgb_n_estimators_best": int(model.best_iteration_),
        "n_train_pairs": int(len(y_train)),
        "n_positives": pos_count,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model  -> {model_path}")
    print(f"  Meta   -> {meta_path}")
    print(f"\n=== Total time: {(time.time()-t_total)/60:.1f} min ===")
    print("\nNext step:")
    print("  python code/business_entity_resolution/src/generate_outputs.py")


if __name__ == "__main__":
    main()

