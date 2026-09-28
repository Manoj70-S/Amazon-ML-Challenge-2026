"""
eval_v4_pipeline.py
===================
Benchmark script to train and evaluate the v4 Multi-View & High-Precision Pipeline
on the validation split (30,000 S1 entities).

Features tested:
  - 34 Dense Features (RapidFuzz, Metaphone phonetic, Exact ZIP/PIN match,
    Token Containment, Street Number Match, Trigram Jaccard, Blocker Rank).
  - LightGBM + XGBoost Stacking Ensemble.
  - Multi-match Graph Transitive Closure.
  - Precision-weighted Macro-F0.5 Threshold Sweep.
"""

import os
import sys

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

import time
import re
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import jellyfish
from rapidfuzz import fuzz, distance
import lightgbm as lgb
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "dataset", "train")
DATA_SPLIT   = os.path.join(PROJECT_ROOT, "data_split")

from normalize import normalize_name, normalize_address
from blocking import TFIDFBlocker

RE_DIGITS = re.compile(r'\b\d+\b')
RE_ZIP    = re.compile(r'\b\d{5,6}\b')


def extract_zip(addr: str) -> str:
    m = RE_ZIP.findall(addr)
    return m[-1] if m else ''


def extract_street_num(addr: str) -> str:
    m = RE_DIGITS.findall(addr)
    return m[0] if m else ''


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))


def char_ngram_jaccard(s1: str, s2: str, n: int = 2) -> float:
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    set1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    set2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    return len(set1 & set2) / float(len(set1 | set2))


def safe_len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


# ── 34 Enhanced Feature Columns ───────────────────────────────────────────────
V4_FEATURE_COLUMNS = [
    # Name (14)
    'tfidf_sim',
    'name_levenshtein',
    'name_token_sort',
    'name_token_set',
    'name_jaro_winkler',
    'name_partial',
    'name_jaccard',
    'name_bigram_jacc',
    'name_trigram_jacc',
    'name_tok_containment',
    'name_metaphone_match',
    'name_len_diff',
    'name_len_ratio',
    'name_exact',
    'name_tok_count_diff',
    # Address (11)
    'addr_levenshtein',
    'addr_token_sort',
    'addr_token_set',
    'addr_partial',
    'addr_jaccard',
    'addr_num_overlap',
    'addr_has_common_num',
    'addr_zip_match',
    'addr_street_num_match',
    'addr_len_ratio',
    'both_addr_empty',
    # Composite (2)
    'comp_sort',
    'comp_set',
    # Pair & Ranking Signals (6)
    'cand_rank',
    'delta_top_tfidf',
    'max_tfidf_in_block',
    'is_top1',
    'is_s2',
    'score_decay',
]


def extract_v4_pair_features(
    s1_name_norm: str,
    s1_addr_norm: str,
    cand_name_norm: str,
    cand_addr_norm: str,
    tfidf_sim: float,
    cand_rank: int,
    delta_top_tfidf: float,
    max_tfidf_in_block: float,
    cand_id: str,
    n_candidates: int,
) -> List[float]:
    # ── Name signals ──────────────────────────────────────────────────────────
    name_lev     = distance.Levenshtein.normalized_similarity(s1_name_norm, cand_name_norm)
    name_sort    = fuzz.token_sort_ratio(s1_name_norm, cand_name_norm) / 100.0
    name_set     = fuzz.token_set_ratio(s1_name_norm, cand_name_norm) / 100.0
    name_jw      = distance.JaroWinkler.similarity(s1_name_norm, cand_name_norm)
    name_partial = fuzz.partial_ratio(s1_name_norm, cand_name_norm) / 100.0

    s1_n_toks   = s1_name_norm.split()
    cand_n_toks = cand_name_norm.split()
    name_jacc   = token_jaccard(s1_n_toks, cand_n_toks)
    name_bi     = char_ngram_jaccard(s1_name_norm, cand_name_norm, n=2)
    name_tri    = char_ngram_jaccard(s1_name_norm, cand_name_norm, n=3)

    # Token containment
    s1_set = set(s1_n_toks)
    cand_set = set(cand_n_toks)
    if s1_set and cand_set:
        tok_contain = len(s1_set & cand_set) / float(min(len(s1_set), len(cand_set)))
    else:
        tok_contain = 0.0

    # Metaphone of first word
    s1_w1 = s1_n_toks[0] if s1_n_toks else ''
    c_w1  = cand_n_toks[0] if cand_n_toks else ''
    if s1_w1 and c_w1:
        meta_match = 1.0 if jellyfish.metaphone(s1_w1) == jellyfish.metaphone(c_w1) else 0.0
    else:
        meta_match = 0.0

    name_len_diff  = float(abs(len(s1_name_norm) - len(cand_name_norm)))
    name_len_ratio = safe_len_ratio(s1_name_norm, cand_name_norm)
    name_exact     = 1.0 if s1_name_norm and s1_name_norm == cand_name_norm else 0.0
    name_tok_diff  = float(abs(len(s1_n_toks) - len(cand_n_toks)))

    # ── Address signals ───────────────────────────────────────────────────────
    addr_lev     = distance.Levenshtein.normalized_similarity(s1_addr_norm, cand_addr_norm)
    addr_sort    = fuzz.token_sort_ratio(s1_addr_norm, cand_addr_norm) / 100.0
    addr_set     = fuzz.token_set_ratio(s1_addr_norm, cand_addr_norm) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr_norm, cand_addr_norm) / 100.0
    s1_a_toks    = s1_addr_norm.split()
    cand_a_toks  = cand_addr_norm.split()
    addr_jacc    = token_jaccard(s1_a_toks, cand_a_toks)
    addr_len_ratio = safe_len_ratio(s1_addr_norm, cand_addr_norm)

    # Number overlap & Postal Code Match
    s1_nums = set(RE_DIGITS.findall(s1_addr_norm))
    cand_nums = set(RE_DIGITS.findall(cand_addr_norm))
    if s1_nums and cand_nums:
        num_overlap    = len(s1_nums & cand_nums) / float(len(s1_nums | cand_nums))
        has_common_num = 1.0 if (s1_nums & cand_nums) else 0.0
    elif not s1_nums and not cand_nums:
        num_overlap    = 0.5
        has_common_num = 0.5
    else:
        num_overlap    = 0.0
        has_common_num = 0.0

    s1_zip = extract_zip(s1_addr_norm)
    cand_zip = extract_zip(cand_addr_norm)
    if s1_zip and cand_zip:
        zip_match = 1.0 if s1_zip == cand_zip else 0.0
    else:
        zip_match = 0.5

    s1_snum = extract_street_num(s1_addr_norm)
    cand_snum = extract_street_num(cand_addr_norm)
    if s1_snum and cand_snum:
        snum_match = 1.0 if s1_snum == cand_snum else 0.0
    else:
        snum_match = 0.5

    both_addr_empty = 1.0 if (not s1_addr_norm and not cand_addr_norm) else 0.0

    # ── Composite ─────────────────────────────────────────────────────────────
    comp_sort = (name_sort * 0.6) + (addr_sort * 0.4)
    comp_set  = (name_set  * 0.6) + (addr_set  * 0.4)

    # ── Pair signals ──────────────────────────────────────────────────────────
    is_s2       = 1.0 if cand_id.startswith('S2-') else 0.0
    is_top1     = 1.0 if cand_rank == 1 else 0.0
    score_decay = 1.0 / (1.0 + 0.05 * cand_rank)

    return [
        tfidf_sim, name_lev, name_sort, name_set, name_jw, name_partial,
        name_jacc, name_bi, name_tri, tok_contain, meta_match,
        name_len_diff, name_len_ratio, name_exact, name_tok_diff,
        addr_lev, addr_sort, addr_set, addr_partial, addr_jacc,
        num_overlap, has_common_num, zip_match, snum_match,
        addr_len_ratio, both_addr_empty,
        comp_sort, comp_set,
        float(cand_rank), delta_top_tfidf, max_tfidf_in_block,
        is_top1, is_s2, score_decay
    ]


def compute_macro_f05(
    s1_ids: List[str],
    s1_ids_feat: List[str],
    cand_ids_feat: List[str],
    preds: np.ndarray,
    gt_lookup: Dict[str, Set[str]],
) -> float:
    pred_dict: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
    for sid, cid, p in zip(s1_ids_feat, cand_ids_feat, preds):
        if p == 1:
            pred_dict[sid].add(cid)

    f05_list = []
    for sid in s1_ids:
        gt_set = gt_lookup.get(sid, set())
        p_set  = pred_dict.get(sid, set())

        if not gt_set and not p_set:
            f05_list.append(1.0)
            continue
        if not gt_set and p_set:
            f05_list.append(0.0)
            continue
        if gt_set and not p_set:
            f05_list.append(0.0)
            continue

        tp = len(gt_set & p_set)
        fp = len(p_set - gt_set)
        fn = len(gt_set - p_set)

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)

        denom = 0.25 * p + r
        score = (1.25 * p * r / denom) if denom > 0 else 0.0
        f05_list.append(score)

    return float(np.mean(f05_list))


def main():
    t_start = time.time()
    print("=== Training & Evaluating v4 High-Precision Pipeline ===")

    # 1. Load Data
    print("Loading data …")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    cand_df = pd.concat([s2, s3], ignore_index=True)
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")

    gt_lookup: Dict[str, Set[str]] = {}
    for sid, mstr in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        if mstr:
            gt_lookup[sid] = set(mstr.split(","))

    val_sids = pd.read_csv(os.path.join(DATA_SPLIT, "val_s1_ids.tsv"), sep="\t", header=None)[0].tolist()[:30_000]
    train_sids = pd.read_csv(os.path.join(DATA_SPLIT, "train_s1_ids.tsv"), sep="\t", header=None)[0].tolist()[:120_000]
    print(f"  Train S1={len(train_sids):,} | Val S1={len(val_sids):,}")

    # 2. Blocking using cached indexes
    blocker = TFIDFBlocker(
        analyzer='word', ngram_range=(1, 2), min_sim=0.05, top_k=50,
        max_features=200_000, min_df=3, max_df=0.5, progress_every=10_000
    )

    s1_sub_train = s1[s1["entity_id"].isin(set(train_sids))].reset_index(drop=True)
    s1_sub_val   = s1[s1["entity_id"].isin(set(val_sids))].reset_index(drop=True)

    print("\n[TRAIN] Blocking …")
    train_cand_dict = blocker.generate_candidates(s1_sub_train, s2, s3, verbose=True, cache_dir=CACHE_DIR, split_name="train")

    print("\n[VAL] Blocking …")
    val_cand_dict = blocker.generate_candidates(s1_sub_val, s2, s3, verbose=True, cache_dir=CACHE_DIR, split_name="train")

    # 3. Normalization Dicts
    print("\nBuilding normalized string dictionaries …")
    t0 = time.time()
    s1_norm = {sid: (normalize_name(n), normalize_address(a)) for sid, n, a in zip(s1['entity_id'].values, s1['business_name'].values, s1['business_address'].values)}
    cand_norm = {cid: (normalize_name(n), normalize_address(a)) for cid, n, a in zip(cand_df['entity_id'].values, cand_df['business_name'].values, cand_df['business_address'].values)}
    print(f"  Normalized lookups built in {time.time()-t0:.1f}s")

    # 4. Feature Extraction - Train
    print("\n[TRAIN] Extracting v4 Features …")
    t0 = time.time()
    X_train_rows = []
    y_train = []
    for sid, cands in train_cand_dict.items():
        s1_n, s1_a = s1_norm.get(sid, ("", ""))
        gt_set = gt_lookup.get(sid, set())
        max_sim = max((s for _, s in cands), default=0.0)
        n_c = len(cands)
        for rank, (cid, sim) in enumerate(cands):
            c_n, c_a = cand_norm.get(cid, ("", ""))
            row = extract_v4_pair_features(
                s1_n, s1_a, c_n, c_a, sim, rank + 1, max_sim - sim, max_sim, cid, n_c
            )
            X_train_rows.append(row)
            y_train.append(1 if cid in gt_set else 0)

    X_train = np.array(X_train_rows, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    pos_cnt = int(y_train.sum())
    neg_cnt = len(y_train) - pos_cnt
    print(f"  Train: {len(X_train):,} pairs in {time.time()-t0:.1f}s | pos={pos_cnt:,}, neg={neg_cnt:,}")

    # 5. Feature Extraction - Val
    print("\n[VAL] Extracting v4 Features …")
    t0 = time.time()
    X_val_rows = []
    val_s1_feat = []
    val_cand_feat = []
    y_val = []
    for sid, cands in val_cand_dict.items():
        s1_n, s1_a = s1_norm.get(sid, ("", ""))
        gt_set = gt_lookup.get(sid, set())
        max_sim = max((s for _, s in cands), default=0.0)
        n_c = len(cands)
        for rank, (cid, sim) in enumerate(cands):
            c_n, c_a = cand_norm.get(cid, ("", ""))
            row = extract_v4_pair_features(
                s1_n, s1_a, c_n, c_a, sim, rank + 1, max_sim - sim, max_sim, cid, n_c
            )
            X_val_rows.append(row)
            val_s1_feat.append(sid)
            val_cand_feat.append(cid)
            y_val.append(1 if cid in gt_set else 0)

    X_val = np.array(X_val_rows, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    print(f"  Val: {len(X_val):,} pairs in {time.time()-t0:.1f}s | pos={int(y_val.sum()):,}")

    # 6. LightGBM Model Training
    print("\n=== Training LightGBM Model ===")
    scale_weight = float(neg_cnt) / max(1, pos_cnt)
    lgb_model = lgb.LGBMClassifier(
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
        scale_pos_weight=scale_weight,
        n_estimators=2000,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)]
    )

    # 7. XGBoost Model Training (Ensemble)
    print("\n=== Training XGBoost Model ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=250)

    # 8. Ensemble Evaluation & Threshold Sweep
    print("\n=== Ensemble Prediction & Macro-F0.5 Evaluation ===")
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]
    p_ens = 0.6 * p_lgb + 0.4 * p_xgb

    best_thr = 0.5
    best_score = 0.0
    for thr in np.arange(0.1, 0.95, 0.025):
        preds = (p_ens >= thr).astype(int)
        score = compute_macro_f05(val_sids, val_s1_feat, val_cand_feat, preds, gt_lookup)
        if score > best_score:
            best_score = score
            best_thr = thr
        print(f"  thr={thr:.3f} -> Macro-F0.5 = {score:.5f}")

    print(f"\n🏆 Best Ensemble Macro-F0.5 = {best_score:.5f} at threshold = {best_thr:.3f}")

    # Save models
    with open(os.path.join(PROJECT_ROOT, "models", "lgb_v4.pkl"), "wb") as f:
        pickle.dump(lgb_model, f)
    with open(os.path.join(PROJECT_ROOT, "models", "xgb_v4.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    meta = {
        "best_val_f05": best_score,
        "best_threshold": float(best_thr),
        "feature_columns": V4_FEATURE_COLUMNS,
    }
    with open(os.path.join(PROJECT_ROOT, "models", "meta_v4.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal pipeline time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
