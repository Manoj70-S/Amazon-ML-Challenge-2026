"""
sweep_thresholds.py
===================
Fast, memory-efficient threshold sweep and post-processing calibration.
"""
import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "dataset", "train")
DATA_SPLIT   = os.path.join(PROJECT_ROOT, "data_split")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

from normalize import normalize_name, normalize_address
from blocking import TFIDFBlocker
from eval_v4_pipeline import V4_FEATURE_COLUMNS, compute_macro_f05, extract_v4_pair_features

def main():
    print("=== Fast High-Precision Threshold Calibration ===")
    t0 = time.time()
    
    # 1. Load models
    print("1. Loading saved v4 ensemble models...")
    with open(os.path.join(MODEL_DIR, "lgb_v4.pkl"), "rb") as f:
        lgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgb_v4.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    print(f"   Models loaded in {time.time()-t0:.1f}s")
    
    # 2. Load val S1 and candidate data
    print("2. Loading validation split...")
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    val_sids = pd.read_csv(os.path.join(DATA_SPLIT, "val_s1_ids.tsv"), sep="\t", header=None)[0].tolist()[:30_000]
    val_sid_set = set(val_sids)
    s1_val = s1[s1["entity_id"].isin(val_sid_set)].reset_index(drop=True)
    del s1
    
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    gt_lookup = {}
    for sid, mstr in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        if mstr:
            gt_lookup[sid] = set(mstr.split(","))
    del gt
    
    # 3. Fast Candidate Generation via cached posting list
    print("3. Generating validation candidates...")
    t0 = time.time()
    blocker = TFIDFBlocker(
        analyzer='word', ngram_range=(1, 2), min_sim=0.05, top_k=50,
        max_features=200_000, min_df=3, max_df=0.5, progress_every=10_000
    )
    val_cand_dict = blocker.generate_candidates(s1_val, s2, s3, verbose=True, cache_dir=CACHE_DIR, split_name="train")
    print(f"   Candidates generated in {time.time()-t0:.1f}s")
    
    # 4. Extract required candidates
    needed_cids = set(cid for cands in val_cand_dict.values() for cid, _ in cands)
    print(f"   Total unique candidate entities needed: {len(needed_cids):,}")
    
    cand_dict_raw = {}
    for df in [s2, s3]:
        sub = df[df["entity_id"].isin(needed_cids)]
        for eid, name, addr in zip(sub["entity_id"].values, sub["business_name"].values, sub["business_address"].values):
            cand_dict_raw[eid] = (name, addr)
    del s2, s3
    
    # 5. Normalize strings
    print("4. Normalizing string lookups...")
    t0 = time.time()
    s1_norm = {sid: (normalize_name(n), normalize_address(a)) for sid, n, a in zip(s1_val["entity_id"].values, s1_val["business_name"].values, s1_val["business_address"].values)}
    cand_norm = {cid: (normalize_name(n), normalize_address(a)) for cid, (n, a) in cand_dict_raw.items()}
    del cand_dict_raw
    print(f"   Normalized in {time.time()-t0:.1f}s")
    
    # 6. Extract v4 features
    print("5. Extracting v4 pair features...")
    t0 = time.time()
    X_val_rows = []
    val_s1_feat = []
    val_cand_feat = []
    
    for sid, cands in val_cand_dict.items():
        s1_n, s1_a = s1_norm.get(sid, ("", ""))
        max_sim = max((s for _, s in cands), default=0.0)
        n_c = len(cands)
        for rank, (cid, sim) in enumerate(cands):
            c_n, c_a = cand_norm.get(cid, ("", ""))
            row = extract_v4_pair_features(s1_n, s1_a, c_n, c_a, sim, rank + 1, max_sim - sim, max_sim, cid, n_c)
            X_val_rows.append(row)
            val_s1_feat.append(sid)
            val_cand_feat.append(cid)
            
    X_val = np.array(X_val_rows, dtype=np.float32)
    print(f"   Feature matrix: {X_val.shape} in {time.time()-t0:.1f}s")
    
    # 7. Model Inference
    print("6. Predicting probabilities with ensemble (0.6 LGB + 0.4 XGB)...")
    t0 = time.time()
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]
    p_ens = 0.6 * p_lgb + 0.4 * p_xgb
    print(f"   Predictions done in {time.time()-t0:.1f}s")
    
    # 8. Sweep thresholds
    print("\n" + "="*60)
    print(f"{'Threshold':>10} | {'Macro-F0.5':>12} | {'Positives':>10} | {'Singletons':>10}")
    print("="*60)
    
    best_thr = 0.5
    best_f05 = 0.0
    for thr in np.arange(0.85, 0.995, 0.01):
        preds = (p_ens >= thr).astype(int)
        score = compute_macro_f05(val_sids, val_s1_feat, val_cand_feat, preds, gt_lookup)
        n_pos = np.sum(preds)
        print(f"{thr:>10.3f} | {score:>12.5f} | {n_pos:>10,d} | {len(val_sids)-len(set(val_s1_feat[i] for i in range(len(preds)) if preds[i]==1)):>10,d}")
        if score > best_f05:
            best_f05 = score
            best_thr = thr
            
    print("="*60)
    print(f"\n>>> PEAK ENSEMBLE SCORE: Macro-F0.5 = {best_f05:.5f} at threshold = {best_thr:.3f} <<<")
    
    # 9. Test Graph Transitive Closure Boost
    print("\n=== Testing Multi-Source Transitive Recovery ===")
    # For every S1, if we predicted an S2 match with high confidence (e.g. >0.90),
    # check candidate S3 that shares exact ZIP and high name similarity with the matched S2!
    for min_s2_conf in [0.85, 0.90, 0.95]:
        for s3_bonus_thr in [0.70, 0.80, 0.85]:
            # Let's map S1 -> predictions
            # and evaluate
            pass

if __name__ == "__main__":
    main()
