"""
test_transitive_boost_fast.py
=============================
Ultra-fast test of multi-source transitive expansion.
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict

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
from rapidfuzz import fuzz

def main():
    print("=== Ultra-Fast Multi-Source Transitive Expansion Test ===")
    
    with open(os.path.join(MODEL_DIR, "lgb_v4.pkl"), "rb") as f:
        lgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgb_v4.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
        
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    val_sids = pd.read_csv(os.path.join(DATA_SPLIT, "val_s1_ids.tsv"), sep="\t", header=None)[0].tolist()[:30_000]
    s1_val = s1[s1["entity_id"].isin(set(val_sids))].reset_index(drop=True)
    del s1
    
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    gt_lookup = {}
    for sid, mstr in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        if mstr:
            gt_lookup[sid] = set(mstr.split(","))
    del gt
    
    blocker = TFIDFBlocker(analyzer='word', ngram_range=(1, 2), min_sim=0.05, top_k=50, max_features=200_000, min_df=3, max_df=0.5)
    val_cand_dict = blocker.generate_candidates(s1_val, s2, s3, verbose=False, cache_dir=CACHE_DIR, split_name="train")
    
    needed_cids = set(cid for cands in val_cand_dict.values() for cid, _ in cands)
    cand_dict_raw = {}
    for df in [s2, s3]:
        sub = df[df["entity_id"].isin(needed_cids)]
        for eid, name, addr in zip(sub["entity_id"].values, sub["business_name"].values, sub["business_address"].values):
            cand_dict_raw[eid] = (name, addr)
    del s2, s3
    
    s1_norm = {sid: (normalize_name(n), normalize_address(a)) for sid, n, a in zip(s1_val["entity_id"].values, s1_val["business_name"].values, s1_val["business_address"].values)}
    cand_norm = {cid: (normalize_name(n), normalize_address(a)) for cid, (n, a) in cand_dict_raw.items()}
    
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
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]
    p_ens = 0.6 * p_lgb + 0.4 * p_xgb
    
    # Store predictions per S1: sid -> list of (cid, prob, c_name, c_addr)
    s1_cand_preds = defaultdict(list)
    for sid, cid, prob in zip(val_s1_feat, val_cand_feat, p_ens):
        c_n, c_a = cand_norm.get(cid, ("", ""))
        s1_cand_preds[sid].append((cid, float(prob), c_n, c_a))
        
    print("\n--- Evaluating Pure High-Precision Thresholds ---")
    for thr in [0.930, 0.935, 0.940, 0.945]:
        base_preds = (p_ens >= thr).astype(int)
        score = compute_macro_f05(val_sids, val_s1_feat, val_cand_feat, base_preds, gt_lookup)
        print(f"  Threshold {thr:.3f} -> Macro-F0.5 = {score:.5f}")

if __name__ == "__main__":
    main()
