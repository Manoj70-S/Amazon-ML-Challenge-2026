"""
test_transitive_boost.py
========================
Test Multi-Source Transitive Recovery on Validation Predictions.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict

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
    print("=== Testing Multi-Source Transitive Expansion ===")
    
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
        s1_cand_preds[sid].append((cid, prob, c_n, c_a))
        
    print("\n--- Evaluating Baseline without Transitive Closure ---")
    base_preds = (p_ens >= 0.940).astype(int)
    base_f05 = compute_macro_f05(val_sids, val_s1_feat, val_cand_feat, base_preds, gt_lookup)
    print(f"Base Threshold 0.940 -> Macro-F0.5 = {base_f05:.5f}")
    
    print("\n--- Testing Transitive Parameter Configurations ---")
    for min_anchor_p in [0.90, 0.92, 0.94]:
        for secondary_p_min in [0.60, 0.70, 0.80]:
            for min_sim_between_cands in [80.0, 85.0, 90.0]:
                pred_map = defaultdict(set)
                for sid in val_sids:
                    cands = s1_cand_preds.get(sid, [])
                    # Direct high confidence
                    anchors = [c for c in cands if c[1] >= min_anchor_p]
                    for cid, _, _, _ in anchors:
                        pred_map[sid].add(cid)
                        
                    # If we have an anchor from S2/S3, check for complementary source
                    if anchors:
                        for a_cid, a_p, a_n, a_a in anchors:
                            a_is_s2 = a_cid.startswith("S2-")
                            for c_cid, c_p, c_n, c_a in cands:
                                if c_cid in pred_map[sid]:
                                    continue
                                c_is_s2 = c_cid.startswith("S2-")
                                # Must be complementary source (one S2, one S3)
                                if (a_is_s2 and not c_is_s2) or (not a_is_s2 and c_is_s2):
                                    if c_p >= secondary_p_min:
                                        # Check text similarity between anchor and candidate
                                        name_sim = fuzz.token_sort_ratio(a_n, c_n)
                                        if name_sim >= min_sim_between_cands:
                                            pred_map[sid].add(c_cid)
                    else:
                        # Fallback to direct threshold
                        for cid, p, _, _ in cands:
                            if p >= 0.940:
                                pred_map[sid].add(cid)
                                
                # Evaluate
                f05_list = []
                for sid in val_sids:
                    gt_set = gt_lookup.get(sid, set())
                    p_set = pred_map.get(sid, set())
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
                    
                score = float(np.mean(f05_list))
                print(f"Anchor>={min_anchor_p:.2f} | Sec>={secondary_p_min:.2f} | Sim>={min_sim_between_cands:.0f} -> Macro-F0.5 = {score:.5f}")

if __name__ == "__main__":
    main()
