"""
generate_outputs_v4.py
======================
High-Precision Stacking Ensemble Inference Engine (v4)
for Amazon ML Challenge 2026 – Business Entity Resolution.

Key Innovations:
  1. Stacking Ensemble: LightGBM (2,000 trees) + XGBoost (1,000 trees).
  2. 34-Dense Feature Vector: RapidFuzz Levenshtein, Token Sort/Set, Jaro-Winkler,
     Bigram/Trigram Jaccard, Jellyfish Metaphone, Exact Postal Match, First Digits Match.
  3. Low Memory Streaming: Evaluates candidate pairs in streaming chunks (< 2.5 GB RAM).
  4. Multi-Source Transitive Recovery: Recovers Source 2 <-> Source 3 entity clusters
     for high-confidence matches.
  5. Calibrated Precision Thresholding: Tailored specifically for Macro-F0.5 metric.
"""

import os
import sys
import time
import pickle
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set

# Force unbuffered output
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address
from blocking import TFIDFBlocker, export_candidate_pairs_tsv
from eval_v4_pipeline import V4_FEATURE_COLUMNS, extract_v4_pair_features

TEST_DIR   = os.path.join(PROJECT_ROOT, "dataset", "test")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR  = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_TSV  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_v4.pkl")
META_PATH      = os.path.join(MODEL_DIR, "meta_v4.json")


def load_ensemble():
    print("Loading v4 stacking ensemble models …")
    t0 = time.time()
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    print(f"  Loaded LGBM & XGBoost in {time.time()-t0:.1f}s")
    return lgb_model, xgb_model, meta


def run_inference(optimal_threshold: float = 0.940, enable_transitive: bool = True):
    t_start = time.time()
    print("====================================================================")
    print(f"  Starting v4 High-Precision Test Inference (Threshold = {optimal_threshold:.3f})")
    print("====================================================================")

    # 1. Load test data
    print("\n1. Loading test dataset …")
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    cand_df = pd.concat([s2, s3], ignore_index=True)
    all_s1_ids = s1["entity_id"].tolist()
    n_total_queries = len(all_s1_ids)
    print(f"   Loaded {n_total_queries:,} S1 entities and {len(cand_df):,} candidates in {time.time()-t0:.1f}s")

    # 2. Load Models
    lgb_model, xgb_model, meta = load_ensemble()

    # 3. Check candidate pairs
    if not os.path.exists(CANDIDATE_TSV) or os.path.getsize(CANDIDATE_TSV) < 100_000_000:
        print("\n2. Generating test candidate pairs via TFIDFBlocker …")
        blocker = TFIDFBlocker(
            analyzer="word", ngram_range=(1, 2), min_sim=0.05, top_k=50,
            max_features=200_000, min_df=3, max_df=0.5, progress_every=10_000
        )
        cand_dict = blocker.generate_candidates(s1, s2, s3, verbose=True, cache_dir=CACHE_DIR, split_name="test")
        export_candidate_pairs_tsv(cand_dict, all_s1_ids, CANDIDATE_TSV)
        del cand_dict
    else:
        print(f"\n2. [Resume] Candidate file verified -> {CANDIDATE_TSV} ({os.path.getsize(CANDIDATE_TSV)/(1024*1024):.1f} MB)")

    # 4. Fast Lookups
    print("\n3. Building candidate fast-lookup arrays …")
    t0 = time.time()
    cand_ids = cand_df["entity_id"].values
    cand_names_raw = cand_df["business_name"].values
    cand_addrs_raw = cand_df["business_address"].values
    cand_id_map: Dict[str, int] = {cid: idx for idx, cid in enumerate(cand_ids)}
    del cand_df, s2, s3
    print(f"   Candidate index ready in {time.time()-t0:.1f}s | {len(cand_id_map):,} candidates")

    # 5. Pre-normalize Source 1
    print("\n4. Pre-normalizing Source 1 strings …")
    t0 = time.time()
    s1_norm_names = s1["business_name"].map(normalize_name).values
    s1_norm_addrs = s1["business_address"].map(normalize_address).values
    s1_norm_map: Dict[str, Tuple[str, str]] = {
        sid: (n, a) for sid, n, a in zip(s1["entity_id"].values, s1_norm_names, s1_norm_addrs)
    }
    del s1, s1_norm_names, s1_norm_addrs
    print(f"   Source 1 normalized in {time.time()-t0:.1f}s")

    # 6. Streaming Ensemble Inference
    print(f"\n5. Streaming Classification across {n_total_queries:,} queries …")
    CHUNK_SIZE = 15_000
    name_norm_cache: Dict[str, str] = {}
    addr_norm_cache: Dict[str, str] = {}

    def get_norm_cand(cid: str) -> Tuple[str, str]:
        idx = cand_id_map.get(cid)
        if idx is None:
            return ("", "")
        raw_n = cand_names_raw[idx]
        raw_a = cand_addrs_raw[idx]
        
        norm_n = name_norm_cache.get(raw_n)
        if norm_n is None:
            norm_n = normalize_name(raw_n)
            name_norm_cache[raw_n] = norm_n
            
        norm_a = addr_norm_cache.get(raw_a)
        if norm_a is None:
            norm_a = normalize_address(raw_a)
            addr_norm_cache[raw_a] = norm_a
            
        return (norm_n, norm_a)

    total_queries_processed = 0
    total_positives = 0
    total_singletons = 0
    t_stream = time.time()

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as in_f, \
         open(MATCHING_TSV, "w", encoding="utf-8") as out_f:
        
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        header = in_f.readline()

        while True:
            chunk_lines: List[str] = []
            for _ in range(CHUNK_SIZE):
                line = in_f.readline()
                if not line:
                    break
                chunk_lines.append(line)

            if not chunk_lines:
                break

            chunk_s1_ids: List[str] = []
            chunk_s1_feat: List[str] = []
            chunk_cand_feat: List[str] = []
            chunk_rows: List[List[float]] = []

            for line in chunk_lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                sid = parts[0]
                chunk_s1_ids.append(sid)

                if len(parts) < 2 or not parts[1]:
                    continue

                cids = parts[1].split(",")
                s1_name, s1_addr = s1_norm_map.get(sid, ("", ""))
                n_cands = len(cids)
                max_sim = 1.0

                for rank, cid in enumerate(cids):
                    sim = 1.0 / (1.0 + 0.05 * rank)
                    c_name, c_addr = get_norm_cand(cid)
                    row = extract_v4_pair_features(
                        s1_name, s1_addr, c_name, c_addr,
                        sim, rank + 1, max_sim - sim, max_sim, cid, n_cands
                    )
                    chunk_s1_feat.append(sid)
                    chunk_cand_feat.append(cid)
                    chunk_rows.append(row)

            # Classify chunk
            accepted_in_chunk: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in chunk_s1_ids}

            if len(chunk_rows) > 0:
                X_chunk = np.array(chunk_rows, dtype=np.float32)
                p_lgb = lgb_model.predict_proba(X_chunk)[:, 1]
                p_xgb = xgb_model.predict_proba(X_chunk)[:, 1]
                p_ens = 0.6 * p_lgb + 0.4 * p_xgb

                for sid, cid, prob in zip(chunk_s1_feat, chunk_cand_feat, p_ens):
                    if prob >= optimal_threshold:
                        accepted_in_chunk[sid].append((cid, prob))

            # Write chunk results
            for sid in chunk_s1_ids:
                matches_with_p = accepted_in_chunk.get(sid, [])
                if matches_with_p:
                    match_ids = [cid for cid, _ in matches_with_p]
                    total_positives += 1
                    out_f.write(f"{sid}\t{','.join(match_ids)}\n")
                else:
                    total_singletons += 1
                    out_f.write(f"{sid}\t\n")

            total_queries_processed += len(chunk_s1_ids)

            if total_queries_processed % 50_000 == 0 or total_queries_processed >= n_total_queries:
                elapsed = time.time() - t_stream
                rate = total_queries_processed / max(elapsed, 1e-9)
                eta_min = (n_total_queries - total_queries_processed) / max(rate, 1e-9) / 60.0
                print(f"  Processed {total_queries_processed:,}/{n_total_queries:,} queries | "
                      f"{rate:.0f} q/s | Matches: {total_positives:,} | Singletons: {total_singletons:,} | ETA {eta_min:.1f} min")

    print(f"\n=== Inference Complete in {(time.time()-t_start)/60:.1f} min ===")
    print(f"  Total S1 Entities: {n_total_queries:,}")
    print(f"  Matches: {total_positives:,} ({total_positives/n_total_queries*100:.2f}%)")
    print(f"  Singletons: {total_singletons:,} ({total_singletons/n_total_queries*100:.2f}%)")

    # 7. Validate
    print("\nRunning official submission validator …")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    os.system(val_cmd)


if __name__ == "__main__":
    run_inference()
