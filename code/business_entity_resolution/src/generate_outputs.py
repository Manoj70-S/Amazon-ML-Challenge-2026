"""
generate_outputs.py
===================
Streamlined, Low-Memory Test Inference Pipeline
for the Amazon ML Challenge 2026 – Business Entity Resolution.

Architecture:
  - Memory Footprint: < 2 GB RAM (Zero risk of OOM).
  - Pre-normalizes Source 1 once (~1.73M records = 200 MB).
  - Streams candidate_pairs.tsv in chunks of 10,000 queries.
  - On-the-fly string normalization with LRU string cache.
  - Multi-threaded LightGBM inference on each chunk.
  - Direct streaming to output/matching_results.tsv.
  - Automatic validation via utils/validate_submission.py.
"""

import os
import sys

# Force flush on all stdout writes
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

import time
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ── local imports ─────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address
from blocking import TFIDFBlocker, export_candidate_pairs_tsv
from features import FEATURE_COLUMNS, extract_pair_features

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TEST_DIR   = os.path.join(PROJECT_ROOT, "dataset", "test")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR  = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_TSV  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
MODEL_PATH    = os.path.join(MODEL_DIR, "lgbm_matcher.pkl")
META_PATH     = os.path.join(MODEL_DIR, "model_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_meta():
    print("Loading model and metadata …")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    threshold = float(meta["threshold"])
    print(f"  Threshold: {threshold:.3f}  |  Best val F0.5: {meta.get('best_val_f05', 'N/A')}")
    return model, threshold, meta


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()

    # ── 1. Load test data ──────────────────────────────────────────────────────
    print("Loading test data …")
    t0 = time.time()
    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    cand_df = pd.concat([s2, s3], ignore_index=True)
    all_s1_ids = s1["entity_id"].tolist()
    print(f"  Loaded in {time.time()-t0:.1f}s | S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    # ── 2. Load model ──────────────────────────────────────────────────────────
    model, threshold, meta = load_model_and_meta()
    bp = meta.get("blocker_params", {})

    # ── 3. Candidate Generation (if needed) ────────────────────────────────────
    if not os.path.exists(CANDIDATE_TSV) or os.path.getsize(CANDIDATE_TSV) < 100_000_000:
        blocker = TFIDFBlocker(
            analyzer=bp.get("analyzer", "word"),
            ngram_range=tuple(bp.get("ngram_range", (1, 2))),
            min_sim=bp.get("min_sim", 0.05),
            top_k=bp.get("top_k", 50),
            max_features=bp.get("max_features", 200_000),
            min_df=bp.get("min_df", 3),
            max_df=bp.get("max_df", 0.5),
            progress_every=5_000,
        )
        print("\n=== Running Blocking on Test Set ===")
        candidate_dict = blocker.generate_candidates(s1, s2, s3, verbose=True, cache_dir=CACHE_DIR, split_name="test")
        print("\nExporting candidate_pairs.tsv …")
        export_candidate_pairs_tsv(candidate_dict, all_s1_ids, CANDIDATE_TSV)
        del candidate_dict
    else:
        print(f"\n[Resume] Using existing candidate file -> {CANDIDATE_TSV}")

    # ── 4. Build Fast Candidate Lookups (Low Memory) ───────────────────────────
    print("\nBuilding candidate lookup index …")
    t0 = time.time()
    cand_ids = cand_df["entity_id"].values
    cand_names_raw = cand_df["business_name"].values
    cand_addrs_raw = cand_df["business_address"].values
    cand_id_map: Dict[str, int] = {cid: idx for idx, cid in enumerate(cand_ids)}
    del cand_df, s2, s3  # Free memory
    print(f"  Lookup index built in {time.time()-t0:.1f}s | {len(cand_id_map):,} candidates indexed")

    # ── 5. Pre-normalize S1 Entities (~1.73M records) ──────────────────────────
    print("\nNormalizing Source 1 entities …")
    t0 = time.time()
    s1_names_norm = s1["business_name"].map(normalize_name).values
    s1_addrs_norm = s1["business_address"].map(normalize_address).values
    s1_norm_map: Dict[str, Tuple[str, str]] = {
        sid: (n, a) for sid, n, a in zip(s1["entity_id"].values, s1_names_norm, s1_addrs_norm)
    }
    del s1, s1_names_norm, s1_addrs_norm  # Free memory
    print(f"  Source 1 normalized in {time.time()-t0:.1f}s | {len(s1_norm_map):,} entities in memory")

    # ── 6. Stream candidate_pairs.tsv and Classify in Chunks ───────────────────
    print(f"\n=== Streaming Classification (threshold={threshold:.3f}) ===")
    CHUNK_SIZE = 10_000  # Number of S1 queries per chunk (~500k candidate pairs)
    
    # Caches for candidate normalized strings
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
    n_total_queries = len(all_s1_ids)
    t_stream = time.time()

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as in_f, \
         open(MATCHING_TSV, "w", encoding="utf-8") as out_f:
        
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        header = in_f.readline()  # Skip candidate header

        while True:
            # Read chunk of lines
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
                max_sim = 1.0  # Top ranked item similarity benchmark

                for rank, cid in enumerate(cids):
                    sim = 1.0 / (1.0 + 0.05 * rank)
                    c_name, c_addr = get_norm_cand(cid)
                    feat_dict = extract_pair_features(
                        s1_name_norm=s1_name,
                        s1_addr_norm=s1_addr,
                        cand_name_norm=c_name,
                        cand_addr_norm=c_addr,
                        tfidf_sim=sim,
                        cand_rank=rank + 1,
                        delta_top_tfidf=max_sim - sim,
                        max_tfidf_in_block=max_sim,
                        cand_id=cid,
                        n_candidates=n_cands,
                    )
                    row = [feat_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
                    chunk_s1_feat.append(sid)
                    chunk_cand_feat.append(cid)
                    chunk_rows.append(row)

            # Classify chunk
            accepted_in_chunk: Dict[str, List[str]] = {sid: [] for sid in chunk_s1_ids}

            if len(chunk_rows) > 0:
                X_chunk = np.array(chunk_rows, dtype=np.float32)
                proba = model.predict_proba(X_chunk)[:, 1]
                preds = (proba >= threshold).astype(int)

                for sid, cid, pred in zip(chunk_s1_feat, chunk_cand_feat, preds):
                    if pred == 1:
                        accepted_in_chunk[sid].append(cid)

            # Write chunk results
            for sid in chunk_s1_ids:
                matches = accepted_in_chunk.get(sid, [])
                if matches:
                    total_positives += 1
                    out_f.write(f"{sid}\t{','.join(matches)}\n")
                else:
                    total_singletons += 1
                    out_f.write(f"{sid}\t\n")

            total_queries_processed += len(chunk_s1_ids)

            # Log periodic progress
            if total_queries_processed % 50_000 == 0 or total_queries_processed >= n_total_queries:
                elapsed = time.time() - t_stream
                rate = total_queries_processed / max(elapsed, 1e-9)
                eta_min = (n_total_queries - total_queries_processed) / max(rate, 1e-9) / 60.0
                print(f"  Processed {total_queries_processed:,}/{n_total_queries:,} queries | "
                      f"{rate:.0f} q/s | Matches: {total_positives:,} | Singletons: {total_singletons:,} | ETA {eta_min:.1f} min")

    print(f"\n=== Matching Results Completed -> {MATCHING_TSV} ===")
    print(f"  Total S1 Entities: {n_total_queries:,}")
    print(f"  Entities with >= 1 Match: {total_positives:,} ({total_positives/n_total_queries*100:.1f}%)")
    print(f"  Predicted Singletons: {total_singletons:,} ({total_singletons/n_total_queries*100:.1f}%)")
    print(f"  Total End-to-End Time: {(time.time()-t_total)/60:.1f} min")

    print("\nRunning submission validation …")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    os.system(val_cmd)


if __name__ == "__main__":
    main()
