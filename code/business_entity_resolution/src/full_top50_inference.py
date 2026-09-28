#!/usr/bin/env python3
"""
full_top50_inference.py
=======================
Full Top-50 Candidate Streaming Inference Pipeline with Incremental Checkpointing
and Automatic Highest-Probability Collision Resolution.

Key Features:
  1. Full Top-50 Evaluation: Evaluates all 50 candidates from candidate_pairs.tsv.
  2. Incremental Checkpointing: Appends partial results to disk after every chunk (10,000 queries)
     and supports resuming seamlessly if interrupted.
  3. Probability Tracking: Stores model ensemble probability p_ens for each accepted pair.
  4. Collision Purge: Globally resolves multi-assigned candidates using highest-probability-wins.
  5. Official Submission Validation: Runs utils/validate_submission.py and confirms PASS.
"""

import os
import sys
import gc
import time
import re
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, NamedTuple
from collections import defaultdict, Counter
from rapidfuzz import fuzz, distance
import jellyfish

# Ensure ASCII-safe flushed stdout on Windows
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

CANDIDATE_TSV    = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
RAW_OUTPUT_TSV   = os.path.join(OUTPUT_DIR, "matching_results_top50_raw.tsv")
FINAL_OUTPUT_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
LGB_MODEL_PATH   = os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH   = os.path.join(MODEL_DIR, "xgb_v4.pkl")

# Threshold Policy
TAU1 = 0.88   # Tier 1: Primary Anchor
TAU2 = 0.80   # Tier 2: Conservative Cluster Expansion
TAU3 = 0.55   # Tier 3: Singleton Rescue
NAME_SORT_T3 = 0.35

CHUNK_SIZE  = 10_000
TOP_K       = 50
CACHE_PURGE = 100_000
TOTAL_S1    = 1_732_544

sys.path.append(SRC_DIR)
from normalize import normalize_name, normalize_address

RE_DIGITS = re.compile(r"\b\d+\b")
RE_ZIP    = re.compile(r"\b\d{5,6}\b")

class EntityRecord(NamedTuple):
    name: str
    addr: str
    n_toks: List[str]
    n_set: Set[str]
    w1_meta: str
    a_toks: List[str]
    a_nums: Set[str]
    zip_code: str
    snum: str

def parse_entity(raw_name: str, raw_addr: str) -> EntityRecord:
    n = normalize_name(raw_name) if raw_name else ""
    a = normalize_address(raw_addr) if raw_addr else ""
    n_toks = n.split()
    n_set = set(n_toks)
    w1_meta = jellyfish.metaphone(n_toks[0]) if n_toks else ""
    a_toks = a.split()
    nums = RE_DIGITS.findall(a)
    a_nums = set(nums)
    z = RE_ZIP.findall(a)
    zip_code = z[-1] if z else ""
    snum = nums[0] if nums else ""
    return EntityRecord(n, a, n_toks, n_set, w1_meta, a_toks, a_nums, zip_code, snum)

def char_ngram_jaccard(s1: str, s2: str, n: int = 2) -> float:
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    set1 = set(s1[i : i + n] for i in range(len(s1) - n + 1))
    set2 = set(s2[i : i + n] for i in range(len(s2) - n + 1))
    denom = len(set1 | set2)
    return (len(set1 & set2) / denom) if denom > 0 else 0.0

def safe_len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0: return 1.0
    if la == 0 or lb == 0: return 0.0
    return min(la, lb) / max(la, lb)

def extract_fast_features(s1: EntityRecord, c: EntityRecord, sim: float, rank: int, delta_top: float, max_sim: float, cid: str) -> List[float]:
    name_lev     = distance.Levenshtein.normalized_similarity(s1.name, c.name)
    name_sort    = fuzz.token_sort_ratio(s1.name, c.name) / 100.0
    name_set     = fuzz.token_set_ratio(s1.name, c.name) / 100.0
    name_jw      = distance.JaroWinkler.similarity(s1.name, c.name)
    name_partial = fuzz.partial_ratio(s1.name, c.name) / 100.0

    if s1.n_set and c.n_set:
        inter       = len(s1.n_set & c.n_set)
        name_jacc   = inter / float(len(s1.n_set | c.n_set))
        tok_contain = inter / float(min(len(s1.n_set), len(c.n_set)))
    else:
        name_jacc   = 0.0
        tok_contain = 0.0

    name_bi       = char_ngram_jaccard(s1.name, c.name, n=2)
    name_tri      = char_ngram_jaccard(s1.name, c.name, n=3)
    meta_match    = 1.0 if (s1.w1_meta and s1.w1_meta == c.w1_meta) else 0.0
    name_len_diff = float(abs(len(s1.name) - len(c.name)))
    name_len_ratio= safe_len_ratio(s1.name, c.name)
    name_exact    = 1.0 if s1.name and s1.name == c.name else 0.0
    name_tok_diff = float(abs(len(s1.n_toks) - len(c.n_toks)))

    addr_lev     = distance.Levenshtein.normalized_similarity(s1.addr, c.addr)
    addr_sort    = fuzz.token_sort_ratio(s1.addr, c.addr) / 100.0
    addr_set     = fuzz.token_set_ratio(s1.addr, c.addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1.addr, c.addr) / 100.0

    if s1.a_toks and c.a_toks:
        set1, set2 = set(s1.a_toks), set(c.a_toks)
        addr_jacc  = len(set1 & set2) / float(len(set1 | set2))
    else:
        addr_jacc  = 0.0

    if s1.a_nums and c.a_nums:
        num_overlap    = len(s1.a_nums & c.a_nums) / float(len(s1.a_nums | c.a_nums))
        has_common_num = 1.0 if (s1.a_nums & c.a_nums) else 0.0
    elif not s1.a_nums and not c.a_nums:
        num_overlap    = 0.5
        has_common_num = 0.5
    else:
        num_overlap    = 0.0
        has_common_num = 0.0

    zip_match = 1.0 if (s1.zip_code and c.zip_code and s1.zip_code == c.zip_code) else (0.5 if not (s1.zip_code or c.zip_code) else 0.0)
    snum_match = 1.0 if (s1.snum and c.snum and s1.snum == c.snum) else (0.5 if not (s1.snum or c.snum) else 0.0)

    addr_len_ratio    = safe_len_ratio(s1.addr, c.addr)
    both_addr_empty   = 1.0 if (not s1.addr and not c.addr) else 0.0

    comp_sort = (name_sort * 0.6) + (addr_sort * 0.4)
    comp_set  = (name_set  * 0.6) + (addr_set  * 0.4)

    is_s2      = 1.0 if cid.startswith("S2-") else 0.0
    is_top1    = 1.0 if rank == 1 else 0.0
    score_decay= 1.0 / (1.0 + 0.05 * rank)

    return [
        sim, name_lev, name_sort, name_set, name_jw, name_partial,
        name_jacc, name_bi, name_tri, tok_contain, meta_match,
        name_len_diff, name_len_ratio, name_exact, name_tok_diff,
        addr_lev, addr_sort, addr_set, addr_partial, addr_jacc,
        num_overlap, has_common_num, zip_match, snum_match,
        addr_len_ratio, both_addr_empty,
        comp_sort, comp_set,
        float(rank), delta_top, max_sim,
        is_top1, is_s2, score_decay,
    ]

def process_chunk_top50(
    chunk_lines: List[str],
    s1_parsed: Dict[str, EntityRecord],
    cand_id_map: Dict[str, int],
    cand_names_raw,
    cand_addrs_raw,
    cand_parsed_cache: Dict[str, EntityRecord],
    lgb_model,
    xgb_model,
    raw_out_f,
) -> Tuple[int, int, int]:
    def get_cand_record(cid: str) -> EntityRecord:
        rec = cand_parsed_cache.get(cid)
        if rec is None:
            idx = cand_id_map.get(cid)
            if idx is None:
                rec = EntityRecord("", "", [], set(), "", [], set(), "", "")
            else:
                rec = parse_entity(cand_names_raw[idx], cand_addrs_raw[idx])
            cand_parsed_cache[cid] = rec
        return rec

    chunk_queries: List[Tuple[str, List[Tuple[int, str]]]] = []
    for line in chunk_lines:
        parts = line.split("\t")
        sid = parts[0]
        if len(parts) < 2 or not parts[1]:
            chunk_queries.append((sid, []))
            continue
        cids = parts[1].split(",")[:TOP_K] # Full top-50
        chunk_queries.append((sid, [(rank, cid) for rank, cid in enumerate(cids)]))

    all_sids: List[str] = []
    all_cids: List[str] = []
    all_rows: List[List[float]] = []
    all_c_recs: List[EntityRecord] = []

    for sid, cid_list in chunk_queries:
        s1_rec = s1_parsed.get(sid)
        if s1_rec is None or not cid_list:
            continue
        max_sim = 1.0
        for rank, cid in cid_list:
            c_rec = get_cand_record(cid)
            sim = 1.0 / (1.0 + 0.05 * rank)
            row = extract_fast_features(s1_rec, c_rec, sim, rank + 1, max_sim - sim, max_sim, cid)
            all_sids.append(sid)
            all_cids.append(cid)
            all_rows.append(row)
            all_c_recs.append(c_rec)

    query_preds: Dict[str, List[Tuple[str, float, EntityRecord]]] = {}
    if all_rows:
        X = np.array(all_rows, dtype=np.float32)
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        p_ens = 0.6 * p_lgb + 0.4 * p_xgb

        for sid, cid, prob, c_rec in zip(all_sids, all_cids, p_ens, all_c_recs):
            if sid not in query_preds:
                query_preds[sid] = []
            query_preds[sid].append((cid, float(prob), c_rec))

    n_q = 0
    n_anchored = 0
    n_rescued = 0

    for sid, cid_list in chunk_queries:
        n_q += 1
        s1_rec = s1_parsed.get(sid)
        preds = query_preds.get(sid, [])

        if not preds or s1_rec is None:
            raw_out_f.write(f"{sid}\t\n")
            continue

        anchors = [(cid, prob) for cid, prob, _ in preds if prob >= TAU1]

        if anchors:
            n_anchored += 1
            accepted_with_probs = [(cid, prob) for cid, prob in anchors]
            anchor_set = set(cid for cid, _ in anchors)

            # Tier 2: cluster expansion (prob >= TAU2)
            for cid, prob, _ in preds:
                if cid not in anchor_set and prob >= TAU2:
                    accepted_with_probs.append((cid, prob))

            entry_str = ",".join(f"{cid}:{prob:.4f}" for cid, prob in accepted_with_probs)
            raw_out_f.write(f"{sid}\t{entry_str}\n")

        else:
            # Tier 3: singleton rescue
            top_cid, top_prob, top_crec = preds[0]
            if top_prob >= TAU3:
                name_sort_v = fuzz.token_sort_ratio(s1_rec.name, top_crec.name) / 100.0
                snum_ok = (bool(s1_rec.snum) and bool(top_crec.snum) and s1_rec.snum == top_crec.snum)
                if name_sort_v >= NAME_SORT_T3 or snum_ok:
                    n_rescued += 1
                    raw_out_f.write(f"{sid}\t{top_cid}:{top_prob:.4f}\n")
                else:
                    raw_out_f.write(f"{sid}\t\n")
            else:
                raw_out_f.write(f"{sid}\t\n")

    return n_q, n_anchored, n_rescued

def main():
    t_start = time.time()
    print("=" * 70)
    print("  Amazon ML Challenge 2026: Full Top-50 Inference with Checkpointing")
    print("=" * 70)
    print(f"  Configuration : Full Top-{TOP_K} Candidates per Entity")
    print(f"  Thresholds    : TAU1={TAU1} (anchor) | TAU2={TAU2} (expansion) | TAU3={TAU3} (rescue)")
    print(f"  Checkpointing : Every {CHUNK_SIZE:,} queries to {RAW_OUTPUT_TSV}")
    print()

    # 1. Load models
    print("1. Loading Stacking Ensemble models ...")
    t0 = time.time()
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lgb_model.set_params(n_jobs=12)
    xgb_model.set_params(n_jobs=12)
    print(f"   Models loaded in {time.time()-t0:.1f}s")

    # 2. Load pre-parsed Source 1
    print("\n2. Loading pre-parsed Source 1 entities from cache/s1_parsed.pkl ...")
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "s1_parsed.pkl"), "rb") as f:
        raw_s1 = pickle.load(f)
    s1_parsed: Dict[str, EntityRecord] = {k: EntityRecord(*v) for k, v in raw_s1.items()}
    del raw_s1
    gc.collect()
    print(f"   Source 1 loaded: {len(s1_parsed):,} entities in {time.time()-t0:.1f}s")

    # 3. Load candidate lookup
    print("\n3. Loading Source 2 & Source 3 candidate data ...")
    t0 = time.time()
    s2_df = pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df

    cand_ids       = cand_df["entity_id"].values
    cand_names_raw = cand_df["business_name"].values
    cand_addrs_raw = cand_df["business_address"].values
    cand_id_map: Dict[str, int] = {cid: idx for idx, cid in enumerate(cand_ids)}
    del cand_df
    gc.collect()
    print(f"   Candidate index ready in {time.time()-t0:.1f}s | {len(cand_id_map):,} candidates")

    # 4. Check for existing checkpoint
    print("\n4. Checking for existing checkpoint ...")
    already_processed = 0
    if os.path.exists(RAW_OUTPUT_TSV):
        with open(RAW_OUTPUT_TSV, "r", encoding="utf-8") as f:
            already_processed = sum(1 for _ in f) - 1 # subtract header
        if already_processed > 0:
            print(f"   Resuming from existing checkpoint: {already_processed:,} queries already done!")

    cand_parsed_cache: Dict[str, EntityRecord] = {}
    total_queries = already_processed
    t_stream = time.time()
    chunk_lines = []

    file_mode = "a" if already_processed > 0 else "w"
    raw_out_f = open(RAW_OUTPUT_TSV, file_mode, encoding="utf-8", buffering=1 << 20)
    if already_processed == 0:
        raw_out_f.write("source1_entity_id\tmatched_entity_ids_with_prob\n")

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as in_f:
        in_f.readline() # header

        # Fast forward past already processed lines
        if already_processed > 0:
            for _ in range(already_processed):
                in_f.readline()

        for line in in_f:
            line = line.strip()
            if not line: continue
            chunk_lines.append(line)

            if len(chunk_lines) >= CHUNK_SIZE:
                n_q, n_a, n_r = process_chunk_top50(
                    chunk_lines, s1_parsed, cand_id_map,
                    cand_names_raw, cand_addrs_raw,
                    cand_parsed_cache, lgb_model, xgb_model, raw_out_f
                )
                raw_out_f.flush()
                total_queries += n_q
                chunk_lines = []

                if len(cand_parsed_cache) > CACHE_PURGE:
                    cand_parsed_cache.clear()
                gc.collect()

                elapsed = time.time() - t_stream
                rate = (total_queries - already_processed) / max(elapsed, 1e-9)
                eta_min = (TOTAL_S1 - total_queries) / max(rate, 1e-9) / 60.0
                pct = total_queries / TOTAL_S1 * 100.0

                print(
                    f"  [{pct:5.1f}%] {total_queries:>10,}/{TOTAL_S1:,} | "
                    f"{rate:>6.1f} q/s | "
                    f"ETA: {eta_min:.1f} min ({eta_min/60:.2f}h) | "
                    f"Checkpoint saved"
                )

        if chunk_lines:
            n_q, n_a, n_r = process_chunk_top50(
                chunk_lines, s1_parsed, cand_id_map,
                cand_names_raw, cand_addrs_raw,
                cand_parsed_cache, lgb_model, xgb_model, raw_out_f
            )
            raw_out_f.flush()
            total_queries += n_q

    raw_out_f.close()
    elapsed_inf = time.time() - t_stream
    print(f"\nInference completed in {elapsed_inf/60:.1f} minutes ({elapsed_inf/3600:.2f} hours)")

    # 5. Global Collision Resolution: Highest-Probability-Wins Purge
    print("\n" + "="*70)
    print("STAGE 2: GLOBAL COLLISION RESOLUTION (HIGHEST-PROBABILITY-WINS)")
    print("="*70)
    print("Streaming raw predictions and mapping candidate claim probabilities ...")
    t_purge = time.time()

    s1_order = []
    # sid -> list of (cid, prob)
    s1_candidates = {}
    # cid -> list of (sid, prob)
    cand_claims = defaultdict(list)

    with open(RAW_OUTPUT_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            sid = p[0]
            s1_order.append(sid)
            s1_candidates[sid] = []
            if len(p) > 1 and p[1]:
                entries = p[1].split(",")
                for entry in entries:
                    if ":" in entry:
                        cid, prob_str = entry.split(":", 1)
                        prob = float(prob_str)
                    else:
                        cid = entry
                        prob = 1.0
                    s1_candidates[sid].append((cid, prob))
                    cand_claims[cid].append((sid, prob))

    total_claims = sum(len(v) for v in cand_claims.values())
    multi_assigned = {c: claims for c, claims in cand_claims.items() if len(claims) > 1}
    excess_links = sum(len(claims) - 1 for claims in multi_assigned.values())

    print(f"  Total candidate claims:           {total_claims:,}")
    print(f"  Unique candidates claimed:        {len(cand_claims):,}")
    print(f"  Multi-assigned candidate IDs:     {len(multi_assigned):,}")
    print(f"  Guaranteed excess false positives: {excess_links:,}")

    # Resolve: Each candidate is assigned EXCLUSIVELY to max(prob) S1
    best_s1_for_cand = {}
    for cid, claims in multi_assigned.items():
        best_sid, best_prob = max(claims, key=lambda x: x[1])
        best_s1_for_cand[cid] = best_sid

    # Write clean final output
    print(f"\nWriting cleaned final output to {FINAL_OUTPUT_TSV} ...")
    final_links = 0
    final_empty = 0
    final_non_empty = 0

    with open(FINAL_OUTPUT_TSV, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_order:
            kept_cands = []
            for cid, prob in s1_candidates[sid]:
                if cid in best_s1_for_cand:
                    if best_s1_for_cand[cid] == sid:
                        kept_cands.append(cid)
                else:
                    kept_cands.append(cid)

            if kept_cands:
                final_non_empty += 1
                final_links += len(kept_cands)
                out_f.write(f"{sid}\t{','.join(kept_cands)}\n")
            else:
                final_empty += 1
                out_f.write(f"{sid}\t\n")

    print(f"  Final Output Summary:")
    print(f"    Total S1 rows:            {len(s1_order):,}")
    print(f"    Total clean match links:  {final_links:,}")
    print(f"    Non-empty rows:           {final_non_empty:,} ({final_non_empty/len(s1_order)*100:.2f}%)")
    print(f"    Empty rows (singletons):  {final_empty:,} ({final_empty/len(s1_order)*100:.2f}%)")
    print(f"    Purge completed in {time.time()-t_purge:.1f}s")

    # Verify zero multi-assignments in final output
    print("\nVerifying 0 multi-assignment collisions in final output ...")
    final_cand_counts = Counter()
    with open(FINAL_OUTPUT_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) > 1 and p[1]:
                for c in p[1].split(","):
                    final_cand_counts[c] += 1

    remaining_dups = sum(1 for v in final_cand_counts.values() if v > 1)
    print(f"  Verification: Candidates assigned to >1 S1: {remaining_dups} (CONFIRMED 0)")
    assert remaining_dups == 0, f"Error: {remaining_dups} collisions remaining!"

    # 6. Official Submission Validator
    print("\n" + "="*70)
    print("STAGE 3: OFFICIAL SUBMISSION VALIDATION")
    print("="*70)
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    ret = os.system(val_cmd)

    if ret == 0:
        print("\nALL STAGES COMPLETE! Validator PASSED!")
        print("output/matching_results.tsv is verified, collision-free, and ready for upload!")
    else:
        print(f"\nValidator returned non-zero exit code: {ret}")

    print(f"\nTotal Pipeline Time: {(time.time()-t_start)/3600:.2f} hours")

if __name__ == "__main__":
    main()
