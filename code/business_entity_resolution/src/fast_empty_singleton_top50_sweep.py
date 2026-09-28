#!/usr/bin/env python3
"""
fast_empty_singleton_top50_sweep.py
===================================
Targeted Top-50 inference specifically on the 184,940 empty singleton queries.
- Candidates ranked 13-50 only (ranks 1-12 were already evaluated).
- Excludes the 4,599,009 candidates already assigned to other S1 entities (prevents collisions upfront).
- Bypasses Metaphone calculation for maximum feature extraction speed.
- Predicts with Stacking Ensemble (0.6 LGBM + 0.4 XGBoost) at threshold p >= 0.88.
- Resolves any intra-batch candidate collisions using highest-probability-wins.
- Merges newly found matches into matching_results.tsv.
- Enforces 0 multi-assignment collisions.
- Automatically validates output with utils/validate_submission.py.
"""

import os
import sys
import gc
import time
import pickle
import shutil
import numpy as np
from typing import Dict, List, Tuple, Set, NamedTuple
from collections import defaultdict, Counter
from rapidfuzz import fuzz, distance

SRC_DIR      = os.path.abspath("code/business_entity_resolution/src")
PROJECT_ROOT = os.path.abspath(".")
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")

CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_TSV  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUTPUT_V9_TSV = os.path.join(OUTPUT_DIR, "matching_results_v9_sweep.tsv")
LGB_MODEL_PATH= os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH= os.path.join(MODEL_DIR, "xgb_v4.pkl")

sys.path.append(SRC_DIR)
from normalize import normalize_name, normalize_address

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

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

def parse_entity_fast(raw_name: str, raw_addr: str) -> EntityRecord:
    n = normalize_name(raw_name) if raw_name else ""
    a = normalize_address(raw_addr) if raw_addr else ""
    n_toks = n.split()
    n_set = set(n_toks)
    w1_meta = ""
    a_toks = a.split()
    nums = [x for x in a.split() if x.isdigit()]
    a_nums = set(nums)
    z = [x for x in nums if len(x) in (5, 6)]
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

def extract_features(s1: EntityRecord, c: EntityRecord, sim: float, rank: int, delta_top: float, max_sim: float, cid: str) -> List[float]:
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
    meta_match    = 0.0

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

def main():
    t_global = time.time()
    print("=" * 70)
    print("  FAST TOP-50 EMPTY SINGLETON RESOLUTION SWEEP")
    print("  Target Deadline: 3:00 PM IST")
    print("=" * 70)

    # 1. Load Stacking Ensemble models
    print("\n1. Loading Stacking Ensemble models ...")
    t0 = time.time()
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lgb_model.set_params(n_jobs=8)
    xgb_model.set_params(n_jobs=8)
    print(f"   Models loaded in {time.time()-t0:.1f}s")

    # 2. Read base submission matching_results.tsv
    print("\n2. Reading base submission matching_results.tsv ...")
    t0 = time.time()
    s1_ordered = []
    base_matches = {}
    assigned_candidates = set()
    empty_s1_list = []

    with open(MATCHING_TSV, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            s1_ordered.append(sid)
            if len(p) > 1 and p[1]:
                mids = [m.strip() for m in p[1].split(",") if m.strip()]
                base_matches[sid] = mids
                for mid in mids:
                    assigned_candidates.add(mid)
            else:
                base_matches[sid] = []
                empty_s1_list.append(sid)

    empty_s1_set = set(empty_s1_list)
    print(f"   Total S1 queries:              {len(s1_ordered):,}")
    print(f"   Currently matched queries:     {len(s1_ordered) - len(empty_s1_list):,}")
    print(f"   Empty singleton queries:       {len(empty_s1_list):,}")
    print(f"   Already assigned candidates:   {len(assigned_candidates):,}")
    print(f"   Read in {time.time()-t0:.1f}s")

    # 3. Read candidate pairs for empty S1 queries (ranks 13-50, unassigned only)
    print("\n3. Scanning candidate_pairs.tsv for empty singletons (ranks 13-50) ...")
    t0 = time.time()
    pairs_by_s1 = defaultdict(list)
    needed_candidates = set()
    total_unassigned_pairs = 0

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in empty_s1_set:
                if len(p) > 1 and p[1]:
                    cands = [c.strip() for c in p[1].split(",")][12:50]
                    for idx, c in enumerate(cands):
                        if c not in assigned_candidates:
                            rank = idx + 13
                            pairs_by_s1[sid].append((c, rank))
                            needed_candidates.add(c)
                            total_unassigned_pairs += 1

    print(f"   Empty queries with candidates: {len(pairs_by_s1):,}")
    print(f"   Total unassigned pairs:        {total_unassigned_pairs:,}")
    print(f"   Distinct unassigned cands:     {len(needed_candidates):,}")
    print(f"   Scanned in {time.time()-t0:.1f}s")

    # 4. Load pre-parsed S1 entities for empty singletons only
    print("\n4. Loading pre-parsed S1 records for empty singletons ...")
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "s1_parsed.pkl"), "rb") as f:
        raw_s1 = pickle.load(f)
    s1_parsed = {k: EntityRecord(*raw_s1[k]) for k in empty_s1_set if k in raw_s1}
    del raw_s1
    gc.collect()
    print(f"   Loaded {len(s1_parsed):,} S1 records in {time.time()-t0:.1f}s")

    # 5. Index raw candidate text for needed candidate IDs
    print(f"\n5. Indexing raw text for {len(needed_candidates):,} unassigned candidates ...")
    t0 = time.time()
    cand_raw_text = {}
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, src)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_candidates:
                    cand_raw_text[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")
    print(f"   Loaded text for {len(cand_raw_text):,} candidates in {time.time()-t0:.1f}s")

    # 6. Stream chunked inference
    print("\n" + "=" * 70)
    print("6. STARTING STREAMING INFERENCE ON EMPTY SINGLETONS")
    print("=" * 70)

    CHUNK_SIZE = 10_000
    t_start_inf = time.time()
    cand_cache = {}
    new_passing_matches = []  # list of (s1_id, cid, prob, rank)
    
    empty_with_cands = [sid for sid in empty_s1_list if sid in pairs_by_s1]
    total_chunks = (len(empty_with_cands) + CHUNK_SIZE - 1) // CHUNK_SIZE
    processed_queries = 0
    processed_pairs = 0

    checkpoint_file = os.path.join(OUTPUT_DIR, "empty_top50_checkpoint.tsv")
    with open(checkpoint_file, "w", encoding="utf-8") as f_chk:
        f_chk.write("source1_entity_id\tcandidate_id\tprob\trank\n")

    for c_idx in range(total_chunks):
        t_c_start = time.time()
        chunk_sids = empty_with_cands[c_idx * CHUNK_SIZE : (c_idx + 1) * CHUNK_SIZE]
        rows = []
        pair_meta = []  # (sid, cid, rank)
        max_sim = 1.0

        for sid in chunk_sids:
            s1_rec = s1_parsed.get(sid)
            if not s1_rec:
                continue
            for cid, rank in pairs_by_s1[sid]:
                c_rec = cand_cache.get(cid)
                if c_rec is None:
                    raw_n, raw_a = cand_raw_text.get(cid, ("", ""))
                    c_rec = parse_entity_fast(raw_n, raw_a)
                    cand_cache[cid] = c_rec

                sim = 1.0 / (1.0 + 0.05 * (rank - 1))
                row = extract_features(s1_rec, c_rec, sim, rank, max_sim - sim, max_sim, cid)
                rows.append(row)
                pair_meta.append((sid, cid, rank))

        chunk_pairs_count = len(rows)
        processed_pairs += chunk_pairs_count
        processed_queries += len(chunk_sids)

        if chunk_pairs_count > 0:
            X = np.array(rows, dtype=np.float32)
            p_lgb = lgb_model.predict_proba(X)[:, 1]
            p_xgb = xgb_model.predict_proba(X)[:, 1]
            p_ens = 0.6 * p_lgb + 0.4 * p_xgb

            chunk_passes = []
            for (sid, cid, rank), p in zip(pair_meta, p_ens):
                if p >= 0.88:
                    chunk_passes.append((sid, cid, float(p), rank))

            new_passing_matches.extend(chunk_passes)

            # Append to checkpoint file
            with open(checkpoint_file, "a", encoding="utf-8") as f_chk:
                for sid, cid, prob, rank in chunk_passes:
                    f_chk.write(f"{sid}\t{cid}\t{prob:.4f}\t{rank}\n")

        elapsed_inf = time.time() - t_start_inf
        query_rate = processed_queries / max(1.0, elapsed_inf)
        remaining_queries = len(empty_with_cands) - processed_queries
        eta_sec = remaining_queries / max(1.0, query_rate)
        eta_min = eta_sec / 60.0

        print(f"   [Chunk {c_idx+1:02d}/{total_chunks:02d}] "
              f"Queries: {processed_queries:,}/{len(empty_with_cands):,} "
              f"| Pairs: {chunk_pairs_count:,} "
              f"| Speed: {query_rate:.1f} q/s "
              f"| New Matches: {len(new_passing_matches):,} "
              f"| ETA: {eta_min:.1f}m")

    t_inf_total = time.time() - t_start_inf
    print(f"\n   Inference completed in {t_inf_total/60.0:.1f} minutes!")
    print(f"   Total passing candidate pairs found (p >= 0.88): {len(new_passing_matches):,}")

    # 7. Collision Purge: Enforce 0 multi-assigned candidates among new matches
    print("\n" + "=" * 70)
    print("7. RESOLVING COLLISIONS AMONG NEW MATCHES (HIGHEST PROBABILITY WINS)")
    print("=" * 70)

    # Sort new passing matches by probability descending
    new_passing_matches.sort(key=lambda x: x[2], reverse=True)

    # Candidate -> best (s1_id, prob)
    cand_to_best_s1 = {}
    purged_collisions = 0

    for sid, cid, prob, rank in new_passing_matches:
        if cid in assigned_candidates:
            # Should already be 0 by construction, but double check
            continue
        if cid not in cand_to_best_s1:
            cand_to_best_s1[cid] = (sid, prob)
        else:
            purged_collisions += 1

    print(f"   Multi-assigned collisions purged: {purged_collisions:,}")
    print(f"   Distinct new candidates successfully assigned: {len(cand_to_best_s1):,}")

    # Group approved new matches by S1 entity ID
    s1_to_new_matches = defaultdict(list)
    for cid, (sid, prob) in cand_to_best_s1.items():
        s1_to_new_matches[sid].append(cid)

    recovered_entities = len(s1_to_new_matches)
    print(f"   Previously empty entities now matched: {recovered_entities:,} "
          f"({recovered_entities / len(empty_s1_list) * 100:.2f}%)")

    # 8. Merge into base matching_results.tsv preserving exact test row order
    print("\n" + "=" * 70)
    print("8. MERGING RESULTS INTO matching_results_v9_sweep.tsv")
    print("=" * 70)

    final_non_empty = 0
    final_empty = 0

    with open(OUTPUT_V9_TSV, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ordered:
            base_m = base_matches.get(sid, [])
            new_m  = s1_to_new_matches.get(sid, [])
            all_m  = base_m + new_m
            if all_m:
                f_out.write(f"{sid}\t{','.join(all_m)}\n")
                final_non_empty += 1
            else:
                f_out.write(f"{sid}\t\n")
                final_empty += 1

    print(f"   Written to: {OUTPUT_V9_TSV}")
    print(f"   Total rows:      {len(s1_ordered):,}")
    print(f"   Non-empty rows:  {final_non_empty:,}")
    print(f"   Empty singletons:{final_empty:,}")

    # 9. Direct verification of 0 multi-assigned candidates
    print("\n" + "=" * 70)
    print("9. STRICT VERIFICATION OF ZERO MULTI-ASSIGNMENT COLLISIONS")
    print("=" * 70)
    c_check = Counter()
    with open(OUTPUT_V9_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1]:
                for cid in p[1].split(","):
                    c_check[cid.strip()] += 1

    dups = {k: v for k, v in c_check.items() if v > 1}
    print(f"   Total distinct candidate assignments: {len(c_check):,}")
    print(f"   Multi-assigned candidate count:       {len(dups):,}")
    assert len(dups) == 0, f"FATAL ERROR: Found {len(dups)} duplicate candidate assignments!"
    print("   ZERO COLLISIONS CONFIRMED! (PASS)")

    # 10. Run official submission validator
    print("\n" + "=" * 70)
    print("10. RUNNING OFFICIAL VALIDATOR: utils/validate_submission.py")
    print("=" * 70)
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(UTILS_DIR, "validate_submission.py"),
        "--matching", OUTPUT_V9_TSV,
        "--candidate", CANDIDATE_TSV,
        "--test-dir", DATA_DIR,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("VALIDATOR FAILED:")
        print(res.stderr)
        sys.exit(1)

    # 11. Overwrite output/matching_results.tsv with validated v9
    backup_path = os.path.join(OUTPUT_DIR, "matching_results_v8_backup.tsv")
    if not os.path.exists(backup_path):
        shutil.copy2(MATCHING_TSV, backup_path)
        print(f"   Created backup: {backup_path}")
    shutil.copy2(OUTPUT_V9_TSV, MATCHING_TSV)
    print(f"   Updated {MATCHING_TSV} with validated Top-50 recovered matches!")

    total_elapsed = time.time() - t_global
    print("\n" + "=" * 70)
    print(f"ALL STEPS COMPLETED SUCCESSFULLY IN {total_elapsed/60.0:.1f} MINUTES!")
    print(f"Submission file is 100% ready and validated: {MATCHING_TSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()
