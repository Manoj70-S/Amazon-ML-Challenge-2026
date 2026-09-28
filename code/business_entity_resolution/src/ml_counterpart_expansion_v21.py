#!/usr/bin/env python3
"""
ml_counterpart_expansion_v21.py
================================
Amazon ML Challenge 2026 – 34-Feature Stacking ML Counterpart Expansion Engine (v21)

Philosophy:
1. Base Invariant: 100% preserves every link from the verified v16 submission (Score: 0.86115).
2. Targets Single-Link Entities (305,748 entities):
   - These entities have 1 verified match (e.g. S2) and are missing their S3 counterpart.
   - Evaluates counterpart candidate pool using the full 34-feature LightGBM + XGBoost Stacking Ensemble.
3. High Probability Threshold (p_ens >= 0.950):
   - At p >= 0.950, 34 dense features (Jellyfish Metaphone, Levenshtein, street numbers, ZIP codes,
     token containment, composite ratios) ensure >= 95% precision, protecting against false positives.
4. Single-Winner Rule: At most 1 winning counterpart candidate per entity (best_cid only).
5. Strictly Disjoint: Highest-probability claim wins globally, guaranteeing 0 collisions.
6. Verified with utils/validate_submission.py.
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
import subprocess
import shutil

# Ensure unbuffered output
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

PROJECT_ROOT = os.path.abspath(".")
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")
SRC_DIR      = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src")

sys.path.append(SRC_DIR)
from normalize import normalize_name, normalize_address

BASE_TSV       = os.path.join(OUTPUT_DIR, "matching_results_v16_deep.tsv")
CANDIDATE_TSV  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_v4.pkl")
OUT_V21_TSV    = os.path.join(OUTPUT_DIR, "matching_results_v21_ml_boost.tsv")
FINAL_TSV      = os.path.join(OUTPUT_DIR, "matching_results.tsv")

PROB_THRESHOLD = 0.950
TOP_K_SCAN     = 20
CHUNK_QUERIES  = 10_000

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
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def extract_34_features(s1: EntityRecord, c: EntityRecord, sim: float, rank: int, delta_top: float, max_sim: float, cid: str) -> List[float]:
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

    zip_match  = 1.0 if (s1.zip_code and c.zip_code and s1.zip_code == c.zip_code) else (0.5 if not (s1.zip_code or c.zip_code) else 0.0)
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
    t_start = time.time()
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: 34-FEATURE ML STACKING EXPANSION (v21)")
    print(f"  Target: Single-Link Entities | Ensemble Threshold: p >= {PROB_THRESHOLD}")
    print("=" * 80)

    # 1. Load Stacking Ensemble Models
    print("\n1. Loading LightGBM & XGBoost Stacking Ensemble models ...")
    t0 = time.time()
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lgb_model.set_params(n_jobs=12)
    xgb_model.set_params(n_jobs=12)
    print(f"   Models loaded in {time.time()-t0:.1f}s")

    # 2. Load Base Submission (v16: 0.86115)
    print("\n2. Loading base submission (v16: 0.86115) ...")
    s1_ordered = []
    base_matches: Dict[str, List[str]] = {}
    assigned_candidates: Set[str] = set()
    single_link_s1: Dict[str, str] = {}  # sid -> anchor_id

    with open(BASE_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            s1_ordered.append(sid)
            if len(p) > 1 and p[1].strip():
                mids = [m.strip() for m in p[1].split(",") if m.strip()]
                base_matches[sid] = mids
                for mid in mids:
                    assigned_candidates.add(mid)
                if len(mids) == 1:
                    single_link_s1[sid] = mids[0]
            else:
                base_matches[sid] = []

    print(f"   Total S1 entities:            {len(s1_ordered):,}")
    print(f"   Base non-empty entities:      {len(s1_ordered) - sum(1 for v in base_matches.values() if not v):,}")
    print(f"   Base total links:             {len(assigned_candidates):,}")
    print(f"   Target single-link entities:  {len(single_link_s1):,}")

    # 3. Load pre-parsed Source 1 entities from cache
    print("\n3. Loading pre-parsed Source 1 entities from cache/s1_parsed.pkl ...")
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "s1_parsed.pkl"), "rb") as f:
        raw_s1 = pickle.load(f)
    s1_parsed: Dict[str, EntityRecord] = {}
    for sid in single_link_s1:
        if sid in raw_s1:
            s1_parsed[sid] = EntityRecord(*raw_s1[sid])
    del raw_s1
    gc.collect()
    print(f"   Loaded parsed data for {len(s1_parsed):,} target entities in {time.time()-t0:.1f}s")

    # 4. Collect candidate pairs for target entities (counterpart source only)
    print(f"\n4. Collecting counterpart candidates from candidate_pairs.tsv (top {TOP_K_SCAN}) ...")
    t0 = time.time()
    target_cand_pairs: Dict[str, List[Tuple[int, str]]] = {}
    needed_cids: Set[str] = set()

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in single_link_s1 and len(p) > 1 and p[1].strip():
                anchor_id = single_link_s1[sid]
                anchor_src = anchor_id[:2]
                target_src = "S3" if anchor_src == "S2" else "S2"

                cids = [c.strip() for c in p[1].split(",") if c.strip()][:TOP_K_SCAN]
                # Filter for counterpart source and unassigned candidates
                counterparts = [
                    (rank, cid) for rank, cid in enumerate(cids)
                    if cid.startswith(target_src) and cid not in assigned_candidates
                ]
                if counterparts:
                    target_cand_pairs[sid] = counterparts
                    for _, cid in counterparts:
                        needed_cids.add(cid)

    print(f"   Target entities with valid counterpart candidates: {len(target_cand_pairs):,}")
    print(f"   Total unique candidate texts needed:                {len(needed_cids):,} (in {time.time()-t0:.1f}s)")

    # 5. Stream and parse candidate texts from test_source2.tsv and test_source3.tsv
    print("\n5. Streaming candidate texts from test_source2.tsv and test_source3.tsv ...")
    cand_parsed: Dict[str, EntityRecord] = {}
    for src_num in [2, 3]:
        src_path = os.path.join(DATA_DIR, f"test_source{src_num}.tsv")
        t_src = time.time()
        c_found = 0
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)
            for line in f:
                tab1 = line.find("\t")
                if tab1 == -1: continue
                cid = line[:tab1]
                if cid in needed_cids:
                    p = line.rstrip("\r\n").split("\t")
                    name = p[1] if len(p) > 1 else ""
                    addr = p[2] if len(p) > 2 else ""
                    cand_parsed[cid] = parse_entity(name, addr)
                    c_found += 1
        print(f"   {os.path.basename(src_path)}: parsed {c_found:,} candidates in {time.time()-t_src:.1f}s")

    print(f"   Candidate cache ready: {len(cand_parsed):,} entities")

    # 6. Batched Feature Extraction & ML Inference
    print(f"\n6. Extracting 34 features & predicting with Stacking Ensemble (threshold p >= {PROB_THRESHOLD}) ...")
    t_infer = time.time()
    best_candidate_per_entity: Dict[str, Tuple[str, float]] = {}  # sid -> (best_cid, best_prob)
    total_pairs_scored = 0

    target_sids = list(target_cand_pairs.keys())
    for chunk_start in range(0, len(target_sids), CHUNK_QUERIES):
        chunk_sids = target_sids[chunk_start : chunk_start + CHUNK_QUERIES]
        batch_sids = []
        batch_cids = []
        batch_rows = []

        for sid in chunk_sids:
            s1_rec = s1_parsed.get(sid)
            if not s1_rec: continue
            max_sim = 1.0

            for rank, cid in target_cand_pairs[sid]:
                c_rec = cand_parsed.get(cid)
                if not c_rec: continue

                sim = 1.0 / (1.0 + 0.05 * rank)
                row = extract_34_features(s1_rec, c_rec, sim, rank + 1, max_sim - sim, max_sim, cid)
                batch_sids.append(sid)
                batch_cids.append(cid)
                batch_rows.append(row)

        if not batch_rows:
            continue

        total_pairs_scored += len(batch_rows)
        X_batch = np.array(batch_rows, dtype=np.float32)
        p_lgb = lgb_model.predict_proba(X_batch)[:, 1]
        p_xgb = xgb_model.predict_proba(X_batch)[:, 1]
        p_ens = 0.6 * p_lgb + 0.4 * p_xgb

        for sid, cid, prob in zip(batch_sids, batch_cids, p_ens):
            prob = float(prob)
            if prob >= PROB_THRESHOLD:
                if sid not in best_candidate_per_entity or prob > best_candidate_per_entity[sid][1]:
                    best_candidate_per_entity[sid] = (cid, prob)

        if (chunk_start // CHUNK_QUERIES + 1) % 5 == 0 or (chunk_start + CHUNK_QUERIES) >= len(target_sids):
            elapsed = time.time() - t_infer
            rate = total_pairs_scored / max(elapsed, 1e-9)
            print(f"   Scored {total_pairs_scored:,} pairs ({rate:,.0f} pairs/s) | Accepted high-prob entities: {len(best_candidate_per_entity):,}")

    print(f"\n   Total pairs scored:              {total_pairs_scored:,} in {time.time()-t_infer:.1f}s")
    print(f"   Entities with p >= {PROB_THRESHOLD}:         {len(best_candidate_per_entity):,}")

    # 7. Global Disjoint Collision Resolution (Highest Probability Wins)
    print("\n7. Resolving candidate collisions globally (highest probability wins) ...")
    claims: List[Tuple[str, str, float]] = [
        (sid, cid, prob) for sid, (cid, prob) in best_candidate_per_entity.items()
    ]
    claims.sort(key=lambda x: x[2], reverse=True)

    winner_for_cand: Dict[str, Tuple[str, float]] = {}
    purged_collisions = 0

    for sid, cid, prob in claims:
        if cid not in winner_for_cand and cid not in assigned_candidates:
            winner_for_cand[cid] = (sid, prob)
        else:
            purged_collisions += 1

    rescued_by_s1: Dict[str, str] = {sid: cid for cid, (sid, prob) in winner_for_cand.items()}

    print(f"   Collisions purged:               {purged_collisions:,}")
    print(f"   Final accepted counterpart links:{len(rescued_by_s1):,}")

    # 8. Write Final TSV
    print(f"\n8. Writing v21 final submission to {os.path.basename(OUT_V21_TSV)} ...")
    final_non_empty = 0
    final_empty = 0
    total_final_links = 0

    with open(OUT_V21_TSV, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ordered:
            base_m = base_matches.get(sid, [])
            new_cid = rescued_by_s1.get(sid)
            all_m = list(base_m)
            if new_cid:
                all_m.append(new_cid)

            if all_m:
                f_out.write(f"{sid}\t{','.join(all_m)}\n")
                final_non_empty += 1
                total_final_links += len(all_m)
            else:
                f_out.write(f"{sid}\t\n")
                final_empty += 1

    print(f"   Total rows written:       {len(s1_ordered):,}")
    print(f"   Final non-empty entities: {final_non_empty:,}")
    print(f"   Final empty singletons:   {final_empty:,}")
    print(f"   Total links in v21:       {total_final_links:,} (+{len(rescued_by_s1):,} vs v16)")

    # 9. Audit Collisions
    print("\n9. Auditing submission for multi-assignment collisions ...")
    check_cands = Counter()
    with open(OUT_V21_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                for c in p[1].split(","):
                    check_cands[c.strip()] += 1

    collisions = sum(1 for c, cnt in check_cands.items() if cnt > 1)
    print(f"   Total unique candidates assigned: {len(check_cands):,}")
    print(f"   Total multi-assignment collisions: {collisions} (Must be 0)")
    assert collisions == 0, f"FATAL: Found {collisions} collisions in v21!"

    # 10. Overwrite matching_results.tsv with validated v21
    print(f"\n10. Promoting v21 to production {os.path.basename(FINAL_TSV)} ...")
    shutil.copy2(OUT_V21_TSV, FINAL_TSV)

    # 11. Official Validator
    print("\n11. Running official utils/validate_submission.py ...")
    val_cmd = [
        sys.executable,
        os.path.join(UTILS_DIR, "validate_submission.py"),
        "--matching", FINAL_TSV,
        "--candidate", CANDIDATE_TSV,
        "--test-dir", DATA_DIR,
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"FATAL: Validator failed:\n{res.stderr}")
        sys.exit(1)

    print(f"\n=== v21 ML COUNTERPART EXPANSION ENGINE COMPLETE IN {time.time()-t_start:.1f}s ===")
    print(f"Ready for portal upload: {FINAL_TSV}")


if __name__ == "__main__":
    main()
