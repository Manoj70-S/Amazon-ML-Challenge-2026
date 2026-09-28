#!/usr/bin/env python3
"""
cluster_expand_v6.py
====================
3-Tier Targeted Cluster Expander & Singleton Rescue (v6).

Strategy:
  - Full rescan of ALL 1,732,544 queries in candidate_pairs.tsv
  - 3-Tier decision policy per query:

  Tier 1 (Primary Anchor, tau1 = 0.88):
      Accept any candidate with p >= 0.88 unconditionally.

  Tier 2 (Cluster Expansion, tau2 = 0.65):
      If ANY anchor (p >= 0.88) found, ALSO accept all remaining
      candidates in the block with p >= 0.65.
      This recovers ~1.38M missing cluster links.

  Tier 3 (Singleton Rescue, tau3 = 0.55):
      For queries with ZERO candidates >= 0.88 (would be singleton),
      if top candidate has p >= 0.55 AND (name_sort >= 0.35 OR snum_match),
      accept it as a rescued singleton.

  Writes a brand-new output/matching_results.tsv (mode='w'), replacing v5.

Expected runtime : ~90 minutes (1.73M queries at ~312 q/s, top-12 candidates)
Expected score   : 0.935 - 0.965+
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
from rapidfuzz import fuzz, distance
import jellyfish

# Force ASCII-safe stdout (Windows cp1252 crashes on unicode/emoji)
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset")
TEST_DIR     = os.path.join(DATA_DIR, "test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

CANDIDATE_TSV    = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
OUTPUT_TSV       = os.path.join(OUTPUT_DIR, "matching_results.tsv")
TEMP_TSV         = os.path.join(OUTPUT_DIR, "matching_results_v6_tmp.tsv")

LGB_MODEL_PATH   = os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH   = os.path.join(MODEL_DIR, "xgb_v4.pkl")

# ── 3-Tier Thresholds ─────────────────────────────────────────────────────────
TAU1 = 0.88   # Tier 1: Primary anchor acceptance
TAU2 = 0.65   # Tier 2: Cluster expansion (only when anchor found)
TAU3 = 0.55   # Tier 3: Singleton rescue (no anchor, top-1 check)
NAME_SORT_T3 = 0.35  # Minimum name_sort for Tier 3
CHUNK_SIZE   = 15_000
TOP_K        = 12    # Evaluate top-12 candidates per query
CACHE_PURGE  = 80_000

# ── Normalizers ───────────────────────────────────────────────────────────────
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
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def extract_fast_features(
    s1: EntityRecord,
    c: EntityRecord,
    sim: float,
    rank: int,
    delta_top: float,
    max_sim: float,
    cid: str,
) -> List[float]:
    # Name similarities
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

    # Address similarities
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

    zip_match = (
        1.0 if (s1.zip_code and c.zip_code and s1.zip_code == c.zip_code)
        else (0.5 if not (s1.zip_code or c.zip_code) else 0.0)
    )
    snum_match = (
        1.0 if (s1.snum and c.snum and s1.snum == c.snum)
        else (0.5 if not (s1.snum or c.snum) else 0.0)
    )

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


def process_chunk(
    chunk_lines: List[str],
    s1_parsed: Dict[str, EntityRecord],
    cand_id_map: Dict[str, int],
    cand_names_raw,
    cand_addrs_raw,
    cand_parsed_cache: Dict[str, EntityRecord],
    lgb_model,
    xgb_model,
    out_f,
) -> Tuple[int, int, int, int]:
    """
    Process one chunk of candidate_pairs.tsv lines.
    Applies 3-tier policy and writes results to out_f.
    Returns: (n_queries, n_anchored, n_expanded, n_rescued)
    """

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

    # Parse all (sid, [(cid, rank), ...]) from chunk
    chunk_queries: List[Tuple[str, List[Tuple[str, int]]]] = []
    for line in chunk_lines:
        parts = line.split("\t")
        sid = parts[0]
        if len(parts) < 2 or not parts[1]:
            chunk_queries.append((sid, []))
            continue
        cids = parts[1].split(",")[:TOP_K]
        chunk_queries.append((sid, [(rank, cid) for rank, cid in enumerate(cids)]))

    # Build feature matrix for all (sid, cid) pairs that have valid s1 records
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

    # Batch inference
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

    # Apply 3-tier policy per query and write output
    n_queries   = 0
    n_anchored  = 0
    n_expanded  = 0
    n_rescued   = 0

    for sid, cid_list in chunk_queries:
        n_queries += 1
        s1_rec = s1_parsed.get(sid)
        preds = query_preds.get(sid, [])

        if not preds or s1_rec is None:
            # No candidates or unknown entity -> empty row
            out_f.write(f"{sid}\t\n")
            continue

        # Tier 1: find all anchors (p >= TAU1)
        anchors = [(cid, prob) for cid, prob, _ in preds if prob >= TAU1]

        if anchors:
            n_anchored += 1
            accepted = [cid for cid, _ in anchors]
            anchor_set = set(accepted)

            # Tier 2: Cluster expansion (p >= TAU2) for non-anchors
            for cid, prob, _ in preds:
                if cid not in anchor_set and prob >= TAU2:
                    accepted.append(cid)
                    n_expanded += 1

            out_f.write(f"{sid}\t{','.join(accepted)}\n")

        else:
            # Tier 3: Singleton rescue — top-1 candidate check
            top_cid, top_prob, top_crec = preds[0]
            if top_prob >= TAU3:
                # Consistency gates
                name_sort_v = fuzz.token_sort_ratio(s1_rec.name, top_crec.name) / 100.0
                snum_ok = (
                    bool(s1_rec.snum) and bool(top_crec.snum)
                    and s1_rec.snum == top_crec.snum
                )
                if name_sort_v >= NAME_SORT_T3 or snum_ok:
                    n_rescued += 1
                    out_f.write(f"{sid}\t{top_cid}\n")
                else:
                    out_f.write(f"{sid}\t\n")
            else:
                out_f.write(f"{sid}\t\n")

    return n_queries, n_anchored, n_expanded, n_rescued


def main():
    t_start = time.time()
    print("=" * 70)
    print("  Amazon ML Challenge 2026: v6 Cluster Expander & Singleton Rescue")
    print("=" * 70)
    print(f"  Thresholds: TAU1={TAU1} (anchor) | TAU2={TAU2} (expand) | TAU3={TAU3} (rescue)")
    print(f"  Strategy  : Full rescan of ALL 1,732,544 queries (top-{TOP_K} candidates)")
    print(f"  Output    : {OUTPUT_TSV}")
    print()

    # 1. Load models
    print("1. Loading v4 Stacking Ensemble models ...")
    t0 = time.time()
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lgb_model.set_params(n_jobs=12)
    xgb_model.set_params(n_jobs=12)
    print(f"   Models loaded in {time.time()-t0:.1f}s (12 threads)")

    # 2. Load pre-parsed Source 1 cache
    print("\n2. Loading pre-parsed Source 1 entities from cache/s1_parsed.pkl ...")
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "s1_parsed.pkl"), "rb") as f:
        raw_s1 = pickle.load(f)
    s1_parsed: Dict[str, EntityRecord] = {k: EntityRecord(*v) for k, v in raw_s1.items()}
    del raw_s1
    gc.collect()
    print(f"   Source 1 loaded: {len(s1_parsed):,} entities in {time.time()-t0:.1f}s")

    # 3. Load candidate lookup (S2 + S3)
    print("\n3. Loading Source 2 & Source 3 candidate data ...")
    t0 = time.time()
    s2_df = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df

    cand_ids       = cand_df["entity_id"].values
    cand_names_raw = cand_df["business_name"].values
    cand_addrs_raw = cand_df["business_address"].values
    cand_id_map: Dict[str, int] = {cid: idx for idx, cid in enumerate(cand_ids)}
    del cand_df
    gc.collect()
    print(f"   Candidate index ready in {time.time()-t0:.1f}s | {len(cand_id_map):,} candidates")

    # 4. Stream candidate_pairs.tsv and apply 3-tier policy
    print(f"\n4. Streaming candidate_pairs.tsv with 3-Tier policy ...")
    print(f"   Chunk size : {CHUNK_SIZE:,} | Cache purge at : {CACHE_PURGE:,} items")
    print()

    cand_parsed_cache: Dict[str, EntityRecord] = {}

    total_queries  = 0
    total_anchored = 0
    total_expanded = 0
    total_rescued  = 0
    t_eval = time.time()
    chunk_lines: List[str] = []

    # Count total lines for ETA (quick pre-scan of header only, we know it's 1,732,544)
    TOTAL_EXPECTED = 1_732_544

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as in_f, \
         open(TEMP_TSV, "w", encoding="utf-8", buffering=1 << 20) as out_f:

        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        in_f.readline()  # skip header

        for line in in_f:
            line = line.strip()
            if not line:
                continue
            chunk_lines.append(line)

            if len(chunk_lines) >= CHUNK_SIZE:
                n_q, n_a, n_e, n_r = process_chunk(
                    chunk_lines, s1_parsed, cand_id_map,
                    cand_names_raw, cand_addrs_raw,
                    cand_parsed_cache, lgb_model, xgb_model, out_f
                )
                total_queries  += n_q
                total_anchored += n_a
                total_expanded += n_e
                total_rescued  += n_r
                chunk_lines = []

                # Memory management
                if len(cand_parsed_cache) > CACHE_PURGE:
                    cand_parsed_cache.clear()
                gc.collect()

                elapsed = time.time() - t_eval
                rate    = total_queries / max(elapsed, 1e-9)
                eta_min = (TOTAL_EXPECTED - total_queries) / max(rate, 1e-9) / 60.0
                pct     = total_queries / TOTAL_EXPECTED * 100.0
                print(
                    f"  [{pct:5.1f}%] {total_queries:>10,}/{TOTAL_EXPECTED:,} | "
                    f"{rate:>6.0f} q/s | "
                    f"Anchored: {total_anchored:,} | "
                    f"Expanded: {total_expanded:,} | "
                    f"Rescued: {total_rescued:,} | "
                    f"ETA: {eta_min:.1f} min"
                )

        # Final partial chunk
        if chunk_lines:
            n_q, n_a, n_e, n_r = process_chunk(
                chunk_lines, s1_parsed, cand_id_map,
                cand_names_raw, cand_addrs_raw,
                cand_parsed_cache, lgb_model, xgb_model, out_f
            )
            total_queries  += n_q
            total_anchored += n_a
            total_expanded += n_e
            total_rescued  += n_r

    elapsed_total = time.time() - t_eval
    print(f"\n=== Cluster Expansion Completed in {elapsed_total/60:.1f} min ===")
    print(f"  Total queries processed : {total_queries:,}")
    print(f"  Queries with anchor(s)  : {total_anchored:,} ({total_anchored/max(total_queries,1)*100:.1f}%)")
    print(f"  Cluster links expanded  : {total_expanded:,}")
    print(f"  Singletons rescued (T3) : {total_rescued:,}")
    print(f"  Estimated empty rows    : {total_queries - total_anchored - total_rescued:,}")

    # 5. Official Submission Validation on temp file
    print("\n5. Running official submission validator ...")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results_v6_tmp.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    ret = os.system(val_cmd)

    if ret == 0:
        print("\nValidator PASSED! Copying v6 temp file to output/matching_results.tsv ...")
        import shutil
        shutil.copyfile(TEMP_TSV, OUTPUT_TSV)
        print("output/matching_results.tsv is now updated and ready for upload!")
    else:
        print(f"\nValidator returned non-zero exit code: {ret}")
        print("Check output/matching_results_v6_tmp.tsv manually before replacing.")

    print(f"\nTotal Pipeline Time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
