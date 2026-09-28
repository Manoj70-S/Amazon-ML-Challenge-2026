"""
fast_inference_v4.py
====================
Ultra-High-Speed Pre-Parsed Streaming Inference Engine (v4)
for Amazon ML Challenge 2026 – Business Entity Resolution.

Architecture:
  1. Pre-parses entities once (strings, tokens, Metaphone, ZIP, street num, digits).
  2. Streaming chunk classification: 50,000 queries per chunk (~2.5M pairs in ~25s).
  3. Stacking Ensemble: LightGBM (2,000 trees) + XGBoost (1,000 trees).
  4. Decision Threshold: Calibrated optimal tau = 0.940 (Macro-F0.5 = 0.89924).
  5. End-to-End Runtime: ~12-15 minutes for all 1,732,544 test queries.
"""

import os
import sys
import time
import pickle
import json
import re
import gc
from typing import Dict, List, Tuple, Set, NamedTuple
import numpy as np
import pandas as pd
import jellyfish
from rapidfuzz import fuzz, distance

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address
from blocking import TFIDFBlocker, export_candidate_pairs_tsv

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

RE_DIGITS = re.compile(r'\b\d+\b')
RE_ZIP    = re.compile(r'\b\d{5,6}\b')


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
    name = normalize_name(raw_name)
    addr = normalize_address(raw_addr)
    n_toks = name.split()
    n_set = set(n_toks)
    w1_meta = jellyfish.metaphone(n_toks[0]) if n_toks else ""
    a_toks = addr.split()
    sn = RE_DIGITS.findall(addr)
    a_nums = set(sn)
    snum = sn[0] if sn else ""
    z = RE_ZIP.findall(addr)
    zip_code = z[-1] if z else ""
    return EntityRecord(name, addr, n_toks, n_set, w1_meta, a_toks, a_nums, zip_code, snum)


def char_ngram_jaccard(s1: str, s2: str, n: int = 2) -> float:
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    set1 = set(s1[i : i + n] for i in range(len(s1) - n + 1))
    set2 = set(s2[i : i + n] for i in range(len(s2) - n + 1))
    return len(set1 & set2) / float(len(set1 | set2))


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
    # Name
    name_lev = distance.Levenshtein.normalized_similarity(s1.name, c.name)
    name_sort = fuzz.token_sort_ratio(s1.name, c.name) / 100.0
    name_set = fuzz.token_set_ratio(s1.name, c.name) / 100.0
    name_jw = distance.JaroWinkler.similarity(s1.name, c.name)
    name_partial = fuzz.partial_ratio(s1.name, c.name) / 100.0

    if s1.n_set and c.n_set:
        inter = len(s1.n_set & c.n_set)
        name_jacc = inter / float(len(s1.n_set | c.n_set))
        tok_contain = inter / float(min(len(s1.n_set), len(c.n_set)))
    else:
        name_jacc = 0.0
        tok_contain = 0.0

    name_bi = char_ngram_jaccard(s1.name, c.name, n=2)
    name_tri = char_ngram_jaccard(s1.name, c.name, n=3)
    meta_match = 1.0 if (s1.w1_meta and s1.w1_meta == c.w1_meta) else 0.0
    name_len_diff = float(abs(len(s1.name) - len(c.name)))
    name_len_ratio = safe_len_ratio(s1.name, c.name)
    name_exact = 1.0 if s1.name and s1.name == c.name else 0.0
    name_tok_diff = float(abs(len(s1.n_toks) - len(c.n_toks)))

    # Address
    addr_lev = distance.Levenshtein.normalized_similarity(s1.addr, c.addr)
    addr_sort = fuzz.token_sort_ratio(s1.addr, c.addr) / 100.0
    addr_set = fuzz.token_set_ratio(s1.addr, c.addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1.addr, c.addr) / 100.0

    if s1.a_toks and c.a_toks:
        set1, set2 = set(s1.a_toks), set(c.a_toks)
        addr_jacc = len(set1 & set2) / float(len(set1 | set2))
    else:
        addr_jacc = 0.0

    if s1.a_nums and c.a_nums:
        num_overlap = len(s1.a_nums & c.a_nums) / float(len(s1.a_nums | c.a_nums))
        has_common_num = 1.0 if (s1.a_nums & c.a_nums) else 0.0
    elif not s1.a_nums and not c.a_nums:
        num_overlap = 0.5
        has_common_num = 0.5
    else:
        num_overlap = 0.0
        has_common_num = 0.0

    if s1.zip_code and c.zip_code:
        zip_match = 1.0 if s1.zip_code == c.zip_code else 0.0
    else:
        zip_match = 0.5

    if s1.snum and c.snum:
        snum_match = 1.0 if s1.snum == c.snum else 0.0
    else:
        snum_match = 0.5

    addr_len_ratio = safe_len_ratio(s1.addr, c.addr)
    both_addr_empty = 1.0 if (not s1.addr and not c.addr) else 0.0

    comp_sort = (name_sort * 0.6) + (addr_sort * 0.4)
    comp_set = (name_set * 0.6) + (addr_set * 0.4)

    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    is_top1 = 1.0 if rank == 1 else 0.0
    score_decay = 1.0 / (1.0 + 0.05 * rank)

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
    optimal_threshold = 0.940

    print("====================================================================")
    print(f"  Ultra-Fast Pre-Parsed v4 Test Inference (Threshold = {optimal_threshold:.3f})")
    print("====================================================================")

    # 1. Load Models
    print("1. Loading v4 Stacking Ensemble models …")
    t0 = time.time()
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lgb_model.set_params(n_jobs=12)
    xgb_model.set_params(n_jobs=12)
    print(f"   Models loaded in {time.time()-t0:.1f}s (12 CPU threads enabled)")

    # 2. Load Source 1 entities from cache
    print("\n2. Loading pre-parsed Source 1 entities from cache/s1_parsed.pkl …")
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "s1_parsed.pkl"), "rb") as f:
        raw_s1 = pickle.load(f)
    s1_parsed: Dict[str, EntityRecord] = {k: EntityRecord(*v) for k, v in raw_s1.items()}
    del raw_s1
    n_total_queries = len(s1_parsed)
    print(f"   Source 1 loaded in {time.time()-t0:.1f}s | {n_total_queries:,} entities in memory")

    # 3. Load Candidate Data (Source 2 & Source 3)
    print("\n3. Loading Source 2 & Source 3 candidate data …")
    t0 = time.time()
    s2_df = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    print(f"   Loaded {len(cand_df):,} candidates in {time.time()-t0:.1f}s")

    # 4. Build Candidate Raw Lookup Arrays
    print("\n4. Building candidate lookup arrays …")
    t0 = time.time()
    cand_ids = cand_df["entity_id"].values
    cand_names_raw = cand_df["business_name"].values
    cand_addrs_raw = cand_df["business_address"].values
    cand_id_map: Dict[str, int] = {cid: idx for idx, cid in enumerate(cand_ids)}
    del cand_df
    print(f"   Candidate index ready in {time.time()-t0:.1f}s | {len(cand_id_map):,} candidates")

    # Candidate parse cache
    cand_parsed_cache: Dict[str, EntityRecord] = {}

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

    # 5. Streaming Inference Loop with Seamless Resume and Bounded Memory
    CHUNK_SIZE = 15_000  # 15k queries per chunk to keep numpy memory small and bounded

    already_processed = 0
    if os.path.exists(MATCHING_TSV):
        with open(MATCHING_TSV, "r", encoding="utf-8") as f:
            already_processed = sum(1 for _ in f) - 1  # exclude header
        if already_processed < 0:
            already_processed = 0

    mode = "a" if already_processed > 0 else "w"
    total_queries_processed = already_processed
    total_positives = 0
    total_singletons = 0
    t_stream = time.time()

    print(f"\n5. Streaming Classification across {n_total_queries:,} queries …")
    if already_processed > 0:
        print(f"   RESUMING from query {already_processed:,} ({already_processed/n_total_queries*100:.1f}% already done)!")

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as in_f, \
         open(MATCHING_TSV, mode, encoding="utf-8") as out_f:
        
        if already_processed == 0:
            out_f.write("source1_entity_id\tmatched_entity_ids\n")
        
        header = in_f.readline()
        if already_processed > 0:
            print(f"   Fast-forwarding candidate file past {already_processed:,} queries …")
            for _ in range(already_processed):
                in_f.readline()
            print(f"   Fast-forward complete. Resuming inference now …")

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
                s1_rec = s1_parsed.get(sid)
                if s1_rec is None:
                    continue
                n_cands = len(cids)
                max_sim = 1.0

                for rank, cid in enumerate(cids):
                    c_rec = get_cand_record(cid)
                    if rank >= 10:
                        # Safe skip: candidate shares 0 tokens, 0 prefix, and 0 metaphone
                        if (
                            not (s1_rec.n_set & c_rec.n_set)
                            and s1_rec.name[:3] != c_rec.name[:3]
                            and not (s1_rec.w1_meta and s1_rec.w1_meta == c_rec.w1_meta)
                        ):
                            continue

                    sim = 1.0 / (1.0 + 0.05 * rank)
                    row = extract_fast_features(
                        s1_rec, c_rec, sim, rank + 1, max_sim - sim, max_sim, cid
                    )
                    chunk_s1_feat.append(sid)
                    chunk_cand_feat.append(cid)
                    chunk_rows.append(row)

            # Classify chunk
            accepted_in_chunk: Dict[str, List[str]] = {sid: [] for sid in chunk_s1_ids}

            if len(chunk_rows) > 0:
                X_chunk = np.array(chunk_rows, dtype=np.float32)
                p_lgb = lgb_model.predict_proba(X_chunk)[:, 1]
                p_xgb = xgb_model.predict_proba(X_chunk)[:, 1]
                p_ens = 0.6 * p_lgb + 0.4 * p_xgb

                for sid, cid, prob in zip(chunk_s1_feat, chunk_cand_feat, p_ens):
                    if prob >= optimal_threshold:
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
            out_f.flush()

            total_queries_processed += len(chunk_s1_ids)

            # Memory management: purge candidate cache if too large to prevent leaks
            if len(cand_parsed_cache) > 80_000:
                cand_parsed_cache.clear()
            del chunk_rows, chunk_s1_feat, chunk_cand_feat, chunk_s1_ids, accepted_in_chunk
            if 'X_chunk' in locals():
                del X_chunk
            gc.collect()

            elapsed = time.time() - t_stream
            queries_this_run = total_queries_processed - already_processed
            rate = queries_this_run / max(elapsed, 1e-9)
            eta_min = (n_total_queries - total_queries_processed) / max(rate, 1e-9) / 60.0
            print(f"  Processed {total_queries_processed:,}/{n_total_queries:,} queries "
                  f"({total_queries_processed/n_total_queries*100:.1f}%) | "
                  f"{rate:.0f} q/s | Matches: {total_positives:,} | Singletons: {total_singletons:,} | ETA {eta_min:.1f} min")

    print(f"\n=== Inference Complete in {(time.time()-t_start)/60:.1f} min ===")
    print(f"  Total S1 Entities: {n_total_queries:,}")
    print(f"  Matches (this run): {total_positives:,}")
    print(f"  Singletons (this run): {total_singletons:,}")

    # 6. Validate Output
    print("\nRunning official submission validator …")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    os.system(val_cmd)


if __name__ == "__main__":
    main()
