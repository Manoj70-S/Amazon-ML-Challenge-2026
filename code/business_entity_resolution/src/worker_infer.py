"""
worker_infer.py
===============
Ultra-Lean Standalone Worker Process for Parallel Partition Inference.
Memory Footprint: < 350 MB RAM per worker.
Usage: python worker_infer.py <part_idx> <cand_part_file> <match_part_file> <threshold>
"""

import os
import sys
import time
import pickle
import json
import re
from typing import Dict, List, Tuple, Set, NamedTuple
import numpy as np
import jellyfish
from rapidfuzz import fuzz, distance

# Force unbuffered stdout
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address

TEST_DIR   = os.path.join(PROJECT_ROOT, "dataset", "test")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_v4.pkl")

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


def run_worker(part_idx: int, cand_file: str, out_file: str, threshold: float = 0.940):
    t0 = time.time()
    print(f"[Worker {part_idx:02d}] Initializing partition: {cand_file} -> {out_file}")

    # 1. Load models
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    lgb_model.set_params(n_jobs=2)
    xgb_model.set_params(n_jobs=2)

    # 2. Read query IDs in this partition
    with open(cand_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        with open(out_file, "w", encoding="utf-8") as f:
            pass
        return

    needed_s1 = set()
    needed_cands = set()
    for line in lines:
        parts = line.split("\t")
        needed_s1.add(parts[0])
        if len(parts) > 1 and parts[1]:
            for c in parts[1].split(","):
                needed_cands.add(c)

    print(f"[Worker {part_idx:02d}] Target Queries: {len(needed_s1):,} | Needed Candidates: {len(needed_cands):,}")

    # 3. Stream Source 1 line-by-line (Zero memory spike)
    s1_parsed: Dict[str, EntityRecord] = {}
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in needed_s1:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                s1_parsed[parts[0]] = parse_entity(name, addr)

    # 4. Stream Candidates line-by-line
    cand_parsed: Dict[str, EntityRecord] = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        with open(os.path.join(TEST_DIR, fname), "r", encoding="utf-8") as f:
            _ = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] in needed_cands:
                    name = parts[1] if len(parts) > 1 else ""
                    addr = parts[2] if len(parts) > 2 else ""
                    cand_parsed[parts[0]] = parse_entity(name, addr)

    print(f"[Worker {part_idx:02d}] Loaded & Pre-parsed in {time.time()-t0:.1f}s | Beginning classification …")

    # 5. Classify in chunks of 10,000 queries
    CHUNK_SIZE = 10_000
    n_queries = len(lines)
    total_processed = 0
    total_matches = 0
    t_stream = time.time()

    with open(out_file, "w", encoding="utf-8") as out_f:
        for ch_start in range(0, n_queries, CHUNK_SIZE):
            ch_lines = lines[ch_start : ch_start + CHUNK_SIZE]
            chunk_s1_ids: List[str] = []
            chunk_s1_feat: List[str] = []
            chunk_cand_feat: List[str] = []
            chunk_rows: List[List[float]] = []

            for line in ch_lines:
                parts = line.split("\t")
                sid = parts[0]
                chunk_s1_ids.append(sid)
                if len(parts) < 2 or not parts[1]:
                    continue
                cids = parts[1].split(",")
                s1_rec = s1_parsed.get(sid)
                if s1_rec is None:
                    continue
                max_sim = 1.0
                for rank, cid in enumerate(cids):
                    c_rec = cand_parsed.get(cid)
                    if c_rec is None:
                        continue
                    sim = 1.0 / (1.0 + 0.05 * rank)
                    row = extract_fast_features(s1_rec, c_rec, sim, rank + 1, max_sim - sim, max_sim, cid)
                    chunk_s1_feat.append(sid)
                    chunk_cand_feat.append(cid)
                    chunk_rows.append(row)

            accepted_in_chunk: Dict[str, List[str]] = {sid: [] for sid in chunk_s1_ids}
            if len(chunk_rows) > 0:
                X_chunk = np.array(chunk_rows, dtype=np.float32)
                p_lgb = lgb_model.predict_proba(X_chunk)[:, 1]
                p_xgb = xgb_model.predict_proba(X_chunk)[:, 1]
                p_ens = 0.6 * p_lgb + 0.4 * p_xgb

                for sid, cid, prob in zip(chunk_s1_feat, chunk_cand_feat, p_ens):
                    if prob >= threshold:
                        accepted_in_chunk[sid].append(cid)

            for sid in chunk_s1_ids:
                matches = accepted_in_chunk.get(sid, [])
                if matches:
                    total_matches += 1
                    out_f.write(f"{sid}\t{','.join(matches)}\n")
                else:
                    out_f.write(f"{sid}\t\n")

            out_f.flush()
            total_processed += len(chunk_s1_ids)
            rate = total_processed / max(time.time() - t_stream, 1e-9)
            print(f"[Worker {part_idx:02d}] {total_processed:,}/{n_queries:,} ({total_processed/n_queries*100:.1f}%) | {rate:.0f} q/s")

    print(f"[Worker {part_idx:02d}] Finished {n_queries:,} queries in {(time.time()-t0)/60:.1f} min | Matches: {total_matches:,}")


if __name__ == "__main__":
    part_idx = int(sys.argv[1])
    cand_file = sys.argv[2]
    out_file = sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.940
    run_worker(part_idx, cand_file, out_file, threshold)
