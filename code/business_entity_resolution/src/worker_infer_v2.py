"""
worker_infer_v2.py
==================
Fixed worker process. Loads pre-built pickle files instead of streaming TSVs.
Memory per worker: ~600 MB (shared pickle loaded read-only from disk, OS page cache).

Usage:
  python worker_infer_v2.py <part_idx> <cand_part_file> <match_part_file> <threshold>
"""

import os, sys, re, time, pickle
from typing import Dict, List, Set, NamedTuple
import numpy as np
import jellyfish
from rapidfuzz import fuzz, distance as rf_distance
import lightgbm as lgb

# Force unbuffered stdout
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address

CACHE_DIR      = os.path.join(PROJECT_ROOT, "cache")
MODEL_DIR      = os.path.join(PROJECT_ROOT, "models")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_v4.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_v4.pkl")
S1_PKL         = os.path.join(CACHE_DIR, "s1_parsed.pkl")
CAND_PKL       = os.path.join(CACHE_DIR, "cand_parsed.pkl")

RE_DIGITS = re.compile(r'\b\d+\b')
RE_ZIP    = re.compile(r'\b\d{5,6}\b')

FEATURE_COLUMNS = [
    "name_tfidf_sim", "name_levenshtein", "name_token_sort", "name_token_set",
    "name_jaro_winkler", "name_partial", "name_jaccard", "name_bigram_jacc",
    "name_trigram_jacc", "tok_containment", "metaphone_match", "name_len_diff",
    "name_len_ratio", "name_exact", "tok_count_diff",
    "addr_levenshtein", "addr_token_sort", "addr_token_set", "addr_partial",
    "addr_jaccard", "addr_num_overlap", "addr_has_common_num",
    "addr_zip_match", "addr_street_num_match", "addr_len_ratio",
    "both_addr_empty",
    "comp_sort", "comp_set",
    "cand_rank", "delta_top_tfidf", "max_tfidf_in_block",
    "is_top1", "is_s2", "score_decay",
]


class EntityRecord(NamedTuple):
    name:    str
    addr:    str
    n_toks:  list
    n_set:   set
    w1_meta: str
    a_toks:  list
    a_nums:  set
    zip_code: str
    snum:    str


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


def token_jaccard(toks1: list, toks2: list) -> float:
    s1, s2 = set(toks1), set(toks2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def extract_features(s1: EntityRecord, s2: EntityRecord,
                     rank: int, top_tfidf: float, max_tfidf: float,
                     is_top1: bool, is_s2_cand: bool, tfidf_score: float) -> list:
    # Name features
    n1, n2  = s1.name, s2.name
    lev_n   = 1.0 - rf_distance.Levenshtein.normalized_distance(n1, n2)
    ts_n    = fuzz.token_sort_ratio(n1, n2) / 100.0
    tset_n  = fuzz.token_set_ratio(n1, n2) / 100.0
    jw_n    = jellyfish.jaro_winkler_similarity(n1, n2)
    part_n  = fuzz.partial_ratio(n1, n2) / 100.0
    jacc_n  = token_jaccard(s1.n_toks, s2.n_toks)
    big_n   = char_ngram_jaccard(n1, n2, 2)
    tri_n   = char_ngram_jaccard(n1, n2, 3)
    toks1s  = s1.n_set
    toks2s  = s2.n_set
    contain = (len(toks1s & toks2s) / len(toks1s)) if toks1s else 0.0
    meta_m  = 1.0 if s1.w1_meta and s1.w1_meta == s2.w1_meta else 0.0
    len1, len2 = len(n1), len(n2)
    len_diff = abs(len1 - len2)
    len_rat  = safe_len_ratio(n1, n2)
    exact_n  = 1.0 if n1 == n2 else 0.0
    tc_diff  = abs(len(s1.n_toks) - len(s2.n_toks))

    # Address features
    a1, a2  = s1.addr, s2.addr
    lev_a   = 1.0 - rf_distance.Levenshtein.normalized_distance(a1, a2)
    ts_a    = fuzz.token_sort_ratio(a1, a2) / 100.0
    tset_a  = fuzz.token_set_ratio(a1, a2) / 100.0
    part_a  = fuzz.partial_ratio(a1, a2) / 100.0
    jacc_a  = token_jaccard(s1.a_toks, s2.a_toks)
    common_nums = s1.a_nums & s2.a_nums
    num_ov  = len(common_nums)
    has_cn  = 1.0 if common_nums else 0.0
    zip_m   = 1.0 if s1.zip_code and s1.zip_code == s2.zip_code else 0.0
    snum_m  = 1.0 if s1.snum and s1.snum == s2.snum else 0.0
    len_a_r = safe_len_ratio(a1, a2)
    both_empty = 1.0 if not a1 and not a2 else 0.0

    # Composite
    comp_s1 = s1.name + " " + s1.addr
    comp_s2 = s2.name + " " + s2.addr
    comp_sort = fuzz.token_sort_ratio(comp_s1, comp_s2) / 100.0
    comp_set  = fuzz.token_set_ratio(comp_s1, comp_s2) / 100.0

    # Ranking signals
    cand_rank   = float(rank)
    delta_top   = top_tfidf - tfidf_score if rank > 0 else 0.0
    decay       = 1.0 / (1.0 + rank)

    return [
        tfidf_score, lev_n, ts_n, tset_n, jw_n, part_n, jacc_n, big_n, tri_n,
        contain, meta_m, float(len_diff), len_rat, exact_n, float(tc_diff),
        lev_a, ts_a, tset_a, part_a, jacc_a, float(num_ov), has_cn,
        zip_m, snum_m, len_a_r, both_empty,
        comp_sort, comp_set,
        cand_rank, delta_top, float(max_tfidf),
        float(is_top1), float(is_s2_cand), decay,
    ]


def run_worker(part_idx: int, cand_file: str, out_file: str, threshold: float):
    t0 = time.time()
    print(f"[Worker {part_idx:02d}] Loading pickles ...")

    def expand(raw_dict):
        """Convert dict[id -> (name_str, addr_str)] to dict[id -> EntityRecord]."""
        out = {}
        for eid, val in raw_dict.items():
            if isinstance(val, EntityRecord):
                out[eid] = val
            elif isinstance(val, tuple) and len(val) == 2:
                # Minimal format: (norm_name, norm_addr) strings
                name, addr = val
                n_toks = name.split()
                n_set  = set(n_toks)
                w1_meta = jellyfish.metaphone(n_toks[0]) if n_toks else ""
                a_toks = addr.split()
                sn     = RE_DIGITS.findall(addr)
                a_nums = set(sn)
                snum   = sn[0] if sn else ""
                z      = RE_ZIP.findall(addr)
                zip_code = z[-1] if z else ""
                out[eid] = EntityRecord(name, addr, n_toks, n_set, w1_meta,
                                        a_toks, a_nums, zip_code, snum)
            else:
                # Full 9-tuple from old format
                out[eid] = EntityRecord(*val)
        return out

    with open(S1_PKL, "rb") as f:
        s1_parsed: Dict[str, EntityRecord] = expand(pickle.load(f))
    print(f"[Worker {part_idx:02d}] S1 loaded+expanded: {len(s1_parsed):,} entities")

    with open(CAND_PKL, "rb") as f:
        cand_parsed: Dict[str, EntityRecord] = expand(pickle.load(f))
    print(f"[Worker {part_idx:02d}] Cand loaded+expanded: {len(cand_parsed):,} entities")

    print(f"[Worker {part_idx:02d}] Loading models ...")
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)  # XGBClassifier (sklearn API)

    print(f"[Worker {part_idx:02d}] Processing {cand_file} ...")
    rows_processed = 0
    pairs_matched  = 0
    batch_size     = 500  # pairs per ML batch

    with open(cand_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        header = fin.readline().strip()  # skip or read header
        # cand_part files have no header (just data lines), so we may have read data
        # Let's check: if header looks like a real header, skip; otherwise process it
        if header.startswith("source1_entity_id"):
            pass  # it was a real header, skip it
        else:
            # It was a data line — process it
            lines_to_process = [header]
        
        # Reset: re-read properly
        fin.seek(0)
        first_line = fin.readline().strip()
        if first_line.startswith("source1_entity_id"):
            pass  # true header, already consumed
        else:
            fin.seek(0)  # rewind, no header

        batch_feats  = []
        batch_meta   = []  # (s1_id, s2_id)

        def flush_batch(fout, batch_feats, batch_meta, threshold, lgb_model, xgb_model, pairs_matched):
            if not batch_feats:
                return pairs_matched
            X = np.array(batch_feats, dtype=np.float32)
            p_lgb = lgb_model.predict_proba(X)[:, 1]
            p_xgb = xgb_model.predict_proba(X)[:, 1]
            p_ens = 0.60 * p_lgb + 0.40 * p_xgb
            for (s1_id, s2_id), score in zip(batch_meta, p_ens):
                if score >= threshold:
                    fout.write(f"{s1_id}\t{s2_id}\n")
                    pairs_matched += 1
            batch_feats.clear()
            batch_meta.clear()
            return pairs_matched

        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            s1_id   = parts[0].strip()
            cand_ids = [c.strip() for c in parts[1].split(",") if c.strip()]
            if not cand_ids or s1_id not in s1_parsed:
                rows_processed += 1
                continue

            s1_rec = s1_parsed[s1_id]
            is_s2  = False  # default; will be set per candidate

            tfidf_scores = [1.0 / (1.0 + i) for i in range(len(cand_ids))]
            top_tfidf = tfidf_scores[0] if tfidf_scores else 1.0
            max_tfidf = top_tfidf

            for rank, cid in enumerate(cand_ids):
                if cid not in cand_parsed:
                    continue
                c_rec    = cand_parsed[cid]
                is_s2_c  = cid.startswith("S2-")
                is_top1  = (rank == 0)
                tf_score = tfidf_scores[rank]

                feats = extract_features(
                    s1_rec, c_rec,
                    rank, top_tfidf, max_tfidf,
                    is_top1, is_s2_c, tf_score
                )
                batch_feats.append(feats)
                batch_meta.append((s1_id, cid))

                if len(batch_feats) >= batch_size:
                    pairs_matched = flush_batch(fout, batch_feats, batch_meta,
                                                threshold, lgb_model, xgb_model, pairs_matched)

            rows_processed += 1
            if rows_processed % 5000 == 0:
                pairs_matched = flush_batch(fout, batch_feats, batch_meta,
                                            threshold, lgb_model, xgb_model, pairs_matched)
                elapsed = time.time() - t0
                rate    = rows_processed / elapsed
                print(f"[Worker {part_idx:02d}] {rows_processed:,} queries | "
                      f"{pairs_matched:,} matches | {rate:.0f} q/s")

        # Final flush
        pairs_matched = flush_batch(fout, batch_feats, batch_meta,
                                    threshold, lgb_model, xgb_model, pairs_matched)

    elapsed = time.time() - t0
    print(f"[Worker {part_idx:02d}] DONE: {rows_processed:,} queries, "
          f"{pairs_matched:,} matches, {elapsed:.1f}s total")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python worker_infer_v2.py <part_idx> <cand_part_file> <match_part_file> <threshold>")
        sys.exit(1)

    part_idx  = int(sys.argv[1])
    cand_file = sys.argv[2]
    out_file  = sys.argv[3]
    threshold = float(sys.argv[4])

    run_worker(part_idx, cand_file, out_file, threshold)
