"""
features.py  (v2 – 28 features)
=================================
Extracts string similarity + structural features for business entity candidate pairs.

Feature groups:
  Name (10): levenshtein, token_sort, token_set, jaro_winkler, partial, jaccard,
             char_bigram_jacc, len_diff, len_ratio, exact_match
  Address (7): levenshtein, token_sort, token_set, partial, jaccard,
               num_overlap, has_common_num
  Composite (3): comp_sort, comp_set, addr_len_ratio
  Pair-level (8): tfidf_sim, cand_rank, delta_top_tfidf, is_top1,
                  max_tfidf_in_block, name_tok_count_diff, is_s2, both_addr_empty
"""

import os
import sys
import re
import numpy as np
from typing import List, Dict, Tuple, Any

from rapidfuzz import fuzz, distance

# Ensure imports from local src work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address

RE_DIGITS = re.compile(r'\b\d+\b')


# ─────────────────────────────────────────────────────────────────────────────
# Small utility functions
# ─────────────────────────────────────────────────────────────────────────────
def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(RE_DIGITS.findall(text))


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))


def char_ngram_jaccard(s1: str, s2: str, n: int = 2) -> float:
    """Jaccard similarity on character n-grams."""
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    set1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    set2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    return len(set1 & set2) / float(len(set1 | set2))


def safe_len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


# ─────────────────────────────────────────────────────────────────────────────
# Core feature extraction for a single pair
# ─────────────────────────────────────────────────────────────────────────────
def extract_pair_features(
    s1_name_norm: str,
    s1_addr_norm: str,
    cand_name_norm: str,
    cand_addr_norm: str,
    tfidf_sim: float,
    cand_rank: int,
    delta_top_tfidf: float,
    max_tfidf_in_block: float,
    cand_id: str,
    n_candidates: int,
) -> Dict[str, float]:
    """
    Extract 28 features for a single candidate pair.
    """
    # ── Name features ─────────────────────────────────────────────────────────
    name_lev     = distance.Levenshtein.normalized_similarity(s1_name_norm, cand_name_norm)
    name_sort    = fuzz.token_sort_ratio(s1_name_norm, cand_name_norm) / 100.0
    name_set     = fuzz.token_set_ratio(s1_name_norm, cand_name_norm) / 100.0
    name_jw      = distance.JaroWinkler.similarity(s1_name_norm, cand_name_norm)
    name_partial = fuzz.partial_ratio(s1_name_norm, cand_name_norm) / 100.0

    s1_n_toks    = s1_name_norm.split()
    cand_n_toks  = cand_name_norm.split()
    name_jacc    = token_jaccard(s1_n_toks, cand_n_toks)
    name_bigram  = char_ngram_jaccard(s1_name_norm, cand_name_norm, n=2)
    name_len_diff  = float(abs(len(s1_name_norm) - len(cand_name_norm)))
    name_len_ratio = safe_len_ratio(s1_name_norm, cand_name_norm)
    name_exact   = 1.0 if s1_name_norm and s1_name_norm == cand_name_norm else 0.0
    name_tok_diff = float(abs(len(s1_n_toks) - len(cand_n_toks)))

    # ── Address features ──────────────────────────────────────────────────────
    addr_lev     = distance.Levenshtein.normalized_similarity(s1_addr_norm, cand_addr_norm)
    addr_sort    = fuzz.token_sort_ratio(s1_addr_norm, cand_addr_norm) / 100.0
    addr_set     = fuzz.token_set_ratio(s1_addr_norm, cand_addr_norm) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr_norm, cand_addr_norm) / 100.0
    s1_a_toks    = s1_addr_norm.split()
    cand_a_toks  = cand_addr_norm.split()
    addr_jacc    = token_jaccard(s1_a_toks, cand_a_toks)
    addr_len_ratio = safe_len_ratio(s1_addr_norm, cand_addr_norm)

    # Number / postal code matching
    s1_nums   = extract_numbers(s1_addr_norm)
    cand_nums = extract_numbers(cand_addr_norm)
    if s1_nums and cand_nums:
        num_overlap    = len(s1_nums & cand_nums) / float(len(s1_nums | cand_nums))
        has_common_num = 1.0 if (s1_nums & cand_nums) else 0.0
    elif not s1_nums and not cand_nums:
        num_overlap    = 0.5
        has_common_num = 0.5
    else:
        num_overlap    = 0.0
        has_common_num = 0.0

    # ── Composite features ────────────────────────────────────────────────────
    comp_sort = (name_sort * 0.6) + (addr_sort * 0.4)
    comp_set  = (name_set  * 0.6) + (addr_set  * 0.4)

    # ── Pair-level features ───────────────────────────────────────────────────
    is_s2          = 1.0 if cand_id.startswith('S2-') else 0.0
    is_top1        = 1.0 if cand_rank == 1 else 0.0
    both_addr_empty = 1.0 if (not s1_addr_norm and not cand_addr_norm) else 0.0

    return {
        # Name (11)
        'tfidf_sim':          tfidf_sim,
        'name_levenshtein':   name_lev,
        'name_token_sort':    name_sort,
        'name_token_set':     name_set,
        'name_jaro_winkler':  name_jw,
        'name_partial':       name_partial,
        'name_jaccard':       name_jacc,
        'name_bigram_jacc':   name_bigram,
        'name_len_diff':      name_len_diff,
        'name_len_ratio':     name_len_ratio,
        'name_exact':         name_exact,
        'name_tok_count_diff': name_tok_diff,
        # Address (7)
        'addr_levenshtein':   addr_lev,
        'addr_token_sort':    addr_sort,
        'addr_token_set':     addr_set,
        'addr_partial':       addr_partial,
        'addr_jaccard':       addr_jacc,
        'addr_num_overlap':   num_overlap,
        'addr_has_common_num': has_common_num,
        # Composite (3)
        'comp_sort':          comp_sort,
        'comp_set':           comp_set,
        'addr_len_ratio':     addr_len_ratio,
        # Pair-level (6)
        'cand_rank':          float(cand_rank),
        'delta_top_tfidf':    delta_top_tfidf,
        'max_tfidf_in_block': max_tfidf_in_block,
        'is_top1':            is_top1,
        'is_s2':              is_s2,
        'both_addr_empty':    both_addr_empty,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature column order (must be consistent across train + inference)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Name
    'tfidf_sim',
    'name_levenshtein',
    'name_token_sort',
    'name_token_set',
    'name_jaro_winkler',
    'name_partial',
    'name_jaccard',
    'name_bigram_jacc',
    'name_len_diff',
    'name_len_ratio',
    'name_exact',
    'name_tok_count_diff',
    # Address
    'addr_levenshtein',
    'addr_token_sort',
    'addr_token_set',
    'addr_partial',
    'addr_jaccard',
    'addr_num_overlap',
    'addr_has_common_num',
    # Composite
    'comp_sort',
    'comp_set',
    'addr_len_ratio',
    # Pair-level
    'cand_rank',
    'delta_top_tfidf',
    'max_tfidf_in_block',
    'is_top1',
    'is_s2',
    'both_addr_empty',
]

assert len(FEATURE_COLUMNS) == 28, f"Expected 28, got {len(FEATURE_COLUMNS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Batch feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_batch(
    candidate_dict: Dict[str, List[Tuple[str, float]]],
    s1_dict: Dict[str, Tuple[str, str]],
    cand_dict_records: Dict[str, Tuple[str, str]],
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Extracts 28 features for all candidate pairs.
    Returns: (list_of_s1_ids, list_of_cand_ids, feature_matrix [N x 28])
    """
    s1_ids_out:   List[str] = []
    cand_ids_out: List[str] = []
    rows:         List[List[float]] = []

    for s1_id, cands in candidate_dict.items():
        if not cands:
            continue
        s1_name, s1_addr = s1_dict.get(s1_id, ("", ""))
        top_tfidf        = cands[0][1]
        max_tfidf        = top_tfidf   # already sorted by score descending
        n_cands          = len(cands)

        for rank, (cand_id, sim) in enumerate(cands, 1):
            cand_name, cand_addr = cand_dict_records.get(cand_id, ("", ""))
            delta_top = top_tfidf - sim

            feats = extract_pair_features(
                s1_name_norm=s1_name,
                s1_addr_norm=s1_addr,
                cand_name_norm=cand_name,
                cand_addr_norm=cand_addr,
                tfidf_sim=sim,
                cand_rank=rank,
                delta_top_tfidf=delta_top,
                max_tfidf_in_block=max_tfidf,
                cand_id=cand_id,
                n_candidates=n_cands,
            )
            s1_ids_out.append(s1_id)
            cand_ids_out.append(cand_id)
            rows.append([feats[col] for col in FEATURE_COLUMNS])

    if not rows:
        return [], [], np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32)

    X = np.array(rows, dtype=np.float32)
    return s1_ids_out, cand_ids_out, X


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    res = extract_pair_features(
        s1_name_norm="orelees barbershop",
        s1_addr_norm="1795 westchester dr high point nc",
        cand_name_norm="orelees barbershop",
        cand_addr_norm="1795 westchester dr high point nc",
        tfidf_sim=0.95,
        cand_rank=1,
        delta_top_tfidf=0.0,
        max_tfidf_in_block=0.95,
        cand_id="S2-001",
        n_candidates=5,
    )
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
    print("All OK [OK]")

