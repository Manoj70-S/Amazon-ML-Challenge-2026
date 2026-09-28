"""
prebuild_pickles_v3.py  — minimal-store, fully vectorized
==========================================================
Stores ONLY (norm_name, norm_addr) strings in pickles.
Workers compute n_toks/n_set/a_nums on-the-fly per pair.

This eliminates the ~1000s Python dict-building loop entirely.
Prebuild time: ~3-5 min total for all 3 sources.
"""

import os, sys, re, time, pickle
from typing import Set
import pandas as pd

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import fast_clean_series, LEGAL_MAPPINGS, ADDRESS_MAPPINGS, US_STATE_MAPPINGS

TEST_DIR  = os.path.join(PROJECT_ROOT, "dataset", "test")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CAND_FILE = os.path.join(PROJECT_ROOT, "output", "candidate_pairs.tsv")


def vect_names(s: pd.Series) -> pd.Series:
    t = fast_clean_series(s)
    for p, r in LEGAL_MAPPINGS:
        t = t.str.replace(p, r, regex=True)
    t = t.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    return t.str.replace(r'\s+', ' ', regex=True).str.strip()


def vect_addrs(s: pd.Series) -> pd.Series:
    t = fast_clean_series(s)
    for p, r in ADDRESS_MAPPINGS:
        t = t.str.replace(p, r, regex=True)
    for p, r in US_STATE_MAPPINGS:
        t = t.str.replace(p, r, regex=True)
    t = t.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    return t.str.replace(r'\s+', ' ', regex=True).str.strip()


def main():
    t0 = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)

    s1_pkl   = os.path.join(CACHE_DIR, "s1_parsed.pkl")
    cand_pkl = os.path.join(CACHE_DIR, "cand_parsed.pkl")

    # ------------------------------------------------------------------ #
    # 1. S1: vectorized normalize → store as dict[id -> (name_str, addr_str)]
    # ------------------------------------------------------------------ #
    print(f"[{time.time()-t0:.0f}s] Loading test_source1.tsv ...", flush=True)
    df1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    print(f"  {len(df1):,} rows", flush=True)

    print(f"[{time.time()-t0:.0f}s] Normalizing S1 names+addrs (vectorized)...", flush=True)
    n1 = vect_names(df1["business_name"])
    a1 = vect_addrs(df1["business_address"])
    print(f"[{time.time()-t0:.0f}s] Building S1 dict...", flush=True)
    s1_parsed = dict(zip(df1["entity_id"], zip(n1, a1)))
    del df1, n1, a1

    print(f"[{time.time()-t0:.0f}s] Saving s1_parsed.pkl ({len(s1_parsed):,} entries)...", flush=True)
    with open(s1_pkl, "wb") as f:
        pickle.dump(s1_parsed, f, protocol=pickle.HIGHEST_PROTOCOL)
    del s1_parsed
    print(f"[{time.time()-t0:.0f}s] S1 done.", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Collect needed candidate IDs                                      #
    # ------------------------------------------------------------------ #
    print(f"\n[{time.time()-t0:.0f}s] Scanning candidate_pairs.tsv ...", flush=True)
    needed: Set[str] = set()
    with open(CAND_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                for cid in parts[1].split(","):
                    cid = cid.strip()
                    if cid:
                        needed.add(cid)
    print(f"  {len(needed):,} unique candidate IDs in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ #
    # 3. S2 + S3: filter + vectorized normalize                           #
    # ------------------------------------------------------------------ #
    cand_parsed = {}
    for src_num in [2, 3]:
        src_file = os.path.join(TEST_DIR, f"test_source{src_num}.tsv")
        print(f"\n[{time.time()-t0:.0f}s] Loading {os.path.basename(src_file)} ...", flush=True)
        df = pd.read_csv(src_file, sep="\t", dtype=str).fillna("")
        print(f"  {len(df):,} rows loaded", flush=True)

        mask = df["entity_id"].isin(needed)
        df   = df[mask].copy()
        print(f"  {len(df):,} needed rows", flush=True)

        print(f"[{time.time()-t0:.0f}s] Normalizing S{src_num}...", flush=True)
        nn = vect_names(df["business_name"])
        aa = vect_addrs(df["business_address"])
        chunk = dict(zip(df["entity_id"], zip(nn, aa)))
        cand_parsed.update(chunk)
        del df, nn, aa, chunk
        print(f"  Running total: {len(cand_parsed):,} candidates", flush=True)

    print(f"\n[{time.time()-t0:.0f}s] Saving cand_parsed.pkl ({len(cand_parsed):,} entries)...", flush=True)
    with open(cand_pkl, "wb") as f:
        pickle.dump(cand_parsed, f, protocol=pickle.HIGHEST_PROTOCOL)
    del cand_parsed

    elapsed = time.time() - t0
    print(f"\n✅ Prebuild done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
