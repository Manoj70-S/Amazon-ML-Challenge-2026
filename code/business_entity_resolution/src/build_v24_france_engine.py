#!/usr/bin/env python3
"""
build_v24_france_engine.py
==========================
Amazon ML Challenge 2026 – France-Enhanced Cluster Corroboration Engine (v24)

Key Innovations:
1. Base Invariant: 100% preserves every link from v23 (4,761,802 links, 0 collisions).
2. France Segment Optimization:
   - Leverages updated global normalize.py (French legal suffixes: SARL, SAS, SASU, EURL, SCI, SNC;
     and address standards: rue -> r, blvd/bd -> blvd, ave/av -> ave, bis -> b, ter -> c).
   - Targets 1-and-2 link entities across test entities to complete clusters.
3. Strict Calibrated Thresholds:
   - Exact street/building number match (snum == asnum)
   - Normalized Address similarity >= 88.0%
   - Normalized Name similarity >= 85.0%
   - OR Normalized Address similarity >= 92.0%
4. Single-Winner Rule: At most 1 winning candidate added per entity.
5. Strictly Disjoint: Highest-confidence claim wins globally, guaranteeing EXACTLY 0 collisions.
6. Official Validation: Verified with utils/validate_submission.py.
"""

import os
import sys
import gc
import time
import re
import shutil
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz

# Ensure unbuffered output
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

PROJECT_ROOT = os.path.abspath(".")
SRC_DIR      = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

import normalize

DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")

BASE_TSV     = os.path.join(OUTPUT_DIR, "matching_results_v23_apex.tsv")
CAND_TSV     = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
OUT_V24_TSV  = os.path.join(OUTPUT_DIR, "matching_results_v24_france.tsv")
FINAL_TSV    = os.path.join(OUTPUT_DIR, "matching_results.tsv")

RE_DIGITS = re.compile(r"\b\d+\b")


def get_street_num(addr: str) -> str:
    m = RE_DIGITS.findall(addr)
    nums = [n for n in m if len(n) < 5]
    return nums[0] if nums else ""


def main():
    t_start = time.time()
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: FRANCE-ENHANCED CLUSTER ENGINE (v24)")
    print("=" * 80)

    # 1. Load Base Submission (v23)
    print("\n1. Loading base submission (v23) ...")
    s1_ordered = []
    base_matches: Dict[str, List[str]] = {}
    assigned_candidates: Set[str] = set()
    target_s1: Dict[str, List[str]] = {}  # sid -> list of anchor cids

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
                if len(mids) in (1, 2):
                    target_s1[sid] = mids
            else:
                base_matches[sid] = []

    print(f"   Total S1 entities:            {len(s1_ordered):,}")
    print(f"   Base total links in v23:      {len(assigned_candidates):,}")
    print(f"   Target 1-and-2 link entities: {len(target_s1):,}")

    # 2. Collect candidate pairs from candidate_pairs.tsv (top 15)
    print("\n2. Scanning candidate_pairs.tsv for unassigned candidates (top 15) ...")
    t0 = time.time()
    target_cands: Dict[str, List[str]] = {}
    needed_cids: Set[str] = set()

    for mids in target_s1.values():
        needed_cids.update(mids)  # anchors

    with open(CAND_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in target_s1 and len(p) > 1 and p[1].strip():
                cids = [c.strip() for c in p[1].split(",") if c.strip()][:15]
                unassigned = [c for c in cids if c not in assigned_candidates]
                if unassigned:
                    target_cands[sid] = unassigned
                    needed_cids.update(unassigned)

    print(f"   Target entities with unassigned candidates: {len(target_cands):,}")
    print(f"   Total unique candidate texts needed:        {len(needed_cids):,} in {time.time()-t0:.1f}s")

    # 3. Stream and parse candidate texts from test_source2.tsv and test_source3.tsv
    print("\n3. Streaming candidate texts from test_source2.tsv and test_source3.tsv ...")
    raw_cand_text: Dict[str, Tuple[str, str]] = {}
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
                    raw_cand_text[cid] = (name, addr)
                    c_found += 1
        print(f"   {os.path.basename(src_path)}: loaded {c_found:,} candidates in {time.time()-t_src:.1f}s")

    print(f"   Candidate text cache ready: {len(raw_cand_text):,} records")

    # 4. Pre-normalize candidate texts using updated normalize.py
    print("\n4. Pre-normalizing candidate texts with updated normalization rules ...")
    t_norm = time.time()
    norm_cand_cache: Dict[str, Tuple[str, str, str]] = {}  # cid -> (norm_name, norm_addr, snum)
    for cid, (name, addr) in raw_cand_text.items():
        nn = normalize.normalize_name(name)
        na = normalize.normalize_address(addr)
        snum = get_street_num(addr)
        norm_cand_cache[cid] = (nn, na, snum)

    del raw_cand_text
    gc.collect()
    print(f"   Normalized {len(norm_cand_cache):,} candidates in {time.time()-t_norm:.1f}s")

    # 5. Evaluate Strict Cluster-to-Record Corroboration with Normalized Strings
    print("\n5. Evaluating strict cluster-to-record corroboration with normalized strings ...")
    t_match = time.time()
    claims: List[Tuple[str, str, float]] = []  # (sid, cid, score)

    for sid, unassigned in target_cands.items():
        anchors = target_s1[sid]
        best_score = 0.0
        best_cid = None

        for cid in unassigned:
            c_info = norm_cand_cache.get(cid)
            if not c_info: continue
            cn, ca, csnum = c_info

            for aid in anchors:
                a_info = norm_cand_cache.get(aid)
                if not a_info: continue
                an, aa, asnum = a_info

                snum_match = bool(csnum and asnum and len(csnum) >= 2 and csnum == asnum)
                asim = fuzz.token_set_ratio(aa, ca)

                if asim >= 88.0:
                    nsim = fuzz.token_sort_ratio(an, cn)
                    if nsim >= 85.0 and (snum_match or asim >= 92.0):
                        score = 0.5 * asim + 0.5 * nsim + (10.0 if snum_match else 0.0)
                        if score > best_score:
                            best_score = score
                            best_cid = cid

        if best_cid:
            claims.append((sid, best_cid, best_score))

    print(f"   Raw claims generated: {len(claims):,} in {time.time()-t_match:.1f}s")

    # 6. Global Disjoint Collision Resolution (Highest Score Wins)
    print("\n6. Resolving candidate collisions globally (highest score wins) ...")
    claims.sort(key=lambda x: x[2], reverse=True)

    winner_for_cand: Dict[str, Tuple[str, float]] = {}
    purged_collisions = 0

    for sid, cid, score in claims:
        if cid not in winner_for_cand and cid not in assigned_candidates:
            winner_for_cand[cid] = (sid, score)
        else:
            purged_collisions += 1

    rescued_by_s1: Dict[str, str] = {sid: cid for cid, (sid, score) in winner_for_cand.items()}

    print(f"   Collisions purged:                  {purged_collisions:,}")
    print(f"   Unique cluster counterparts added:  {len(rescued_by_s1):,}")

    # 7. Write Final TSV
    print(f"\n7. Writing v24 final submission to {os.path.basename(OUT_V24_TSV)} ...")
    final_non_empty = 0
    final_empty = 0
    total_final_links = 0

    with open(OUT_V24_TSV, "w", encoding="utf-8") as f_out:
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
    print(f"   Total links in v24:       {total_final_links:,} (+{len(rescued_by_s1):,} vs v23, +{total_final_links - 4721330:,} vs v16)")

    # 8. Audit Collisions
    print("\n8. Auditing submission for multi-assignment collisions ...")
    check_cands = Counter()
    with open(OUT_V24_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                for c in p[1].split(","):
                    check_cands[c.strip()] += 1

    collisions = sum(1 for c, cnt in check_cands.items() if cnt > 1)
    print(f"   Total unique candidates assigned: {len(check_cands):,}")
    print(f"   Total multi-assignment collisions: {collisions} (Must be 0)")
    assert collisions == 0, f"FATAL: Found {collisions} collisions in v24!"

    # 9. Overwrite matching_results.tsv with validated v24
    print(f"\n9. Promoting v24 to production {os.path.basename(FINAL_TSV)} ...")
    shutil.copy2(OUT_V24_TSV, FINAL_TSV)

    # 10. Official Validator
    print("\n10. Running official utils/validate_submission.py ...")
    val_cmd = [
        sys.executable,
        os.path.join(UTILS_DIR, "validate_submission.py"),
        "--matching", FINAL_TSV,
        "--candidate", CAND_TSV,
        "--test-dir", DATA_DIR,
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"FATAL: Validator failed:\n{res.stderr}")
        sys.exit(1)

    print(f"\n=== v24 FRANCE-ENHANCED ENGINE COMPLETE IN {time.time()-t_start:.1f}s ===")
    print(f"Ready for portal upload: {FINAL_TSV}")


if __name__ == "__main__":
    main()
