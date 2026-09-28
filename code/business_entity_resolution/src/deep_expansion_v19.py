#!/usr/bin/env python3
"""
deep_expansion_v19.py
=====================
Amazon ML Challenge 2026 – Street-Address & Compound-Key Expansion Engine (v19)

Philosophy:
1. Immutable Base: 100% preserves the verified v16 submission (Score: 0.86115).
2. Targets Single-Link Entities (305,748 entities):
   - In ground truth, only 5.4% have 1 link; 94.6% have multi-links.
   - Recovers the missing counterpart source (S2 <-> S3) using exact street number matching
     and high name token-set fidelity (>= 90%).
3. Robust to Cross-Lingual & Unit Formatting Variations:
   - Handles "Unit 11" vs "PMB 1523", "Rd" vs "Road", and transliterated Indic/French names.
4. Conservative Single-Winner Policy:
   - Exactly ONE counterpart candidate per entity (best_cid only), avoiding multi-link flooding.
5. Strictly Disjoint Winner Resolution:
   - Globally resolves claims so every candidate is assigned to at most one S1 entity (0 collisions).
6. Official Validation:
   - Verified via utils/validate_submission.py.
"""

import os
import sys
import re
import time
import shutil
import unicodedata
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz

RE_DIGITS = re.compile(r'\b\d+\b')
RE_PIN = re.compile(r'\b\d{5,6}\b')

PROJECT_ROOT = os.path.abspath(".")
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")
SRC_DIR      = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src")

sys.path.append(SRC_DIR)
from normalize import normalize_name, normalize_address

BASE_TSV      = os.path.join(OUTPUT_DIR, "matching_results_v16_deep.tsv")
CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
S1_TSV        = os.path.join(DATA_DIR, "test_source1.tsv")
S2_TSV        = os.path.join(DATA_DIR, "test_source2.tsv")
S3_TSV        = os.path.join(DATA_DIR, "test_source3.tsv")
OUT_V19_TSV   = os.path.join(OUTPUT_DIR, "matching_results_v19_compound.tsv")
FINAL_TSV     = os.path.join(OUTPUT_DIR, "matching_results.tsv")


def get_street_num(addr: str) -> str:
    m = RE_DIGITS.findall(addr)
    nums = [n for n in m if len(n) < 5]
    return nums[0] if nums else ''


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  AMAZON ML CHALLENGE 2026: STREET-COMPOUND EXPANSION ENGINE (v19)", flush=True)
    print("  Target: Single-Link Counterpart Recovery (Base: v16 Score 0.86115)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load base submission (v16: 0.86115)
    print("\n1. Loading base submission (v16: 0.86115) ...", flush=True)
    s1_ordered = []
    base_matches: Dict[str, List[str]] = {}
    assigned_candidates: Set[str] = set()
    single_link_s1: Set[str] = set()

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
                    single_link_s1.add(sid)
            else:
                base_matches[sid] = []

    print(f"   Total S1 entities:            {len(s1_ordered):,}", flush=True)
    print(f"   Base non-empty entities:      {len(s1_ordered) - sum(1 for v in base_matches.values() if not v):,}", flush=True)
    print(f"   Single-link entities:         {len(single_link_s1):,}", flush=True)
    print(f"   Already assigned candidates:  {len(assigned_candidates):,}", flush=True)

    # 2. Load text for single-link target S1 entities
    print(f"\n2. Loading text for {len(single_link_s1):,} target entities ...", flush=True)
    s1_data = {}  # sid -> (norm_n, norm_a, street_num)

    with open(S1_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in single_link_s1:
                raw_n = p[1] if len(p) > 1 else ""
                raw_a = p[2] if len(p) > 2 else ""
                norm_n = normalize_name(raw_n)
                norm_a = normalize_address(raw_a)
                street_num = get_street_num(norm_a)
                s1_data[sid] = (norm_n, norm_a, street_num)

    print(f"   Loaded data for {len(s1_data):,} entities", flush=True)

    # 3. Read unassigned candidate IDs from candidate_pairs.tsv (up to rank 25)
    print("\n3. Collecting candidate IDs from candidate_pairs.tsv (up to rank 25) ...", flush=True)
    target_cands = defaultdict(list)
    needed_cids = set()

    # Anchor candidate IDs from single_link entities
    anchor_cids = set()
    for sid in single_link_s1:
        mids = base_matches.get(sid, [])
        if mids:
            anchor_cids.add(mids[0])

    needed_cids.update(anchor_cids)

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in single_link_s1 and len(p) > 1 and p[1].strip():
                cids = [c.strip() for c in p[1].split(",") if c.strip()][:25]
                unassigned = [c for c in cids if c not in assigned_candidates]
                if unassigned:
                    target_cands[sid] = unassigned
                    needed_cids.update(unassigned)

    print(f"   Target entities with unassigned candidates: {len(target_cands):,}", flush=True)
    print(f"   Anchor candidates needed:                   {len(anchor_cids):,}", flush=True)
    print(f"   Total unique candidates needed from S2/S3:  {len(needed_cids):,}", flush=True)

    # 4. Stream S2 & S3 candidate texts using fast scanner
    print(f"\n4. Streaming candidate texts from test_source2.tsv and test_source3.tsv ...", flush=True)
    cand_data = {}  # cid -> (norm_n, norm_a, street_num)

    for src_path in [S2_TSV, S3_TSV]:
        t_src = time.time()
        c_found = 0
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)
            for line in f:
                tab1 = line.find("\t")
                if tab1 == -1:
                    continue
                cid = line[:tab1]
                if cid in needed_cids:
                    p = line.rstrip("\r\n").split("\t")
                    raw_n = p[1] if len(p) > 1 else ""
                    raw_a = p[2] if len(p) > 2 else ""
                    norm_n = normalize_name(raw_n)
                    norm_a = normalize_address(raw_a)
                    street_num = get_street_num(norm_a)
                    cand_data[cid] = (norm_n, norm_a, street_num)
                    c_found += 1
        print(f"   {os.path.basename(src_path)}: loaded {c_found:,} candidates in {time.time()-t_src:.1f}s", flush=True)

    print(f"   Candidate text cache ready: {len(cand_data):,} records", flush=True)

    # 5. Evaluate Compound-Key Counterpart Expansion
    print("\n5. Evaluating Compound-Key Counterpart Recovery ...", flush=True)
    t_match = time.time()
    candidate_claims: List[Tuple[str, str, float, str]] = []  # (sid, cid, score, rule_name)
    rule_counts = Counter()

    for sid, cids in target_cands.items():
        s_info = s1_data.get(sid)
        if not s_info:
            continue
        sn, sa, s_num = s_info

        mids = base_matches.get(sid, [])
        if not mids:
            continue
        anchor_id = mids[0]
        anchor_info = cand_data.get(anchor_id)
        if not anchor_info:
            continue
        an, aa, a_num = anchor_info
        anchor_source = anchor_id[:2]
        target_source = "S3" if anchor_source == "S2" else "S2"

        best_score = 0.0
        best_cid = None
        best_rule = ""

        for cid in cids:
            if not cid.startswith(target_source):
                continue
            c_info = cand_data.get(cid)
            if not c_info:
                continue
            cn, ca, c_num = c_info

            # Check 1: Exact Street Number Match + High Name Fidelity
            if s_num and c_num and s_num == c_num:
                n_set = fuzz.token_set_ratio(sn, cn)
                a_set = fuzz.token_set_ratio(sa, ca)
                if n_set >= 90.0 and a_set >= 70.0:
                    score = 0.5 * n_set + 0.5 * a_set + 5.0  # Priority premium
                    if score > best_score:
                        best_score = score
                        best_cid = cid
                        best_rule = "C1_StreetNum_NameSet90"
                        continue

            # Check 2: High Address & Name Alignment with Anchor Corroboration
            s_asim = fuzz.token_set_ratio(sa, ca)
            if s_asim >= 80.0:
                s_nsim = fuzz.token_sort_ratio(sn, cn)
                if s_nsim >= 80.0:
                    a_asim = fuzz.token_set_ratio(aa, ca)
                    if a_asim >= 78.0:
                        a_nsim = fuzz.token_sort_ratio(an, cn)
                        if a_nsim >= 78.0:
                            score = (s_asim + s_nsim + a_asim + a_nsim) / 4.0
                            if score > best_score:
                                best_score = score
                                best_cid = cid
                                best_rule = "C2_Triangular_Calibrated"

        if best_cid:
            candidate_claims.append((sid, best_cid, best_score, best_rule))
            rule_counts[best_rule] += 1

    print(f"   Raw expansion claims generated: {len(candidate_claims):,} in {time.time()-t_match:.1f}s", flush=True)
    print("   Claims by Rule:")
    for rname, cnt in rule_counts.most_common():
        print(f"     {rname:28s}: {cnt:6,d}", flush=True)

    # 6. Global Disjoint Collision Resolution (Highest Score Wins)
    print("\n6. Global Disjoint Collision Resolution (Highest Score Wins) ...", flush=True)
    candidate_claims.sort(key=lambda x: x[2], reverse=True)

    winner_for_cand: Dict[str, Tuple[str, float, str]] = {}
    purged_collisions = 0

    for sid, cid, score, rule in candidate_claims:
        if cid not in winner_for_cand and cid not in assigned_candidates:
            winner_for_cand[cid] = (sid, score, rule)
        else:
            purged_collisions += 1

    rescued_by_s1: Dict[str, List[str]] = defaultdict(list)
    for cid, (sid, score, rule) in winner_for_cand.items():
        rescued_by_s1[sid].append(cid)

    print(f"   Collisions purged:                  {purged_collisions:,}", flush=True)
    print(f"   Unique candidates winning:          {len(winner_for_cand):,}", flush=True)
    print(f"   Single-link entities expanded:      {len(rescued_by_s1):,}", flush=True)

    # 7. Write Final TSV
    print(f"\n7. Writing v19 final submission to {os.path.basename(OUT_V19_TSV)} ...", flush=True)
    final_non_empty = 0
    final_empty = 0
    total_final_links = 0

    with open(OUT_V19_TSV, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ordered:
            base_m = base_matches.get(sid, [])
            new_m  = rescued_by_s1.get(sid, [])
            all_m  = base_m + new_m
            if all_m:
                f_out.write(f"{sid}\t{','.join(all_m)}\n")
                final_non_empty += 1
                total_final_links += len(all_m)
            else:
                f_out.write(f"{sid}\t\n")
                final_empty += 1

    print(f"   Total rows written:       {len(s1_ordered):,}", flush=True)
    print(f"   Final non-empty entities: {final_non_empty:,}", flush=True)
    print(f"   Final empty singletons:   {final_empty:,}", flush=True)
    print(f"   Total links in v19:       {total_final_links:,} (+{len(winner_for_cand):,} vs v16)", flush=True)

    # 8. Verify ZERO Collisions
    print("\n8. Auditing 100% submission for multi-assignment collisions ...", flush=True)
    check_cands = Counter()
    with open(OUT_V19_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                for c in p[1].split(","):
                    check_cands[c.strip()] += 1

    collisions = sum(1 for c, cnt in check_cands.items() if cnt > 1)
    print(f"   Total unique candidates assigned: {len(check_cands):,}")
    print(f"   Total multi-assignment collisions: {collisions} (Must be 0)")
    assert collisions == 0, f"FATAL: Found {collisions} collisions in v19!"

    # 9. Overwrite matching_results.tsv with validated v19
    print(f"\n9. Promoting v19 to production {os.path.basename(FINAL_TSV)} ...", flush=True)
    shutil.copy2(OUT_V19_TSV, FINAL_TSV)

    # 10. Official Validator
    print("\n10. Running official utils/validate_submission.py ...", flush=True)
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

    print(f"\n=== v19 STREET-COMPOUND ENGINE COMPLETE IN {time.time()-t0:.1f}s ===")
    print(f"Ready for portal upload: {FINAL_TSV}")


if __name__ == "__main__":
    main()
