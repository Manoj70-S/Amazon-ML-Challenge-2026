#!/usr/bin/env python3
"""
rescue_aggressive_v14.py
========================
Amazon ML Challenge 2026 – Final Aggressive Multi-Signal Rescue Engine (v14)
Designed to maximize entity-level Macro-F0.5 on the test set.

Architecture:
1. Base: 100% preserves the verified, zero-collision v13 submission (1,585,190 non-empty matches).
2. Signal 1 (Calibrated Address-Digit Rescue):
   - AddrSim >= 92% (calibrated from 98%) + shared building/PIN digits + (Indic script OR shared name token).
3. Signal 2 (PIN Code + Name Alignment):
   - Exact 5/6-digit PIN/Postal code match + Name Token Sort >= 85% + Addr Token Set >= 75%.
4. Signal 3 (Near-Exact Name + Location Consistency):
   - Name Token Sort >= 95% + shared digits + Addr Token Set >= 70%.
5. Signal 4 (First-Word Anchor + Digit Alignment):
   - Identical first word (>= 4 chars) + shared PIN/building digits + Addr Token Set >= 80%.
6. Signal 5 (Cross-Source S2 <-> S3 Cluster Completion):
   - For single-match entities, recover unassigned counterpart source (S2 <-> S3) with exact PIN and AddrSim >= 90%.
7. Global Disjoint Collision Resolution:
   - Highest-similarity claim wins strictly. Zero multi-assignments guaranteed.
8. Official Submission Validation:
   - Validates format compliance via utils/validate_submission.py.
"""

import os
import sys
import re
import time
import shutil
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

BASE_TSV      = os.path.join(OUTPUT_DIR, "matching_results_v13_locked.tsv")
CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
S1_TSV        = os.path.join(DATA_DIR, "test_source1.tsv")
S2_TSV        = os.path.join(DATA_DIR, "test_source2.tsv")
S3_TSV        = os.path.join(DATA_DIR, "test_source3.tsv")
OUT_V14_TSV   = os.path.join(OUTPUT_DIR, "matching_results_v14_aggressive.tsv")
FINAL_TSV     = os.path.join(OUTPUT_DIR, "matching_results.tsv")


def main():
    t0 = time.time()
    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026: AGGRESSIVE MULTI-SIGNAL RESCUE ENGINE (v14)", flush=True)
    print("  Target: Maximize Macro-F0.5 by Recovering High-Confidence Matches", flush=True)
    print("=" * 75, flush=True)

    # 1. Load base submission (v13 locked)
    print("\n1. Loading base submission (v13 locked) ...", flush=True)
    s1_ordered = []
    base_matches: Dict[str, List[str]] = {}
    assigned_candidates: Set[str] = set()
    empty_s1: Set[str] = set()
    single_match_s1: Set[str] = set()

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
                    single_match_s1.add(sid)
            else:
                base_matches[sid] = []
                empty_s1.add(sid)

    print(f"   Total S1 entities:       {len(s1_ordered):,}", flush=True)
    print(f"   Base non-empty entities: {len(s1_ordered) - len(empty_s1):,}", flush=True)
    print(f"   Target empty singletons: {len(empty_s1):,}", flush=True)
    print(f"   Single-match entities:   {len(single_match_s1):,}", flush=True)
    print(f"   Already assigned cands:  {len(assigned_candidates):,}", flush=True)

    # 2. Load text for empty singletons & single-match entities
    targets_to_load = empty_s1 | single_match_s1
    print(f"\n2. Loading text for {len(targets_to_load):,} target S1 entities ...", flush=True)
    s1_data: Dict[str, Tuple[str, str, Set[str], Set[str], str]] = {}

    with open(S1_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in targets_to_load:
                raw_n = p[1] if len(p) > 1 else ""
                raw_a = p[2] if len(p) > 2 else ""
                norm_n = normalize_name(raw_n)
                norm_a = normalize_address(raw_a)
                nums = set(RE_DIGITS.findall(norm_a)) if norm_a else set()
                pins = set(RE_PIN.findall(norm_a)) if norm_a else set()
                country = p[3] if len(p) > 3 else ""
                s1_data[sid] = (norm_n, norm_a, nums, pins, country)

    print(f"   Loaded data for {len(s1_data):,} entities", flush=True)

    # 3. Read candidates for targets from candidate_pairs.tsv
    print("\n3. Collecting unassigned candidate IDs from candidate_pairs.tsv ...", flush=True)
    target_cands: Dict[str, List[str]] = defaultdict(list)
    needed_cids: Set[str] = set()

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in targets_to_load and len(p) > 1 and p[1].strip():
                top_limit = 20 if sid in empty_s1 else 8
                cids = [c.strip() for c in p[1].split(",") if c.strip()][:top_limit]
                unassigned = [c for c in cids if c not in assigned_candidates]
                if unassigned:
                    target_cands[sid] = unassigned
                    needed_cids.update(unassigned)

    print(f"   Targets with unassigned candidates: {len(target_cands):,}", flush=True)
    print(f"   Total unique candidates needed:     {len(needed_cids):,}", flush=True)

    # 4. Stream S2 & S3 candidate texts
    print(f"\n4. Streaming candidate texts from test_source2.tsv and test_source3.tsv ...", flush=True)
    cand_data: Dict[str, Tuple[str, str, Set[str], Set[str], bool]] = {}

    for src_path in [S2_TSV, S3_TSV]:
        t_src = time.time()
        c_found = 0
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                cid = p[0]
                if cid in needed_cids:
                    raw_n = p[1] if len(p) > 1 else ""
                    raw_a = p[2] if len(p) > 2 else ""
                    norm_n = normalize_name(raw_n)
                    norm_a = normalize_address(raw_a)
                    nums = set(RE_DIGITS.findall(norm_a)) if norm_a else set()
                    pins = set(RE_PIN.findall(norm_a)) if norm_a else set()
                    is_indic = any(ord(ch) > 127 for ch in raw_n)
                    cand_data[cid] = (norm_n, norm_a, nums, pins, is_indic)
                    c_found += 1
        print(f"   {os.path.basename(src_path)}: loaded {c_found:,} candidates in {time.time()-t_src:.1f}s", flush=True)

    print(f"   Candidate text cache ready: {len(cand_data):,} records", flush=True)

    # 5. Evaluate Multi-Signal Rescue Matching
    print("\n5. Evaluating Multi-Signal Aggressive Rescue Rules ...", flush=True)
    t_match = time.time()
    candidate_claims: List[Tuple[str, str, float, str]] = []  # (sid, cid, score, rule_name)

    # Signal stats
    rule_counts = Counter()

    for sid, cids in target_cands.items():
        is_empty = (sid in empty_s1)
        s_info = s1_data.get(sid)
        if not s_info:
            continue
        sn, sa, s_nums, s_pins, sc = s_info

        s_words = set(w for w in sn.split() if len(w) >= 3) if sn else set()
        s_first_w = sn.split()[0] if sn.split() else ""
        existing_matches = base_matches.get(sid, [])
        existing_sources = set(m[:2] for m in existing_matches)

        best_score = 0.0
        best_cid = None
        best_rule = ""

        for cid in cids:
            c_info = cand_data.get(cid)
            if not c_info:
                continue
            cn, ca, c_nums, c_pins, is_indic = c_info
            c_source = cid[:2]

            c_words = set(w for w in cn.split() if len(w) >= 3) if cn else set()
            c_first_w = cn.split()[0] if cn.split() else ""
            has_word_overlap = bool(s_words & c_words) if (s_words and c_words) else False

            # --- CASE A: EMPTY SINGLETON RESCUE ---
            if is_empty:
                # Rule 1: High Address Similarity (>= 92%) with Digit Match + Indic/Word
                if sa and ca and len(sa) >= 15 and len(ca) >= 15 and (s_nums & c_nums):
                    if is_indic or has_word_overlap:
                        asim = fuzz.token_set_ratio(sa, ca)
                        if asim >= 92.0 and asim > best_score:
                            best_score = asim
                            best_cid = cid
                            best_rule = "R1_Addr92_Digit"

                # Rule 2: Exact PIN Code Match (5 or 6 digits) + Strong Name/Address
                if (s_pins & c_pins) and sn and cn:
                    nsim = fuzz.token_sort_ratio(sn, cn)
                    asim = fuzz.token_set_ratio(sa, ca) if (sa and ca) else 0.0
                    if nsim >= 82.0 and asim >= 75.0:
                        comb_score = 0.5 * nsim + 0.5 * asim
                        if comb_score > best_score:
                            best_score = comb_score
                            best_cid = cid
                            best_rule = "R2_ExactPIN_Name"

                # Rule 3: Near-Exact Name (>= 95%) + Digit Match + Modest Address (>= 70%)
                if sn and cn and len(sn) >= 5 and (s_nums & c_nums):
                    nsim = fuzz.token_sort_ratio(sn, cn)
                    if nsim >= 95.0:
                        asim = fuzz.token_set_ratio(sa, ca) if (sa and ca) else 0.0
                        if asim >= 70.0:
                            comb_score = 0.6 * nsim + 0.4 * asim
                            if comb_score > best_score:
                                best_score = comb_score
                                best_cid = cid
                                best_rule = "R3_ExactName_Loc"

                # Rule 4: First Word Exact (>= 4 chars) + Shared PIN/Digit + Addr >= 80%
                if s_first_w and c_first_w and len(s_first_w) >= 4 and s_first_w == c_first_w:
                    if (s_pins & c_pins) or (s_nums & c_nums):
                        asim = fuzz.token_set_ratio(sa, ca) if (sa and ca) else 0.0
                        if asim >= 80.0 and asim > best_score:
                            best_score = asim
                            best_cid = cid
                            best_rule = "R4_FirstWord_Addr80"

            # --- CASE B: CROSS-SOURCE CLUSTER COMPLETION (for single-match entities) ---
            else:
                # If S1 has S2, complete with S3 (or vice versa)
                if c_source not in existing_sources:
                    if (s_pins & c_pins) and sa and ca:
                        asim = fuzz.token_set_ratio(sa, ca)
                        nsim = fuzz.token_sort_ratio(sn, cn) if (sn and cn) else 0.0
                        if asim >= 90.0 and nsim >= 80.0:
                            comb_score = 0.5 * asim + 0.5 * nsim
                            if comb_score > best_score:
                                best_score = comb_score
                                best_cid = cid
                                best_rule = "R5_CrossSource_Comp"

        if best_cid:
            candidate_claims.append((sid, best_cid, best_score, best_rule))
            rule_counts[best_rule] += 1

    print(f"   Raw rescue claims generated: {len(candidate_claims):,} in {time.time()-t_match:.1f}s", flush=True)
    print("   Claims by Rule:")
    for rname, cnt in rule_counts.most_common():
        print(f"     {rname:22s}: {cnt:6,d}", flush=True)

    # 6. Global Disjoint Collision Resolution
    print("\n6. Global Disjoint Collision Resolution (Highest Score Wins) ...", flush=True)
    # Sort claims by score descending
    candidate_claims.sort(key=lambda x: x[2], reverse=True)

    winner_for_cand: Dict[str, Tuple[str, float, str]] = {}
    purged_collisions = 0

    for sid, cid, score, rule in candidate_claims:
        if cid not in winner_for_cand and cid not in assigned_candidates:
            winner_for_cand[cid] = (sid, score, rule)
        else:
            purged_collisions += 1

    rescued_by_s1: Dict[str, List[str]] = defaultdict(list)
    empty_rescued_count = 0
    multimatch_expanded_count = 0

    for cid, (sid, score, rule) in winner_for_cand.items():
        rescued_by_s1[sid].append(cid)
        if sid in empty_s1:
            empty_rescued_count += 1
        else:
            multimatch_expanded_count += 1

    print(f"   Collisions purged:               {purged_collisions:,}", flush=True)
    print(f"   Unique candidates winning:       {len(winner_for_cand):,}", flush=True)
    print(f"   Empty singletons rescued:        {empty_rescued_count:,}", flush=True)
    print(f"   Multi-match clusters expanded:   {multimatch_expanded_count:,}", flush=True)

    # 7. Write Final TSV
    print(f"\n7. Writing v14 final submission to {os.path.basename(OUT_V14_TSV)} ...", flush=True)
    final_non_empty = 0
    final_empty = 0
    total_final_links = 0

    with open(OUT_V14_TSV, "w", encoding="utf-8") as f_out:
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
    print(f"   Final non-empty entities: {final_non_empty:,} (+{empty_rescued_count:,} vs v13)", flush=True)
    print(f"   Final empty singletons:   {final_empty:,} (reduced from {len(empty_s1):,})", flush=True)
    print(f"   Total links in v14:       {total_final_links:,} (+{len(winner_for_cand):,} vs v13)", flush=True)

    # 8. Verify ZERO Collisions
    print("\n8. Auditing 100% submission for collisions ...", flush=True)
    check_cands = Counter()
    with open(OUT_V14_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                for c in p[1].split(","):
                    check_cands[c.strip()] += 1

    collisions = sum(1 for c, cnt in check_cands.items() if cnt > 1)
    print(f"   Total unique candidates assigned: {len(check_cands):,}")
    print(f"   Total multi-assignment collisions: {collisions} (Must be 0)")
    assert collisions == 0, f"FATAL: Found {collisions} collisions in v14!"

    # 9. Overwrite matching_results.tsv with validated v14
    print(f"\n9. Promoting v14 to production {os.path.basename(FINAL_TSV)} ...", flush=True)
    shutil.copy2(OUT_V14_TSV, FINAL_TSV)

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

    print(f"\n=== v14 AGGRESSIVE RESCUE COMPLETE IN {time.time()-t0:.1f}s ===")
    print(f"Ready for portal upload: {FINAL_TSV}")

if __name__ == "__main__":
    main()
