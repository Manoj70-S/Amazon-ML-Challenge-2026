#!/usr/bin/env python3
"""
purge_collisions_v7.py
======================
Purges all multi-assigned candidate collisions from matching_results.tsv.

Why this is critical:
  - In Amazon's ground truth, every S2/S3 entity belongs to at most ONE S1 entity (0 multi-matches).
  - The v6 output has 54,903 candidates assigned to multiple S1 entities, creating 79,032 guaranteed
    false-positive links.
  - Since the metric is Macro F0.5 (which weights precision 2.5x more than recall), these 79,032
    false positives heavily penalize the score.

Method:
  - For each multi-assigned candidate, evaluate its similarity against all competing S1 entities.
  - Assign the candidate EXCLUSIVELY to its single best-matching S1 entity (highest similarity).
  - Strips the candidate from all inferior S1 rows.
  - Output is strictly 1-to-many disjoint matching with 0 collisions.
  - Validates output with official validate_submission.py.
"""

import os
import sys
import time
from collections import defaultdict, Counter
from rapidfuzz import fuzz

# Ensure UTF-8 output
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")

MATCHING_TSV    = os.path.join(OUTPUT_DIR, "matching_results.tsv")
MATCHING_V7_TSV = os.path.join(OUTPUT_DIR, "matching_results_v7.tsv")
CANDIDATE_TSV   = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

def main():
    t0 = time.time()
    print("=" * 70)
    print("  Amazon ML Challenge 2026: False-Positive Collision Purge (v7)")
    print("=" * 70)

    # 1. Stream existing matching_results.tsv
    print("\n1. Streaming current matching_results.tsv ...")
    cand_to_s1 = defaultdict(list)
    s1_rows = []
    total_links_orig = 0

    with open(MATCHING_TSV, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sid = parts[0]
            mstr = parts[1] if len(parts) > 1 else ""
            cands = mstr.split(",") if mstr else []
            s1_rows.append((sid, cands))
            for c in cands:
                cand_to_s1[c].append(sid)
                total_links_orig += 1

    multi_cands = {c: s1s for c, s1s in cand_to_s1.items() if len(s1s) > 1}
    excess_links = sum(len(s1s) - 1 for s1s in multi_cands.values())

    print(f"   Total S1 queries:              {len(s1_rows):,}")
    print(f"   Total links originally:        {total_links_orig:,}")
    print(f"   Total unique candidates:       {len(cand_to_s1):,}")
    print(f"   Multi-assigned candidates:     {len(multi_cands):,}")
    print(f"   Guaranteed false-positive links to purge: {excess_links:,}")

    if not multi_cands:
        print("No multi-assigned candidates found! File already clean.")
        return

    # Set of S1 IDs that are competing for multi-assigned candidates
    needed_s1_ids = set()
    for s1s in multi_cands.values():
        for s in s1s:
            needed_s1_ids.add(s)
    needed_cand_ids = set(multi_cands.keys())

    print(f"\n2. Loading text data for {len(needed_cand_ids):,} conflicting candidates and {len(needed_s1_ids):,} competing S1 entities ...")

    # Load only the needed S1 records
    s1_text = {}
    with open(os.path.join(DATA_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if p[0] in needed_s1_ids:
                name = p[1] if len(p) > 1 else ""
                addr = p[2] if len(p) > 2 else ""
                s1_text[p[0]] = (name.lower(), addr.lower())
            if len(s1_text) == len(needed_s1_ids):
                break

    # Load only the needed candidate records (from test_source2 and test_source3)
    cand_text = {}
    for src_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, src_file)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if p[0] in needed_cand_ids:
                    name = p[1] if len(p) > 1 else ""
                    addr = p[2] if len(p) > 2 else ""
                    cand_text[p[0]] = (name.lower(), addr.lower())

    print(f"   Loaded text for {len(cand_text):,} candidates and {len(s1_text):,} S1 entities in {time.time()-t0:.1f}s")

    # 3. Resolve conflicts: assign each multi-assigned candidate to its BEST S1 match
    print("\n3. Resolving candidate collisions using similarity argmax ...")
    t_res = time.time()
    best_s1_for_cand = {}

    for cid, competing_s1s in multi_cands.items():
        c_name, c_addr = cand_text.get(cid, ("", ""))
        best_s1 = None
        best_score = -1.0

        for sid in competing_s1s:
            s_name, s_addr = s1_text.get(sid, ("", ""))
            
            # Compute token sort similarity
            name_score = fuzz.token_sort_ratio(c_name, s_name) if c_name and s_name else 0.0
            addr_score = fuzz.token_sort_ratio(c_addr, s_addr) if c_addr and s_addr else 0.0
            
            if c_addr and s_addr:
                score = (0.6 * name_score) + (0.4 * addr_score)
            else:
                score = name_score

            if score > best_score:
                best_score = score
                best_s1 = sid

        best_s1_for_cand[cid] = best_s1

    print(f"   Resolved {len(best_s1_for_cand):,} conflicts in {time.time()-t_res:.2f}s")

    # 4. Reconstruct clean, strictly disjoint matching_results
    print(f"\n4. Writing clean matching results to {MATCHING_V7_TSV} ...")
    purged_links = 0
    clean_links = 0
    empty_s1 = 0
    non_empty_s1 = 0

    with open(MATCHING_V7_TSV, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, cands in s1_rows:
            kept_cands = []
            for c in cands:
                if c in best_s1_for_cand:
                    # It was multi-assigned: keep only if sid is the chosen best_s1
                    if best_s1_for_cand[c] == sid:
                        kept_cands.append(c)
                        clean_links += 1
                    else:
                        purged_links += 1
                else:
                    # It was already uniquely assigned
                    kept_cands.append(c)
                    clean_links += 1

            if kept_cands:
                non_empty_s1 += 1
                out_f.write(f"{sid}\t{','.join(kept_cands)}\n")
            else:
                empty_s1 += 1
                out_f.write(f"{sid}\t\n")

    print(f"   Cleaned results written:")
    print(f"     Total S1 entities:            {len(s1_rows):,}")
    print(f"     Non-empty entities:           {non_empty_s1:,} ({non_empty_s1/len(s1_rows)*100:.2f}%)")
    print(f"     Empty entities (singletons):  {empty_s1:,} ({empty_s1/len(s1_rows)*100:.2f}%)")
    print(f"     Purged false-positive links:  {purged_links:,}")
    print(f"     Remaining valid match links:  {clean_links:,}")

    # 5. Verify 0 multi-assignments in new file
    new_counts = Counter()
    with open(MATCHING_V7_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) > 1 and p[1]:
                for c in p[1].split(","):
                    new_counts[c] += 1

    remaining_dups = sum(1 for v in new_counts.values() if v > 1)
    print(f"   Verification: Candidates assigned to >1 S1 in v7: {remaining_dups} (must be 0)")
    assert remaining_dups == 0, "Error: duplicates still exist!"

    # 6. Official Submission Validation
    print("\n5. Running official submission validator ...")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results_v7.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    ret = os.system(val_cmd)

    if ret == 0:
        print("\nValidator PASSED! Overwriting output/matching_results.tsv with v7 clean results ...")
        import shutil
        shutil.copyfile(MATCHING_V7_TSV, MATCHING_TSV)
        print("output/matching_results.tsv is now updated and ready for upload!")
    else:
        print(f"\nValidator returned non-zero exit code: {ret}")

    print(f"\nTotal Elapsed Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
