#!/usr/bin/env python3
"""
patch_domain_s3.py
==================
Surgical patch for Domain-Style S3 records using Trigram Dice similarity at tau >= 0.80.

Operates strictly as requested:
  1. Loads existing clean v7 matching_results.tsv (with 0 collisions).
  2. Scans test_source3.tsv to index domain-style S3 records (matching domain regex or fused string).
  3. Scans candidate_pairs.tsv for S1 entities that have domain-style S3 candidates.
  4. Evaluates Trigram Dice similarity (tau >= 0.80) on stripped domain vs S1 compact name.
  5. Applies address consistency check (requires non-contradicting address).
  6. Enforces strict disjoint 1-to-many constraint: NO candidate can be assigned to multiple S1s.
  7. Formats and writes updated output, runs official validator to confirm PASS.
"""

import os
import sys
import time
import re
from collections import defaultdict, Counter

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")

MATCHING_TSV     = os.path.join(OUTPUT_DIR, "matching_results.tsv")
PATCHED_TEMP_TSV = os.path.join(OUTPUT_DIR, "matching_results_v8_tmp.tsv")
CANDIDATE_TSV    = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

DOMAIN_REGEX = re.compile(r"\b([a-zA-Z0-9_\-]+)\.(com|net|org|in|co|io|biz|info|us|gov|edu)\b", re.IGNORECASE)

def strip_domain(name: str) -> str:
    s = re.sub(r"\.(com|net|org|in|co|io|biz|info|us|gov|edu)", "", name, flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    return s

def normalize_s1_compact(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()

def get_ngrams(s: str, n: int = 3):
    if len(s) < n:
        return {s} if s else set()
    return set(s[i:i+n] for i in range(len(s) - n + 1))

def dice_from_sets(set1, set2):
    inter = len(set1 & set2)
    total = len(set1) + len(set2)
    return (2.0 * inter) / total if total > 0 else 0.0

def main():
    t0 = time.time()
    print("=" * 70)
    print("  Amazon ML Challenge 2026: Surgical Domain S3 Patch (Trigram Dice >= 0.80)")
    print("=" * 70)

    # 1. Load existing predictions from matching_results.tsv
    print("\n1. Loading existing predictions from output/matching_results.tsv ...")
    s1_order = []
    s1_matches = {}
    assigned_candidates = {} # cid -> sid

    with open(MATCHING_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            sid = p[0]
            s1_order.append(sid)
            m_str = p[1] if len(p) > 1 else ""
            cands = m_str.split(",") if m_str else []
            s1_matches[sid] = list(cands)
            for c in cands:
                assigned_candidates[c] = sid

    empty_count_init = sum(1 for sid in s1_order if not s1_matches[sid])
    total_links_init = sum(len(m) for m in s1_matches.values())
    print(f"   Loaded {len(s1_order):,} S1 entities | {total_links_init:,} existing match links")
    print(f"   Initial empty singletons: {empty_count_init:,} ({empty_count_init/len(s1_order)*100:.2f}%)")

    # 2. Scan test_source3.tsv to index domain-style S3 records
    print("\n2. Scanning test_source3.tsv for domain-style records ...")
    t_s3 = time.time()
    domain_s3_dict = {} # cid -> (stripped_name, ngrams, addr_clean, country)

    with open(os.path.join(DATA_DIR, "test_source3.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            cid = p[0]
            name = p[1] if len(p) > 1 else ""
            is_domain = bool(DOMAIN_REGEX.search(name)) or (" " not in name.strip() and len(name.strip()) >= 8)
            if is_domain:
                stripped = strip_domain(name)
                if len(stripped) >= 4:
                    addr = p[2].lower() if len(p) > 2 else ""
                    ctry = p[3] if len(p) > 3 else ""
                    domain_s3_dict[cid] = (stripped, get_ngrams(stripped, 3), addr, ctry)

    print(f"   Indexed {len(domain_s3_dict):,} domain/fused S3 records in {time.time()-t_s3:.1f}s")

    # 3. Scan candidate_pairs.tsv for relevant (S1, S3_domain) candidate pairs
    print("\n3. Scanning candidate_pairs.tsv for domain S3 candidates ...")
    t_cand = time.time()
    s1_domain_candidates = defaultdict(list)
    needed_s1_ids = set()

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            sid = p[0]
            cands_str = p[1] if len(p) > 1 else ""
            if not cands_str:
                continue
            cands = cands_str.split(",")
            for c in cands:
                if c.startswith("S3-") and c in domain_s3_dict:
                    # Only consider if not already matched to this S1
                    if c not in s1_matches[sid]:
                        s1_domain_candidates[sid].append(c)
                        needed_s1_ids.add(sid)

    print(f"   Found {len(s1_domain_candidates):,} S1 entities with {sum(len(v) for v in s1_domain_candidates.values()):,} domain S3 candidate pairs in {time.time()-t_cand:.1f}s")

    # 4. Load S1 text for the candidate pairs
    print(f"\n4. Loading text for {len(needed_s1_ids):,} candidate S1 entities ...")
    t_s1 = time.time()
    s1_text = {} # sid -> (compact_name, ngrams, addr_clean, country)

    with open(os.path.join(DATA_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            sid = p[0]
            if sid in needed_s1_ids:
                raw_name = p[1] if len(p) > 1 else ""
                compact = normalize_s1_compact(raw_name)
                addr = p[2].lower() if len(p) > 2 else ""
                ctry = p[3] if len(p) > 3 else ""
                s1_text[sid] = (compact, get_ngrams(compact, 3), addr, ctry)
            if len(s1_text) == len(needed_s1_ids):
                break

    print(f"   Loaded in {time.time()-t_s1:.1f}s")

    # 5. Evaluate Trigram Dice at tau >= 0.80 and filter
    print("\n5. Re-scoring candidate pairs with Trigram Dice (tau >= 0.80) ...")
    t_eval = time.time()
    TAU_DICE = 0.80

    # Proposed new links: cid -> list of (sid, dice_score)
    proposed_links = defaultdict(list)
    pairs_evaluated = 0

    for sid, cand_list in s1_domain_candidates.items():
        s_compact, s_ngrams, s_addr, s_ctry = s1_text.get(sid, ("", set(), "", ""))
        if not s_compact:
            continue

        for cid in cand_list:
            pairs_evaluated += 1
            c_stripped, c_ngrams, c_addr, c_ctry = domain_s3_dict[cid]

            # Fast character length check
            len_ratio = min(len(c_stripped), len(s_compact)) / max(len(c_stripped), len(s_compact))
            if len_ratio < 0.50:
                continue

            dice = dice_from_sets(c_ngrams, s_ngrams)
            if dice >= TAU_DICE:
                # Country check (must match)
                if s_ctry and c_ctry and s_ctry != c_ctry:
                    continue

                # Address sanity check: if both addresses exist and have numbers/tokens, ensure no harsh conflict
                s_digits = set(re.findall(r"\b\d+\b", s_addr))
                c_digits = set(re.findall(r"\b\d+\b", c_addr))
                if s_digits and c_digits:
                    # If both have street numbers/zip, require at least one overlap or no strict contradiction
                    if not (s_digits & c_digits) and len(s_digits) >= 2 and len(c_digits) >= 2:
                        continue

                proposed_links[cid].append((sid, dice))

    print(f"   Evaluated {pairs_evaluated:,} candidate pairs in {time.time()-t_eval:.2f}s")
    print(f"   Found {len(proposed_links):,} unique domain S3 candidates meeting tau >= {TAU_DICE}")

    # 6. Resolve any collisions (enforce strict disjoint 1-to-many constraint)
    print("\n6. Enforcing strict disjoint collision check ...")
    added_matches = 0
    conflicts_resolved = 0
    skipped_already_assigned = 0

    for cid, proposals in proposed_links.items():
        # Pick the best S1 for this candidate
        best_sid, best_dice = max(proposals, key=lambda x: x[1])

        # Check if already assigned in v7
        if cid in assigned_candidates:
            # Candidate was already assigned in v7 to assigned_candidates[cid]
            # Keep whichever has higher confidence: v7 existing assignments had high ensemble confidence (tau >= 0.88 or argmax)
            # Skip replacing to preserve precision
            skipped_already_assigned += 1
            continue

        # Add the match
        s1_matches[best_sid].append(cid)
        assigned_candidates[cid] = best_sid
        added_matches += 1
        if len(proposals) > 1:
            conflicts_resolved += 1

    print(f"   New matches added:                  {added_matches:,}")
    print(f"   Multi-proposal conflicts resolved:   {conflicts_resolved:,}")
    print(f"   Skipped (already assigned in v7):    {skipped_already_assigned:,}")

    # 7. Write updated output to temporary file
    print(f"\n7. Writing updated results to {PATCHED_TEMP_TSV} ...")
    t_w = time.time()
    final_empty_count = 0
    final_total_links = 0

    with open(PATCHED_TEMP_TSV, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_order:
            m = s1_matches[sid]
            if m:
                final_total_links += len(m)
                out_f.write(f"{sid}\t{','.join(m)}\n")
            else:
                final_empty_count += 1
                out_f.write(f"{sid}\t\n")

    print(f"   Written in {time.time()-t_w:.1f}s")
    print(f"   Final total links:         {final_total_links:,} (+{added_matches:,})")
    print(f"   Final empty singletons:    {final_empty_count:,} (was {empty_count_init:,})")

    # 8. Re-run false-positive collision check on the output file
    print("\n8. Re-running false-positive collision check across ALL 1,732,544 rows ...")
    cand_counts = Counter()
    with open(PATCHED_TEMP_TSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) > 1 and p[1]:
                for c in p[1].split(","):
                    cand_counts[c] += 1

    collisions = sum(1 for v in cand_counts.values() if v > 1)
    print(f"   Verification: Candidates assigned to >1 S1: {collisions} (MUST BE 0)")
    assert collisions == 0, f"Error: found {collisions} collisions!"

    # 9. Run official submission validator
    print("\n9. Running official submission validator ...")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results_v8_tmp.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    ret = os.system(val_cmd)

    if ret == 0:
        print("\nValidator PASSED! Overwriting output/matching_results.tsv with patched results ...")
        import shutil
        shutil.copyfile(PATCHED_TEMP_TSV, MATCHING_TSV)
        print("output/matching_results.tsv is now updated and ready for upload!")
    else:
        print(f"\nValidator returned non-zero exit code: {ret}")

    print(f"\nTotal Pipeline Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
