#!/usr/bin/env python3
"""
merge_ultra_high_p995_v20.py
============================
Amazon ML Challenge 2026 – Ultra-High-Confidence ML Ensemble Merge (v20)

Philosophy:
1. Base Invariant: 100% preserves every link from v16 (Score: 0.86115).
2. Ultra-High Confidence Threshold: Accepts ONLY candidates with ML ensemble probability p >= 0.995
   from the 34-feature LightGBM (2000 trees) + XGBoost (1000 trees) model.
3. Single-Winner Rule: At most ONE high-confidence candidate added per entity.
4. Strictly Disjoint: Highest-probability claim wins globally, guaranteeing 0 collisions.
5. Verified by official submission validator.
"""

import os
import sys
import shutil
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(".")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset", "test")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")

BASE_TSV       = os.path.join(OUTPUT_DIR, "matching_results_v16_deep.tsv")
RAW_TOP50_TSV  = os.path.join(OUTPUT_DIR, "matching_results_top50_raw.tsv")
CANDIDATE_TSV  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
OUT_V20_TSV    = os.path.join(OUTPUT_DIR, "matching_results_v20_ultra.tsv")
FINAL_TSV      = os.path.join(OUTPUT_DIR, "matching_results.tsv")

PROB_THRESHOLD = 0.995


def main():
    print("=" * 80, flush=True)
    print("  AMAZON ML CHALLENGE 2026: ULTRA-HIGH CONFIDENCE ML MERGE (v20)", flush=True)
    print(f"  Base: v16 (0.86115) + ML Ensemble Predictions (p >= {PROB_THRESHOLD})", flush=True)
    print("=" * 80, flush=True)

    # 1. Load base submission (v16: 0.86115)
    print("\n1. Loading base submission (v16: 0.86115) ...", flush=True)
    s1_ordered = []
    base_matches: Dict[str, List[str]] = {}
    assigned_candidates: Set[str] = set()

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
            else:
                base_matches[sid] = []

    print(f"   Total S1 entities:            {len(s1_ordered):,}", flush=True)
    print(f"   Base non-empty entities:      {sum(1 for v in base_matches.values() if v):,}", flush=True)
    print(f"   Base total links:             {sum(len(v) for v in base_matches.values()):,}", flush=True)
    print(f"   Already assigned candidates:  {len(assigned_candidates):,}", flush=True)

    # 2. Scan raw ML predictions with prob >= 0.995
    print(f"\n2. Scanning {os.path.basename(RAW_TOP50_TSV)} for p >= {PROB_THRESHOLD} ...", flush=True)
    claims: List[Tuple[str, str, float]] = []  # (sid, cid, prob)
    total_raw_rows = 0

    with open(RAW_TOP50_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            total_raw_rows += 1
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if len(p) > 1 and p[1].strip():
                for item in p[1].split(","):
                    item = item.strip()
                    if ":" in item:
                        cid, prob_str = item.split(":", 1)
                        try:
                            prob = float(prob_str)
                            if prob >= PROB_THRESHOLD and cid not in assigned_candidates:
                                claims.append((sid, cid, prob))
                        except ValueError:
                            pass

    print(f"   Raw rows scanned:             {total_raw_rows:,}", flush=True)
    print(f"   Candidate claims (p >= {PROB_THRESHOLD}): {len(claims):,}", flush=True)

    # 3. Disjoint collision resolution (highest probability wins)
    print("\n3. Resolving claims: highest probability wins ...", flush=True)
    claims.sort(key=lambda x: x[2], reverse=True)

    winner_for_cand: Dict[str, Tuple[str, float]] = {}
    purged_collisions = 0

    for sid, cid, prob in claims:
        if cid not in winner_for_cand and cid not in assigned_candidates:
            winner_for_cand[cid] = (sid, prob)
        else:
            purged_collisions += 1

    rescued_by_s1: Dict[str, List[str]] = defaultdict(list)
    for cid, (sid, prob) in winner_for_cand.items():
        # Conservative single-winner policy: at most 1 new candidate per entity
        if len(rescued_by_s1[sid]) < 1:
            rescued_by_s1[sid].append(cid)

    total_added = sum(len(v) for v in rescued_by_s1.values())
    print(f"   Collisions purged:            {purged_collisions:,}", flush=True)
    print(f"   Unique candidates accepted:   {total_added:,}", flush=True)
    print(f"   Entities receiving new link:  {len(rescued_by_s1):,}", flush=True)

    # 4. Write v20 final TSV
    print(f"\n4. Writing v20 submission to {os.path.basename(OUT_V20_TSV)} ...", flush=True)
    final_non_empty = 0
    final_empty = 0
    total_final_links = 0

    with open(OUT_V20_TSV, "w", encoding="utf-8") as f_out:
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
    print(f"   Total links in v20:       {total_final_links:,} (+{total_added:,} vs v16)", flush=True)

    # 5. Audit Collisions
    print("\n5. Auditing submission for multi-assignment collisions ...", flush=True)
    check_cands = Counter()
    with open(OUT_V20_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                for c in p[1].split(","):
                    check_cands[c.strip()] += 1

    collisions = sum(1 for c, cnt in check_cands.items() if cnt > 1)
    print(f"   Total unique candidates assigned: {len(check_cands):,}")
    print(f"   Total multi-assignment collisions: {collisions} (Must be 0)")
    assert collisions == 0, f"FATAL: Found {collisions} collisions in v20!"

    # 6. Overwrite matching_results.tsv with validated v20
    print(f"\n6. Promoting v20 to production {os.path.basename(FINAL_TSV)} ...", flush=True)
    shutil.copy2(OUT_V20_TSV, FINAL_TSV)

    # 7. Official Validator
    print("\n7. Running official utils/validate_submission.py ...", flush=True)
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

    print(f"\n=== v20 ULTRA-HIGH CONFIDENCE ML MERGE COMPLETE ===")
    print(f"Ready for portal upload: {FINAL_TSV}")


if __name__ == "__main__":
    main()
