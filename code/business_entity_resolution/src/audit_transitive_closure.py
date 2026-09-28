"""
audit_transitive_closure.py
===========================
Rigorous empirical audit of the proposed intra-source duplicate transitive closure strategy.
Measures:
  1. Exact duplicate count in train_source2 and train_source3.
  2. Ground-truth hit rate: P(S1 matches C_B | S1 matches C_A and C_A == C_B).
  3. Reverse-risk rate: P(C_B is NOT matched to S1 | S1 matches C_A and C_A == C_B).
  4. Cross-collision rate: P(C_B matches S1_diff | S1 matches C_A and C_A == C_B).
  5. Franchise/chain stress test: identical business names across different addresses vs same address.
"""

import os, sys, time
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_name, normalize_address

TRAIN_GT = os.path.join(PROJECT_ROOT, "dataset", "train", "train_ground_truth.tsv")
TRAIN_S2 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source3.tsv")

def run_audit(sample_size=300000):
    print("=" * 70)
    print("EMPIRICAL AUDIT: Intra-Source Duplicate Transitive Closure")
    print(f"Sample size: {sample_size:,} rows per source file")
    print("=" * 70)

    # 1. Invert Ground Truth
    t0 = time.time()
    print("1. Indexing train_ground_truth.tsv ...", flush=True)
    cand_to_s1 = {}
    s1_to_cands = {}
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1].strip():
                continue
            s1 = parts[0]
            cands = set(c.strip() for c in parts[1].split(",") if c.strip())
            s1_to_cands[s1] = cands
            for c in cands:
                cand_to_s1[c] = s1

    print(f"   Indexed in {time.time()-t0:.1f}s | {len(s1_to_cands):,} S1s | {len(cand_to_s1):,} matched candidates", flush=True)

    # 2. Audit Source 2 and Source 3
    for src_name, src_path in [("Source 2", TRAIN_S2), ("Source 3", TRAIN_S3)]:
        print(f"\n2. Scanning {src_name} ({os.path.basename(src_path)}) for exact duplicates ...", flush=True)
        t_src = time.time()
        
        # Exact duplicate definition: exact match on normalized name AND normalized address AND country
        key_to_ids = defaultdict(list)
        name_to_records = defaultdict(list) # for franchise analysis
        
        n_scanned = 0
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                n_scanned += 1
                if n_scanned > sample_size:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                cid, name, addr = parts[0], parts[1], parts[2]
                country = parts[3] if len(parts) > 3 else ""
                
                norm_n = normalize_name(name)
                norm_a = normalize_address(addr)
                
                key = (norm_n, norm_a, country)
                key_to_ids[key].append(cid)
                name_to_records[norm_n].append((cid, norm_a, country))
                
                if n_scanned % 100000 == 0:
                    print(f"   Scanned {n_scanned:,} rows ({time.time()-t_src:.1f}s) ...", flush=True)

        # Count duplicate groups
        dup_groups = {k: v for k, v in key_to_ids.items() if len(v) > 1}
        total_dup_pairs = sum(len(v) * (len(v) - 1) // 2 for v in dup_groups.values())
        print(f"\n   Results for {src_name} ({n_scanned:,} records analyzed):")
        print(f"   - Unique (Name, Addr, Country) clusters: {len(key_to_ids):,}")
        print(f"   - Exact-Duplicate clusters (size > 1):   {len(dup_groups):,}")
        print(f"   - Total Exact-Duplicate pairs:            {total_dup_pairs:,}")

        # Check against ground truth
        both_match_same_s1 = 0
        one_matches_other_unmatched = 0
        match_different_s1 = 0
        neither_matched = 0
        
        anchor_pairs_evaluated = 0

        for key, ids in dup_groups.items():
            # For each pair in the group
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    c1, c2 = ids[i], ids[j]
                    s1_1 = cand_to_s1.get(c1)
                    s1_2 = cand_to_s1.get(c2)
                    
                    if s1_1 is None and s1_2 is None:
                        neither_matched += 1
                    elif s1_1 is not None and s1_2 is not None:
                        anchor_pairs_evaluated += 1
                        if s1_1 == s1_2:
                            both_match_same_s1 += 1
                        else:
                            match_different_s1 += 1
                    else:
                        # Exactly one is matched to an S1, the other is NOT matched to any S1!
                        anchor_pairs_evaluated += 1
                        one_matches_other_unmatched += 1

        print(f"\n   Ground-Truth Verification of Exact Duplicates ({src_name}):")
        print(f"   - Neither entity in ground truth:           {neither_matched:,} pairs")
        print(f"   - Pairs where at least one is a GT match:   {anchor_pairs_evaluated:,} pairs")
        
        if anchor_pairs_evaluated > 0:
            hit_rate = (both_match_same_s1 / anchor_pairs_evaluated) * 100.0
            fp_unmatched_rate = (one_matches_other_unmatched / anchor_pairs_evaluated) * 100.0
            collision_rate = (match_different_s1 / anchor_pairs_evaluated) * 100.0
            print(f"     * TRUE TRANSITIVE HIT (both match same S1):  {both_match_same_s1:,} ({hit_rate:.2f}%)")
            print(f"     * REVERSE-RISK FALSE POSITIVE (one matched): {one_matches_other_unmatched:,} ({fp_unmatched_rate:.2f}%)")
            print(f"     * CATASTROPHIC COLLISION (different S1s):   {match_different_s1:,} ({collision_rate:.2f}%)")
        else:
            print("     * No anchor pairs found in ground truth sample.")

        # Franchise / Chain Stress Test
        print(f"\n   Franchise / Chain Analysis ({src_name}):")
        chain_names = {n: recs for n, recs in name_to_records.items() if len(recs) >= 3 and len(set(r[1] for r in recs)) >= 2}
        print(f"   - Business names appearing at multiple distinct addresses: {len(chain_names):,}")
        
        # Check if chain businesses with identical address match different S1s
        chain_same_addr_diff_s1 = 0
        chain_eval_count = 0
        for n, recs in chain_names.items():
            addr_groups = defaultdict(list)
            for cid, norm_a, ctry in recs:
                addr_groups[(norm_a, ctry)].append(cid)
            for (norm_a, ctry), cids in addr_groups.items():
                if len(cids) > 1:
                    for i in range(len(cids)):
                        for j in range(i + 1, len(cids)):
                            c1, c2 = cids[i], cids[j]
                            s1_1 = cand_to_s1.get(c1)
                            s1_2 = cand_to_s1.get(c2)
                            if s1_1 and s1_2:
                                chain_eval_count += 1
                                if s1_1 != s1_2:
                                    chain_same_addr_diff_s1 += 1
        print(f"   - Chain instances sharing exact address evaluated: {chain_eval_count:,}")
        print(f"   - Chain instances sharing exact address matching DIFFERENT S1s: {chain_same_addr_diff_s1:,}")

    print("\n" + "=" * 70)
    print("AUDIT EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_audit(sample_size=300000)
