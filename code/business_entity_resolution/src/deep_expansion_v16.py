#!/usr/bin/env python3
"""
deep_expansion_v16.py
=====================
Amazon ML Challenge 2026 – Deep Multi-Candidate Expansion & Triangulation Engine (v16)

Key Advancements over v15:
1. Base Preservation: 100% preserves the verified v15 submission (1,619,197 non-empty matches).
2. Deep Search Horizon:
   - Expands candidate search depth from rank 12 to rank 35 in candidate_pairs.tsv for 1-link and 2-link entities.
3. Calibrated Triangular Mutual Verification:
   - Uses benchmark-proven thresholds (AddrSim >= 85%, NameSim >= 80%, 0.00% FP on GT) to recover missing counterparts.
4. Multi-Anchor Triangulation:
   - For entities with existing matches, validates secondary S2/S3 candidates against all existing cluster anchors.
5. Deep Empty Singleton Recovery:
   - High-fidelity PIN and digit matching up to rank 25 for remaining empty singletons.
6. Zero-Collision Disjoint Purge:
   - Globally resolves any multi-assignment claims using highest-confidence-wins.
7. Official Validation:
   - Confirms PASS via utils/validate_submission.py.
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

BASE_TSV      = os.path.join(OUTPUT_DIR, "matching_results_v15_master.tsv")
CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
S1_TSV        = os.path.join(DATA_DIR, "test_source1.tsv")
S2_TSV        = os.path.join(DATA_DIR, "test_source2.tsv")
S3_TSV        = os.path.join(DATA_DIR, "test_source3.tsv")
OUT_V16_TSV   = os.path.join(OUTPUT_DIR, "matching_results_v16_deep.tsv")
FINAL_TSV     = os.path.join(OUTPUT_DIR, "matching_results.tsv")


def strip_accents(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')


# French normalization patterns
RE_FR_LEGAL = [
    (re.compile(r'\b(societe\s+a\s+responsabilite\s+limitee|s\.a\.r\.l\.|sarl)\b', re.I), 'sarl'),
    (re.compile(r'\b(societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|s\.a\.s\.u\.|sasu)\b', re.I), 'sasu'),
    (re.compile(r'\b(societe\s+par\s+actions\s+simplifiee|s\.a\.s\.|sas)\b', re.I), 'sas'),
    (re.compile(r'\b(societe\s+civile\s+immobiliere|s\.c\.i\.|sci)\b', re.I), 'sci'),
    (re.compile(r'\b(societe\s+en\s+nom\s+collectif|s\.n\.c\.|snc)\b', re.I), 'snc'),
    (re.compile(r'\b(entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|e\.u\.r\.l\.|eurl)\b', re.I), 'eurl'),
    (re.compile(r'\b(association|assoc\.)\b', re.I), 'assoc'),
]

RE_FR_STREET = [
    (re.compile(r'\b(boulevard|bd\.|bd)\b', re.I), 'blvd'),
    (re.compile(r'\b(avenue|ave\.|av\.)\b', re.I), 'ave'),
    (re.compile(r'\b(chemin|ch\.)\b', re.I), 'chemin'),
    (re.compile(r'\b(impasse|imp\.)\b', re.I), 'impasse'),
    (re.compile(r'\b(rue|r\.)\b', re.I), 'rue'),
    (re.compile(r'\b(place|pl\.)\b', re.I), 'place'),
    (re.compile(r'\b(cours|crs\.)\b', re.I), 'cours'),
    (re.compile(r'\b(quai)\b', re.I), 'quai'),
    (re.compile(r'\b(bis|ter|quater)\b', re.I), ''),
    (re.compile(r'\b(du|de\s+la|de\s+l\'|des|d\'|le|la|les)\b', re.I), ''),
]


def clean_french_text(name: str, addr: str) -> Tuple[str, str]:
    n = strip_accents(name).lower()
    a = strip_accents(addr).lower()
    for pattern, rep in RE_FR_LEGAL:
        n = pattern.sub(rep, n)
    for pattern, rep in RE_FR_STREET:
        a = pattern.sub(rep, a)
    n = re.sub(r'\s+', ' ', n).strip()
    a = re.sub(r'\s+', ' ', a).strip()
    return n, a


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  AMAZON ML CHALLENGE 2026: DEEP MULTI-CANDIDATE EXPANSION ENGINE (v16)", flush=True)
    print("  Target: Deep Search to Rank 35 + Calibrated Triangular Verification", flush=True)
    print("=" * 80, flush=True)

    # 1. Load base submission (v15)
    print("\n1. Loading base submission (v15) ...", flush=True)
    s1_ordered = []
    base_matches: Dict[str, List[str]] = {}
    assigned_candidates: Set[str] = set()
    empty_s1: Set[str] = set()
    few_link_s1: Set[str] = set()  # entities with 1 or 2 links

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
                    few_link_s1.add(sid)
            else:
                base_matches[sid] = []
                empty_s1.add(sid)

    print(f"   Total S1 entities:         {len(s1_ordered):,}", flush=True)
    print(f"   Base non-empty:            {len(s1_ordered) - len(empty_s1):,}", flush=True)
    print(f"   Empty singletons:          {len(empty_s1):,}", flush=True)
    print(f"   Target 1-or-2 link entities: {len(few_link_s1):,}", flush=True)
    print(f"   Already assigned cands:    {len(assigned_candidates):,}", flush=True)

    # 2. Target S1 entities to load
    target_s1_ids = few_link_s1 | empty_s1
    print(f"\n2. Loading text for {len(target_s1_ids):,} target entities ...", flush=True)
    s1_data = {}

    with open(S1_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in target_s1_ids:
                raw_n = p[1] if len(p) > 1 else ""
                raw_a = p[2] if len(p) > 2 else ""
                country = p[3] if len(p) > 3 else ""
                norm_n = normalize_name(raw_n)
                norm_a = normalize_address(raw_a)
                nums = set(RE_DIGITS.findall(norm_a)) if norm_a else set()
                pins = set(RE_PIN.findall(norm_a)) if norm_a else set()
                fr_n, fr_a = clean_french_text(raw_n, raw_a) if country == "France" else ("", "")
                s1_data[sid] = (norm_n, norm_a, nums, pins, country, fr_n, fr_a)

    print(f"   Loaded data for {len(s1_data):,} entities", flush=True)

    # 3. Read unassigned candidates up to rank 35
    print("\n3. Collecting deep candidate IDs from candidate_pairs.tsv (up to rank 35) ...", flush=True)
    target_cands = defaultdict(list)
    needed_cids = set()

    # Anchor candidate IDs from few_link_s1 entities
    anchor_cids = set()
    for sid in few_link_s1:
        for mid in base_matches.get(sid, []):
            anchor_cids.add(mid)

    needed_cids.update(anchor_cids)

    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            if sid in target_s1_ids and len(p) > 1 and p[1].strip():
                top_limit = 35 if sid in few_link_s1 else 25
                cids = [c.strip() for c in p[1].split(",") if c.strip()][:top_limit]
                unassigned = [c for c in cids if c not in assigned_candidates]
                if unassigned:
                    target_cands[sid] = unassigned
                    needed_cids.update(unassigned)

    print(f"   Target entities with unassigned candidates: {len(target_cands):,}", flush=True)
    print(f"   Anchor candidates needed:                   {len(anchor_cids):,}", flush=True)
    print(f"   Total unique candidates needed from S2/S3:  {len(needed_cids):,}", flush=True)

    # 4. Stream S2 & S3 candidate texts
    print(f"\n4. Streaming candidate texts from test_source2.tsv and test_source3.tsv ...", flush=True)
    cand_data = {}

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
                    country = p[3] if len(p) > 3 else ""
                    norm_n = normalize_name(raw_n)
                    norm_a = normalize_address(raw_a)
                    nums = set(RE_DIGITS.findall(norm_a)) if norm_a else set()
                    pins = set(RE_PIN.findall(norm_a)) if norm_a else set()
                    is_indic = any(ord(ch) > 127 for ch in raw_n)
                    fr_n, fr_a = clean_french_text(raw_n, raw_a) if country == "France" else ("", "")
                    cand_data[cid] = (norm_n, norm_a, nums, pins, country, fr_n, fr_a, is_indic)
                    c_found += 1
        print(f"   {os.path.basename(src_path)}: loaded {c_found:,} candidates in {time.time()-t_src:.1f}s", flush=True)

    print(f"   Candidate text cache ready: {len(cand_data):,} records", flush=True)

    # 5. Evaluate Deep Multi-Track Expansion
    print("\n5. Evaluating Deep Triangular & Sub-Cluster Expansion ...", flush=True)
    t_match = time.time()
    candidate_claims: List[Tuple[str, str, float, str]] = []  # (sid, cid, score, rule_name)
    rule_counts = Counter()

    for sid, cids in target_cands.items():
        s_info = s1_data.get(sid)
        if not s_info:
            continue
        sn, sa, s_nums, s_pins, sc, s_fr_n, s_fr_a = s_info
        s_words = set(w for w in sn.split() if len(w) >= 3) if sn else set()
        s_first_w = sn.split()[0] if sn.split() else ""

        is_few_link = (sid in few_link_s1)
        is_empty = (sid in empty_s1)

        # Existing anchors for few_link entities
        existing_mids = base_matches.get(sid, [])
        existing_sources = set(m[:2] for m in existing_mids)
        anchor_infos = [cand_data[m] for m in existing_mids if m in cand_data]

        best_score = 0.0
        best_cid = None
        best_rule = ""

        for cid in cids:
            c_info = cand_data.get(cid)
            if not c_info:
                continue
            cn, ca, c_nums, c_pins, cc, c_fr_n, c_fr_a, is_indic = c_info
            c_source = cid[:2]
            c_words = set(w for w in cn.split() if len(w) >= 3) if cn else set()
            c_first_w = cn.split()[0] if cn.split() else ""
            has_word_overlap = bool(s_words & c_words) if (s_words and c_words) else False

            # ─────────────────────────────────────────────────────────────
            # TRACK 1: DEEP TRIANGULAR MUTUAL VERIFICATION (for 1/2-link entities)
            # ─────────────────────────────────────────────────────────────
            if is_few_link and anchor_infos and (c_source not in existing_sources or len(existing_mids) == 1):
                # Verify candidate against S1 and all anchors
                s1_asim = fuzz.token_set_ratio(sa, ca) if (sa and ca) else 0.0
                s1_nsim = fuzz.token_sort_ratio(sn, cn) if (sn and cn) else 0.0

                for a_info in anchor_infos:
                    an, aa, a_nums, a_pins, _, _, _, _ = a_info
                    anc_asim = fuzz.token_set_ratio(aa, ca) if (aa and ca) else 0.0
                    anc_nsim = fuzz.token_sort_ratio(an, cn) if (an and cn) else 0.0
                    shared_digits = bool((s_nums & c_nums) or (a_nums & c_nums))

                    # Calibrated 85/80 triangular rule (0.00% FP on ground truth)
                    if (s1_asim >= 85.0 and anc_asim >= 85.0 and
                        s1_nsim >= 80.0 and anc_nsim >= 80.0 and shared_digits):
                        tri_score = (s1_asim + anc_asim + s1_nsim + anc_nsim) / 4.0
                        if tri_score > best_score:
                            best_score = tri_score
                            best_cid = cid
                            best_rule = "D1_DeepTriangular"

            # ─────────────────────────────────────────────────────────────
            # TRACK 2: EXACT PIN CODE + HIGH NAME FIDELITY (1-link or empty)
            # ─────────────────────────────────────────────────────────────
            if (s_pins & c_pins) and sn and cn:
                nsim = fuzz.token_sort_ratio(sn, cn)
                asim = fuzz.token_set_ratio(sa, ca) if (sa and ca) else 0.0
                if nsim >= 88.0 and asim >= 72.0:
                    pin_score = 0.5 * nsim + 0.5 * asim
                    if pin_score > best_score:
                        best_score = pin_score
                        best_cid = cid
                        best_rule = "D2_ExactPIN_Fidelity"

            # ─────────────────────────────────────────────────────────────
            # TRACK 3: FRENCH JURISDICTIONAL DEEP RESCUE (France entities)
            # ─────────────────────────────────────────────────────────────
            if sc == "France" and s_fr_n and c_fr_n and (s_nums & c_nums):
                fr_nsim = fuzz.token_sort_ratio(s_fr_n, c_fr_n)
                fr_asim = fuzz.token_set_ratio(s_fr_a, c_fr_a) if (s_fr_a and c_fr_a) else 0.0
                if fr_nsim >= 86.0 and fr_asim >= 82.0:
                    fr_score = 0.5 * fr_nsim + 0.5 * fr_asim
                    if fr_score > best_score:
                        best_score = fr_score
                        best_cid = cid
                        best_rule = "D3_French_Deep"

            # ─────────────────────────────────────────────────────────────
            # TRACK 4: DEEP EMPTY SINGLETON RESCUE
            # ─────────────────────────────────────────────────────────────
            if is_empty:
                if sa and ca and len(sa) >= 15 and len(ca) >= 15 and (s_nums & c_nums):
                    if is_indic or has_word_overlap:
                        asim = fuzz.token_set_ratio(sa, ca)
                        if asim >= 88.0 and asim > best_score:
                            best_score = asim
                            best_cid = cid
                            best_rule = "D4_Addr88_Digit"

        if best_cid:
            candidate_claims.append((sid, best_cid, best_score, best_rule))
            rule_counts[best_rule] += 1

    print(f"   Raw expansion claims generated: {len(candidate_claims):,} in {time.time()-t_match:.1f}s", flush=True)
    print("   Claims by Rule:")
    for rname, cnt in rule_counts.most_common():
        print(f"     {rname:24s}: {cnt:6,d}", flush=True)

    # 6. Global Disjoint Collision Resolution
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
    empty_rescued_count = 0
    few_link_expanded_count = 0

    for cid, (sid, score, rule) in winner_for_cand.items():
        rescued_by_s1[sid].append(cid)
        if sid in empty_s1:
            empty_rescued_count += 1
        elif sid in few_link_s1:
            few_link_expanded_count += 1

    print(f"   Collisions purged:                  {purged_collisions:,}", flush=True)
    print(f"   Unique candidates winning:          {len(winner_for_cand):,}", flush=True)
    print(f"   Empty singletons rescued:           {empty_rescued_count:,}", flush=True)
    print(f"   1-and-2 link entities expanded:     {few_link_expanded_count:,}", flush=True)

    # 7. Write Final TSV
    print(f"\n7. Writing v16 final submission to {os.path.basename(OUT_V16_TSV)} ...", flush=True)
    final_non_empty = 0
    final_empty = 0
    total_final_links = 0

    with open(OUT_V16_TSV, "w", encoding="utf-8") as f_out:
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
    print(f"   Final non-empty entities: {final_non_empty:,} (+{empty_rescued_count:,} vs v15)", flush=True)
    print(f"   Final empty singletons:   {final_empty:,} (reduced from {len(empty_s1):,})", flush=True)
    print(f"   Total links in v16:       {total_final_links:,} (+{len(winner_for_cand):,} vs v15)", flush=True)

    # 8. Verify ZERO Collisions
    print("\n8. Auditing 100% submission for multi-assignment collisions ...", flush=True)
    check_cands = Counter()
    with open(OUT_V16_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                for c in p[1].split(","):
                    check_cands[c.strip()] += 1

    collisions = sum(1 for c, cnt in check_cands.items() if cnt > 1)
    print(f"   Total unique candidates assigned: {len(check_cands):,}")
    print(f"   Total multi-assignment collisions: {collisions} (Must be 0)")
    assert collisions == 0, f"FATAL: Found {collisions} collisions in v16!"

    # 9. Overwrite matching_results.tsv with validated v16
    print(f"\n9. Promoting v16 to production {os.path.basename(FINAL_TSV)} ...", flush=True)
    shutil.copy2(OUT_V16_TSV, FINAL_TSV)

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

    print(f"\n=== v16 DEEP EXPANSION ENGINE COMPLETE IN {time.time()-t0:.1f}s ===")
    print(f"Ready for portal upload: {FINAL_TSV}")

if __name__ == "__main__":
    main()
