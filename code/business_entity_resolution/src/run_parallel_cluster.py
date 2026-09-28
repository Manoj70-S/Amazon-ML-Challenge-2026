"""
run_parallel_cluster.py
=======================
Master Controller for 12-Worker High-Throughput Test Inference.
"""

import os
import sys
import time
import subprocess
from typing import List, Tuple

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache", "parts")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_TSV  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
WORKER_SCRIPT = os.path.join(SRC_DIR, "worker_infer.py")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def partition_candidate_file(num_workers: int = 12) -> List[Tuple[str, str]]:
    print(f"\n1. Partitioning candidate_pairs.tsv into {num_workers} parallel parts …")
    t0 = time.time()
    
    with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
        header = f.readline()
        lines = f.readlines()
        
    n_total = len(lines)
    chunk_size = (n_total + num_workers - 1) // num_workers
    print(f"   Total queries: {n_total:,} | Per worker: ~{chunk_size:,}")
    
    file_pairs = []
    for i in range(num_workers):
        part_cand_file = os.path.join(CACHE_DIR, f"cand_part_{i:02d}.tsv")
        part_match_file = os.path.join(CACHE_DIR, f"match_part_{i:02d}.tsv")
        
        start_idx = i * chunk_size
        end_idx = min(n_total, (i + 1) * chunk_size)
        part_lines = lines[start_idx:end_idx]
        
        with open(part_cand_file, "w", encoding="utf-8") as out_f:
            out_f.writelines(part_lines)
            
        file_pairs.append((part_cand_file, part_match_file))
        
    print(f"   Partitioned into {num_workers} files in {time.time()-t0:.1f}s -> {CACHE_DIR}")
    return file_pairs


def main(num_workers: int = 12, threshold: float = 0.940):
    t_start = time.time()
    print("====================================================================")
    print(f"  Amazon ML Challenge 2026: 12-Core Parallel Inference Cluster")
    print(f"  Target Threshold = {threshold:.3f} (Macro-F0.5 = 0.89924)")
    print("====================================================================")
    
    # 1. Partition input
    file_pairs = partition_candidate_file(num_workers=num_workers)
    
    # 2. Launch 12 workers in parallel
    print(f"\n2. Spawning {num_workers} independent worker processes …")
    procs = []
    log_handles = []
    
    for i, (cand_f, match_f) in enumerate(file_pairs):
        log_f = os.path.join(CACHE_DIR, f"worker_{i:02d}.log")
        lh = open(log_f, "w", encoding="utf-8")
        log_handles.append(lh)
        
        cmd = [
            sys.executable,
            "-u",
            WORKER_SCRIPT,
            str(i),
            cand_f,
            match_f,
            str(threshold),
        ]
        
        p = subprocess.Popen(
            cmd,
            stdout=lh,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        procs.append(p)
        print(f"   Launched Worker {i:02d} (PID {p.pid}) -> {log_f}")
        
    # 3. Monitor live progress until all complete
    print(f"\n3. Monitoring cluster execution across all {num_workers} workers …")
    while True:
        time.sleep(15)
        running = [p.poll() is None for p in procs]
        n_running = sum(running)
        
        # Read latest progress lines from all worker logs
        progress_summary = []
        for i in range(num_workers):
            log_f = os.path.join(CACHE_DIR, f"worker_{i:02d}.log")
            if os.path.exists(log_f):
                try:
                    with open(log_f, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        last = lines[-1]
                        if "Finished" in last:
                            progress_summary.append(f"W{i:02d}: DONE")
                        elif "%" in last:
                            # Extract percentage
                            perc = [tok for tok in last.split() if "%" in tok]
                            progress_summary.append(f"W{i:02d}: {perc[0] if perc else 'RUN'}")
                        else:
                            progress_summary.append(f"W{i:02d}: INIT")
                    else:
                        progress_summary.append(f"W{i:02d}: INIT")
                except:
                    progress_summary.append(f"W{i:02d}: RUN")
                    
        elapsed = time.time() - t_start
        print(f"  [{elapsed/60:.1f}m] Active Workers: {n_running}/{num_workers} | " + " | ".join(progress_summary[:6]))
        if len(progress_summary) > 6:
            print(f"        " + " | ".join(progress_summary[6:]))
            
        if n_running == 0:
            break
            
    for lh in log_handles:
        lh.close()
        
    # 4. Concatenate all part files into final matching_results.tsv
    print(f"\n4. Concatenating worker outputs into {MATCHING_TSV} …")
    t0 = time.time()
    n_total_lines = 0
    with open(MATCHING_TSV, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for _, match_f in file_pairs:
            with open(match_f, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    n_total_lines += 1
                    
    print(f"   Concatenated {n_total_lines:,} rows in {time.time()-t0:.1f}s -> {MATCHING_TSV}")
    print(f"\n>>> Total End-to-End Pipeline Runtime: {(time.time()-t_start)/60:.1f} minutes <<<")
    
    # 5. Run official submission validation
    print("\n5. Running official submission format validator …")
    val_cmd = (
        f'"{sys.executable}" utils/validate_submission.py '
        f'--matching output/matching_results.tsv '
        f'--candidate output/candidate_pairs.tsv '
        f'--test-dir dataset/test'
    )
    os.system(val_cmd)


if __name__ == "__main__":
    main(num_workers=8, threshold=0.940)
