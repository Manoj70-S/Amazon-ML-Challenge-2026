"""
run_parallel_v2.py
==================
Master script for fixed parallel inference pipeline.

Steps:
  1. Repartition candidate_pairs.tsv into 12 parts (uses all 12 CPU cores)
  2. Run prebuild_pickles.py to create shared S1 + cand pickle files
  3. Launch 12 worker processes (worker_infer_v2.py) in parallel
  4. Concatenate all match_part_XX.tsv files -> output/matching_results.tsv
  5. Report final stats
"""

import os, sys, subprocess, time, glob

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
PARTS_DIR    = os.path.join(PROJECT_ROOT, "cache", "parts")
OUTPUT_FILE  = os.path.join(PROJECT_ROOT, "output", "matching_results.tsv")
CAND_FILE    = os.path.join(PROJECT_ROOT, "output", "candidate_pairs.tsv")
PYTHON       = sys.executable

NUM_PARTS    = 12
THRESHOLD    = 0.940
WORKER_SCRIPT = os.path.join(SRC_DIR, "worker_infer_v2.py")
PREBUILD     = os.path.join(SRC_DIR, "prebuild_pickles.py")

# Force unbuffered stdout
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)


def repartition(num_parts: int):
    """Split candidate_pairs.tsv into num_parts equal chunks."""
    print(f"\n{'='*60}")
    print(f"Step 1: Repartitioning into {num_parts} parts ...")
    os.makedirs(PARTS_DIR, exist_ok=True)

    # Count total lines (excluding header)
    print("  Counting lines in candidate_pairs.tsv ...")
    with open(CAND_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        total  = sum(1 for _ in f)
    print(f"  Total S1 queries: {total:,}")

    chunk = (total + num_parts - 1) // num_parts
    part_files = []

    with open(CAND_FILE, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for part_idx in range(num_parts):
            part_file = os.path.join(PARTS_DIR, f"cand_part_{part_idx:02d}.tsv")
            part_files.append(part_file)
            count = 0
            with open(part_file, "w", encoding="utf-8") as fout:
                for line in f:
                    fout.write(line)
                    count += 1
                    if count >= chunk:
                        break
            print(f"  Part {part_idx:02d}: {count:,} queries -> {part_file}")

    print(f"  Repartition done.")
    return part_files


def prebuild():
    """Run prebuild_pickles.py to create shared S1+cand pickle files."""
    print(f"\n{'='*60}")
    print("Step 2: Building shared pickle files ...")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, PREBUILD],
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: prebuild_pickles.py failed with code {result.returncode}")
        sys.exit(1)
    print(f"  Prebuild done in {time.time()-t0:.1f}s")


def launch_workers(part_files: list):
    """Launch all workers in parallel, stream their output."""
    print(f"\n{'='*60}")
    print(f"Step 3: Launching {len(part_files)} workers ...")
    t0 = time.time()

    procs   = []
    log_fhs = []

    for idx, cand_file in enumerate(part_files):
        match_file = os.path.join(PARTS_DIR, f"match_part_{idx:02d}.tsv")
        log_file   = os.path.join(PARTS_DIR, f"worker_v2_{idx:02d}.log")
        log_fh     = open(log_file, "w", encoding="utf-8")
        log_fhs.append(log_fh)

        cmd = [PYTHON, WORKER_SCRIPT,
               str(idx), cand_file, match_file, str(THRESHOLD)]
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        procs.append((idx, proc, log_file, match_file))
        print(f"  Worker {idx:02d} started (PID {proc.pid})")

    # Poll until all done
    print("\nWaiting for workers to finish ...")
    while True:
        time.sleep(30)
        done    = []
        running = []
        failed  = []
        for idx, proc, log_file, match_file in procs:
            rc = proc.poll()
            if rc is None:
                running.append(idx)
            elif rc == 0:
                done.append(idx)
            else:
                failed.append((idx, rc))

        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] Done: {done} | Running: {running} | Failed: {[i for i,_ in failed]}")

        if not running:
            break

    for fh in log_fhs:
        fh.close()

    if failed:
        print(f"\nERROR: {len(failed)} workers failed!")
        for idx, rc in failed:
            log_file = os.path.join(PARTS_DIR, f"worker_v2_{idx:02d}.log")
            print(f"\n  Worker {idx:02d} (rc={rc}) log tail:")
            try:
                with open(log_file) as lf:
                    lines = lf.readlines()
                    for l in lines[-15:]:
                        print("   ", l.rstrip())
            except Exception as e:
                print(f"   (could not read log: {e})")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n  All {len(procs)} workers finished in {elapsed:.1f}s")
    return [mf for _, _, _, mf in procs]


def concatenate(match_files: list):
    """Concatenate all part files into final output."""
    print(f"\n{'='*60}")
    print(f"Step 4: Concatenating {len(match_files)} part files ...")
    t0 = time.time()

    total_matches = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tsource_entity_id\n")
        for mf in match_files:
            if not os.path.exists(mf):
                print(f"  WARNING: {mf} not found!")
                continue
            count = 0
            with open(mf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    count += 1
            total_matches += count
            print(f"  {os.path.basename(mf)}: {count:,} matches")

    print(f"\n  Total matches: {total_matches:,}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Concatenation done in {time.time()-t0:.1f}s")
    return total_matches


def main():
    t_start = time.time()
    print("=" * 60)
    print("Fixed Parallel Inference Pipeline v2")
    print(f"  Workers:   {NUM_PARTS}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Output:    {OUTPUT_FILE}")
    print("=" * 60)

    # Step 1: Repartition
    part_files = repartition(NUM_PARTS)

    # Step 2: Prebuild pickles
    prebuild()

    # Step 3: Launch workers
    match_files = launch_workers(part_files)

    # Step 4: Concatenate
    total = concatenate(match_files)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {OUTPUT_FILE}  ({total:,} matched pairs)")
    print("="*60)


if __name__ == "__main__":
    main()
