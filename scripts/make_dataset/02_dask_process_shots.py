#!/usr/bin/env python
"""Submit shot-processing tasks to a Dask cluster.

Each task = one shot. Workers call `morph-gs-process-shot` (package CLI)
as a subprocess and write per-shot NPZ to <shots-dir> atomically.

Usage:
    uv run python scripts/dask_process_shots.py \
        --scheduler tcp://<HOST>:8786 \
        --shot-list experiments/12_dataset_generation/results/shot_list_v2.json \
        --shots-dir /DATALAKE/mast_gs/shots_v2

    # Dry-run:
    uv run python scripts/dask_process_shots.py \
        --scheduler tcp://localhost:8786 \
        --shot-list experiments/12_dataset_generation/results/shot_list_v2.json \
        --shots-dir /DATALAKE/mast_gs/shots_v2 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _worker_process_shot(shot_id: int, shots_dir_str: str, split: str, n_times: int,
                         project_root: str) -> dict:
    """Wrapper executed on Dask workers.

    Spawns a fresh subprocess to avoid numpy/xarray conflicts in the Dask
    worker process. Uses PYTHONPATH so morph_gs is importable regardless of
    the worker's default Python environment.
    """
    import ast
    import os
    import signal
    import subprocess
    import tempfile

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}/src:" + env.get("PYTHONPATH", "")

    cmd = [
        "/usr/local/bin/python", "-m", "morph_gs.process_shot",
        "--shot",    str(shot_id),
        "--out-dir", shots_dir_str,
        "--split",   split,
        "--n-times", str(n_times),
    ]

    # Write stdout/stderr to temp files to avoid pipe deadlock when
    # subprocesses (freegsnke/BLAS) keep the pipe open after being killed.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as fo, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.err', delete=False) as fe:
        out_path, err_path = fo.name, fe.name

    try:
        with open(out_path, 'w') as fo, open(err_path, 'w') as fe:
            proc = subprocess.Popen(
                cmd, stdout=fo, stderr=fe,
                cwd="/tmp", env=env,
                preexec_fn=os.setsid,
            )
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return {"shot": shot_id, "n_samples": 0, "n_skipped": n_times, "status": "timeout"}

        stdout = open(out_path).read()
        stderr = open(err_path).read()

        if proc.returncode != 0:
            return {
                "shot": shot_id, "n_samples": 0, "n_skipped": n_times,
                "status": f"subprocess_fail:{stderr[-300:]}",
            }

        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return ast.literal_eval(line)
                except Exception:
                    pass
        return {
            "shot": shot_id, "n_samples": 0, "n_skipped": n_times,
            "status": f"parse_fail:{stdout[-100:]}",
        }
    finally:
        for p in (out_path, err_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scheduler",   required=True,
                   help="Dask scheduler address, e.g. tcp://HOST:8786")
    p.add_argument("--shot-list",   required=True,
                   help="JSON file produced by 01_discover_shots.py")
    p.add_argument("--shots-dir",   required=True,
                   help="Output directory for per-shot NPZ files")
    p.add_argument("--n-times",     type=int, default=30,
                   help="Number of time slices to sample per shot (default: 30)")
    p.add_argument("--batch-size",  type=int, default=0,
                   help="Max shots to submit in this run (0 = all)")
    p.add_argument("--log-out",     default=None,
                   help="JSON file to write run log (default: <shots-dir>/run_log.json)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print tasks without submitting")
    return p.parse_args()


def main():
    args = parse_args()

    manifest = json.loads(Path(args.shot_list).read_text())
    splits_dict = manifest["split"]
    all_shots: list[tuple[int, str]] = []
    for split in ("train", "val", "test"):
        for shot in splits_dict.get(split, []):
            all_shots.append((shot, split))

    shots_dir = Path(args.shots_dir)
    shots_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (shot, split) for shot, split in all_shots
        if not (shots_dir / f"shot_{shot}.npz").exists()
    ]

    if args.batch_size > 0:
        pending = pending[: args.batch_size]

    print(f"Total shots in list : {len(all_shots)}")
    print(f"Already processed   : {len(all_shots) - len(pending)}")
    print(f"Submitting now      : {len(pending)}")

    if args.dry_run:
        for shot, split in pending[:10]:
            print(f"  [dry-run] process_shot({shot}, {shots_dir}, '{split}')")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    from dask.distributed import Client, as_completed

    project_root = str(Path(__file__).resolve().parents[2])

    client = Client(args.scheduler)
    print(f"Connected to Dask: {args.scheduler}")
    nt = client.nthreads()
    print(f"Workers: {len(nt)}, total_threads: {sum(nt.values())}")

    run_id = int(time.time())
    futures = {}
    for shot, split in pending:
        future = client.submit(
            _worker_process_shot,
            shot,
            str(shots_dir),
            split,
            args.n_times,
            project_root,
            key=f"shot-{shot}-{run_id}",
            pure=False,
        )
        futures[future] = shot

    results = []
    n_ok = n_fail = n_skip = 0
    for future in as_completed(futures):
        shot = futures[future]
        try:
            result = future.result()
        except Exception as e:
            result = {"shot": shot, "n_samples": 0, "n_skipped": args.n_times,
                      "status": f"worker_exception:{e}"}
            n_fail += 1

        status = result.get("status", "?")
        if status == "ok":
            n_ok += 1
            print(f"OK    shot={shot:6d}  n_samples={result['n_samples']}  "
                  f"elapsed={result.get('elapsed_s', '?')}s")
        elif status == "skipped":
            n_skip += 1
        else:
            n_fail += 1
            print(f"FAIL  shot={shot:6d}  status={status}")

        results.append(result)

    print(f"\nDone: {n_ok} ok  {n_fail} failed  {n_skip} skipped")

    log_path = Path(args.log_out) if args.log_out else shots_dir / "run_log.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
