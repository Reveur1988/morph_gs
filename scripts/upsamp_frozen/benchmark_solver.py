#!/usr/bin/env python
"""Scaling benchmark for MorphGSE warm-start pipeline.

Measures per-equilibrium solve times (cold and warm NK) across multiple model
seeds and reports T_total components for the bootstrap protocol (§4.4–§4.5).

Usage (Dask mode, recommended):
    UV_PROJECT_ENVIRONMENT=.venv_cpu01 UV_LINK_MODE=copy \\
    uv run --locked python scripts/morph_adapted/benchmark_solver.py \\
        --weights results/morph_warmup/pretrained/N1000/seed42/morphgse_best.pth \\
        --model-seeds 42,0,1 \\
        --n-pairs 100 --seed 42 --mode both \\
        --batch-size 256 \\
        --scheduler tcp://192.168.0.103:8786 \\
        --out reports/benchmark_100_5w.json

Sequential mode (single-worker baseline):
    uv run --locked python scripts/morph_adapted/benchmark_solver.py \\
        --weights results/morph_warmup/pretrained/N1000/seed42/morphgse_best.pth \\
        --model-seeds 42 --n-pairs 20 --seed 42 --mode both --batch-size 256 \\
        --out /tmp/bench_seq.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_EXP02_SCRIPTS = _PROJECT_ROOT / "experiments" / "02_freegsnke_efit_init" / "scripts"
for p in [str(_EXP02_SCRIPTS), str(_PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from config import NX, NY, RMIN, RMAX, ZMIN, ZMAX, SOLVER_TOL, SOLVER_MAXITS, SOLVER_PICARD_HANDOVER
from morph_gs import FieldStats, GSDatasetV2, MorphGSE
from morph_gs.dataset_v2 import FIELD_NAMES

from validate import (  # reuse helpers from validate.py (client-side only)
    _load_model,
    _predict_psi,
)

_PAD = 72


# ── Dask task: fully self-contained copy (no validate import on workers) ───────

def _dask_solve_pair(pair_idx, idx, shot_id, solver_inputs, psi_pred):
    """Cold + warm GS solve via subprocess with timeout.

    All imports are local — cloudpickle sees no deps on validate module.
    Records t_task_start / t_task_end (wall-clock on worker) and worker_host
    so the caller can compute effective parallelism from timestamps.
    """
    import os
    import pickle
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    _root      = Path("/DATALAKE/mast_gs/mt_experiments")
    _helper    = str(_root / "scripts" / "_gs_solve_worker.py")
    _timeout_s = 120

    t_task_start = time.time()
    _worker_host = os.uname().nodename

    _nan  = float("nan")
    _fail = {
        "idx": idx, "shot_id": shot_id,
        "iters_cold": -1, "iters_warm": -1,
        "ratio": _nan,
        "t_cold_s": 0.0, "t_warm_s": 0.0,
        "converged_cold": False, "converged_warm": False,
        "converged_both": False, "psi_consistent": "",
        "t_task_start": t_task_start, "t_task_end": t_task_start,
        "worker_host": _worker_host,
    }

    try:
        with tempfile.NamedTemporaryFile(suffix=".in.pkl",  delete=False) as fi:
            in_path = fi.name
            pickle.dump({"solver_inputs": solver_inputs, "psi_pred": psi_pred}, fi)
        with tempfile.NamedTemporaryFile(suffix=".out.pkl", delete=False) as fo:
            out_path = fo.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stdout", delete=False) as fs:
            stdout_path = fs.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stderr", delete=False) as fe:
            stderr_path = fe.name
    except Exception:
        return _fail

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{_root}/src"
        f":{_root}/experiments/02_freegsnke_efit_init/scripts"
        ":" + env.get("PYTHONPATH", "")
    )

    result = None
    try:
        with open(stdout_path, "w") as fso, open(stderr_path, "w") as fse:
            proc = subprocess.Popen(
                [sys.executable, _helper, in_path, out_path],
                stdout=fso, stderr=fse,
                cwd="/tmp", env=env,
                preexec_fn=os.setsid,
            )
        try:
            proc.wait(timeout=_timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return _fail

        try:
            with open(out_path, "rb") as f:
                result = pickle.load(f)
        except Exception:
            return _fail
    except Exception:
        return _fail
    finally:
        for p in [in_path, out_path, stdout_path, stderr_path]:
            try:
                os.unlink(p)
            except Exception:
                pass

    result["idx"]       = idx
    result["shot_id"]   = shot_id
    result["t_task_start"] = t_task_start
    result["t_task_end"]   = time.time()
    result["worker_host"]  = _worker_host
    return result


def parse_args():
    p = argparse.ArgumentParser(description="MorphGSE scaling benchmark")
    p.add_argument("--weights",      required=True,
                   help="Path to MorphGSE checkpoint (.pth) for warm-start inference")
    p.add_argument("--model-seeds",  default="42,0,1",
                   help="Comma-separated model init seeds (default: 42,0,1)")
    p.add_argument("--n-pairs",      type=int, default=100,
                   help="Number of test equilibria to benchmark (default: 100)")
    p.add_argument("--seed",         type=int, default=42,
                   help="RNG seed for sampling test pairs (default: 42)")
    p.add_argument("--mode",         choices=["cold", "warm", "both"], default="both",
                   help="Solve mode: cold-only, warm-only, or both (default: both)")
    p.add_argument("--batch-size",   type=int, default=256,
                   help="GPU inference batch size B* (default: 256 = optimal from benchmark)")
    p.add_argument("--n-workers",    type=int, default=None,
                   help="Target Dask worker count (informational; actual count depends on cluster)")
    p.add_argument("--scheduler",    type=str, default=None,
                   help="Dask scheduler URL (e.g. tcp://192.168.0.103:8786)")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for model inference (default: cuda if available)")
    p.add_argument("--h5",           default=str(_PROJECT_ROOT / "data" / "gs_dataset_v2.h5"))
    p.add_argument("--stats",        default=str(_PROJECT_ROOT / "data" / "gs_dataset_v2.stats.json"))
    p.add_argument("--out",          required=True,
                   help="Output JSON path for benchmark results")
    return p.parse_args()


def _run_gpu_inference(weights_path: str, model_seeds: list[int], indices: list[int],
                       ds: GSDatasetV2, device: str, batch_size: int) -> dict[int, list[np.ndarray]]:
    """Run batched GPU inference for all model seeds.

    Returns dict: model_seed -> list of psi_pred arrays (one per index, same order as indices).
    Also returns per-seed and batch timing info.
    """
    results: dict[int, list[np.ndarray]] = {}
    timing: dict[int, dict] = {}

    for mseed in model_seeds:
        # Derive checkpoint path by replacing seed in the given weights path.
        ckpt_path = Path(weights_path)
        # If caller passed seed42, replace with target seed
        seed_str_in_path = None
        for part in ckpt_path.parts:
            if part.startswith("seed"):
                seed_str_in_path = part
        if seed_str_in_path is not None:
            candidate = Path(str(ckpt_path).replace(seed_str_in_path, f"seed{mseed}"))
            if candidate.exists():
                ckpt_path = candidate

        print(f"\n[inference] model_seed={mseed}  ckpt={ckpt_path.name}")
        t_load_start = time.perf_counter()
        model, stats = _load_model(ckpt_path, device)
        t_load = time.perf_counter() - t_load_start
        print(f"  loaded in {t_load:.2f}s  device={device}")

        psi_preds = []
        t_inf_start = time.perf_counter()
        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            for idx in batch_idx:
                try:
                    pred = _predict_psi(model, stats,
                                        ds.psi_init[idx], ds.pprime_map[idx],
                                        ds.ffprime_map[idx], device)
                except Exception:
                    pred = None
                psi_preds.append(pred)
        t_inf_total = time.perf_counter() - t_inf_start

        n_ok = sum(p is not None for p in psi_preds)
        t_per_sample = t_inf_total / len(indices) if indices else 0
        n_batches = int(np.ceil(len(indices) / batch_size))
        t_per_batch = t_inf_total / n_batches if n_batches else 0
        print(f"  inference: {n_ok}/{len(indices)} ok  "
              f"total={t_inf_total*1000:.0f}ms  "
              f"per_batch={t_per_batch*1000:.1f}ms  "
              f"per_sample={t_per_sample*1000:.2f}ms")

        results[mseed] = psi_preds
        timing[mseed] = {
            "t_load_s": round(t_load, 3),
            "t_inf_total_s": round(t_inf_total, 4),
            "t_inf_per_sample_ms": round(t_per_sample * 1000, 3),
            "t_inf_per_batch_ms": round(t_per_batch * 1000, 2),
            "batch_size": batch_size,
            "n_batches": n_batches,
            "n_ok": n_ok,
        }

    return results, timing


def _run_dask(args, ds, indices, psi_preds_by_seed, mode):
    from dask.distributed import Client, as_completed

    client = Client(args.scheduler, timeout=30)

    # scheduler_info()["workers"] may miss workers that connect via external IPs.
    # Use client.run() to get the authoritative worker count and per-host breakdown.
    def _worker_identity():
        import os
        return {"host": os.uname().nodename, "pid": os.getpid()}

    run_ids = client.run(_worker_identity)
    n_workers_actual = len(run_ids)
    hosts: dict[str, int] = {}
    for info in run_ids.values():
        h = info["host"]
        hosts[h] = hosts.get(h, 0) + 1
    host_summary = "  ".join(f"{h}×{n}" for h, n in sorted(hosts.items()))
    print(f"\nDask: {n_workers_actual} workers (via client.run)  [{host_summary}]")
    print(f"  scheduler_info workers: {len(client.scheduler_info()['workers'])}")

    records_by_seed = {}
    t_scatter_by_seed = {}
    t_solve_by_seed = {}
    t_wall_corrected_by_seed = {}

    for mseed, psi_preds in psi_preds_by_seed.items():
        print(f"\n[dask solve] model_seed={mseed}  n_pairs={len(indices)}")
        t_scatter_by_seed[mseed] = 0.0

        pair_data = [
            (i, idx,
             int(ds.shots[idx]) if hasattr(ds, "shots") else -1,
             ds.get_solver_inputs(idx),
             psi_preds[i] if mode != "cold" else None)
            for i, idx in enumerate(indices)
        ]

        t_solve_start = time.perf_counter()
        futures = [
            client.submit(_dask_solve_pair, d[0], d[1], d[2], d[3], d[4])
            for d in pair_data
        ]

        records = []
        n_done = 0
        for future in as_completed(futures):
            records.append(future.result())
            n_done += 1
            if n_done % 25 == 0 or n_done == len(futures):
                print(f"  completed {n_done}/{len(futures)}")
        t_solve_wall = time.perf_counter() - t_solve_start

        # Corrected wall time from worker timestamps: max(t_task_end) - min(t_task_start)
        ts_all = [r["t_task_start"] for r in records if r.get("t_task_start")]
        te_all = [r["t_task_end"]   for r in records if r.get("t_task_end")]
        t_wall_corrected = (max(te_all) - min(ts_all)) if (ts_all and te_all) else t_solve_wall
        print(f"  wall-clock (outer): {t_solve_wall:.1f}s  corrected (timestamps): {t_wall_corrected:.2f}s")
        t_wall_corrected_by_seed[mseed] = round(t_wall_corrected, 3)

        records.sort(key=lambda r: r["idx"])
        records_by_seed[mseed] = records
        t_solve_by_seed[mseed] = round(t_solve_wall, 3)

    client.close()
    return records_by_seed, t_scatter_by_seed, t_solve_by_seed, t_wall_corrected_by_seed, n_workers_actual, hosts


def _run_sequential(args, ds, indices, psi_preds_by_seed, mode):
    import multiprocessing as mp
    from validate import _dask_solve_pair

    records_by_seed = {}
    t_scatter_by_seed = {}
    t_solve_by_seed = {}

    for mseed, psi_preds in psi_preds_by_seed.items():
        print(f"\n[sequential solve] model_seed={mseed}")
        t_scatter_by_seed[mseed] = 0.0

        t_solve_start = time.perf_counter()
        records = []
        for i, idx in enumerate(indices):
            shot_id = int(ds.shots[idx]) if hasattr(ds, "shots") else -1
            rec = _dask_solve_pair(i, idx, shot_id, ds.get_solver_inputs(idx),
                                   psi_preds[i] if mode != "cold" else None)
            records.append(rec)
            if (i + 1) % 10 == 0 or (i + 1) == len(indices):
                print(f"  {i+1}/{len(indices)}")

        t_solve_wall = time.perf_counter() - t_solve_start
        records.sort(key=lambda r: r["idx"])
        records_by_seed[mseed] = records
        t_solve_by_seed[mseed] = round(t_solve_wall, 3)

    return records_by_seed, t_scatter_by_seed, t_solve_by_seed, 1


def _aggregate(records, mode):
    ratios, t_cold_list, t_warm_list, iters_cold_list, iters_warm_list = [], [], [], [], []
    n_conv_cold = n_conv_warm = n_conv_both = n_psi_ok = 0
    for r in records:
        if r.get("converged_cold"):
            n_conv_cold += 1
        if r.get("converged_warm"):
            n_conv_warm += 1
        if r.get("converged_both"):
            n_conv_both += 1
        psi_ok = str(r.get("psi_consistent", "")).strip() not in ("False", "false", "")
        if r.get("converged_both") and psi_ok:
            n_psi_ok += 1
            ratio = r.get("ratio", float("nan"))
            if ratio == ratio:  # not nan
                ratios.append(ratio)
                t_cold_list.append(r.get("t_cold_s", 0))
                t_warm_list.append(r.get("t_warm_s", 0))
                iters_cold_list.append(r.get("iters_cold", 0))
                iters_warm_list.append(r.get("iters_warm", 0))

    def _ci(arr, n_boot=2000, seed=42):
        if len(arr) < 2:
            return float("nan"), float("nan")
        a = np.array(arr)
        rng = np.random.default_rng(seed)
        boots = np.array([np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)])
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    ratio_mean = float(np.mean(ratios)) if ratios else float("nan")
    ratio_ci = _ci(ratios)
    t_cold_mean = float(np.mean(t_cold_list)) if t_cold_list else float("nan")
    t_warm_mean = float(np.mean(t_warm_list)) if t_warm_list else float("nan")

    return {
        "n_total": len(records),
        "n_conv_cold": n_conv_cold,
        "n_conv_warm": n_conv_warm,
        "n_conv_both": n_conv_both,
        "n_primary": n_psi_ok,
        "mean_ratio": round(ratio_mean, 5),
        "ratio_ci95": [round(ratio_ci[0], 5), round(ratio_ci[1], 5)],
        "mean_t_cold_s": round(t_cold_mean, 4),
        "mean_t_warm_s": round(t_warm_mean, 4),
        "mean_iters_cold": round(float(np.mean(iters_cold_list)), 2) if iters_cold_list else None,
        "mean_iters_warm": round(float(np.mean(iters_warm_list)), 2) if iters_warm_list else None,
        "c1_pass": ratio_mean <= 0.5 if ratio_mean == ratio_mean else None,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_seeds = [int(s) for s in args.model_seeds.split(",")]
    print(f"Benchmark: n_pairs={args.n_pairs}  seed={args.seed}  "
          f"model_seeds={model_seeds}  mode={args.mode}  batch_size={args.batch_size}")

    ds = GSDatasetV2(args.h5, split="test")
    n_test = len(ds)
    n_pairs = min(args.n_pairs, n_test)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_test, size=n_pairs, replace=False).tolist()
    print(f"Selected {n_pairs} indices from test split (n_test={n_test})")

    # GPU inference for all model seeds
    t_wall_inf_start = time.perf_counter()
    psi_preds_by_seed, inference_timing = _run_gpu_inference(
        args.weights, model_seeds, indices, ds, args.device, args.batch_size
    )
    t_wall_inf = time.perf_counter() - t_wall_inf_start

    # Dask or sequential solve
    if args.scheduler:
        records_by_seed, t_scatter_by_seed, t_solve_by_seed, t_wall_corrected_by_seed, n_workers, worker_hosts = _run_dask(
            args, ds, indices, psi_preds_by_seed, args.mode
        )
    else:
        records_by_seed, t_scatter_by_seed, t_solve_by_seed, n_workers = _run_sequential(
            args, ds, indices, psi_preds_by_seed, args.mode
        )
        t_wall_corrected_by_seed = {}
        worker_hosts = {}

    # Aggregate per seed
    agg = {}
    for mseed, records in records_by_seed.items():
        agg[mseed] = _aggregate(records, args.mode)
        print(f"\n[agg] seed={mseed}: "
              f"n_primary={agg[mseed]['n_primary']}  "
              f"mean_ratio={agg[mseed]['mean_ratio']:.4f}  "
              f"CI95={agg[mseed]['ratio_ci95']}  "
              f"C1={'PASS' if agg[mseed].get('c1_pass') else 'FAIL'}")

    # Aggregate across seeds (pooled)
    all_records = [r for recs in records_by_seed.values() for r in recs]
    agg_pooled = _aggregate(all_records, args.mode)
    print(f"\n[pooled] n_primary={agg_pooled['n_primary']}  "
          f"mean_ratio={agg_pooled['mean_ratio']:.4f}  CI95={agg_pooled['ratio_ci95']}  "
          f"C1={'PASS' if agg_pooled.get('c1_pass') else 'FAIL'}")

    # Compute per-seed effective parallelism from timestamps (if available)
    eff_par_by_seed: dict[int, float] = {}
    for mseed, records in records_by_seed.items():
        ts = [r.get("t_task_start") for r in records if r.get("t_task_start")]
        te = [r.get("t_task_end")   for r in records if r.get("t_task_end")]
        if ts and te:
            wall = t_solve_by_seed[mseed]
            total_task = sum(
                r.get("t_task_end", 0) - r.get("t_task_start", 0)
                for r in records
                if r.get("t_task_start") and r.get("t_task_end")
            )
            eff_par_by_seed[mseed] = round(total_task / wall, 2) if wall > 0 else None

    output = {
        "meta": {
            "n_pairs": n_pairs,
            "sampling_seed": args.seed,
            "model_seeds": model_seeds,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "device": args.device,
            "n_workers": n_workers,
            "worker_hosts": worker_hosts,
            "scheduler": args.scheduler,
            "weights": str(args.weights),
        },
        "effective_parallelism": eff_par_by_seed,
        "inference_timing": inference_timing,
        "t_serialize_s": t_scatter_by_seed,
        "t_solve_wall_s": t_solve_by_seed,
        "t_wall_corrected_s": t_wall_corrected_by_seed,
        "t_inference_total_s": round(t_wall_inf, 3),
        "aggregate_per_seed": {str(k): v for k, v in agg.items()},
        "aggregate_pooled": agg_pooled,
        "records_per_seed": {str(k): v for k, v in records_by_seed.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
