#!/usr/bin/env python
"""Validate MorphGSE checkpoint: median warm/cold NK iteration ratio.

Sequential mode (default):
    uv run python scripts/morph_adapted/validate.py \\
        --weights results/morph_adapted/unfrozen_seed42/morphgse_best.pth \\
        --n-pairs 50 --seed 42 --device cpu

Dask mode (parallel, suitable for n-pairs=1000+):
    uv run python scripts/morph_adapted/validate.py \\
        --weights results/morph_adapted/unfrozen_seed42/morphgse_best.pth \\
        --n-pairs 1000 --seed 42 \\
        --scheduler tcp://192.168.0.103:8786
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
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
from morph_gs.fields import build_machine, make_equilibrium

import freegs4e  # noqa: F401
from freegsnke.GSstaticsolver import NKGSsolver
from freegsnke.jtor_update import ConstrainPaxisIp

_PAD             = 72
_SOLVE_TIMEOUT_S = 120

# Worker-level model cache — populated once per Dask worker process.
_WORKER_CACHE: dict = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   required=True)
    p.add_argument("--h5",        default=str(_PROJECT_ROOT / "data" / "gs_dataset_v2.h5"))
    p.add_argument("--stats",     default=str(_PROJECT_ROOT / "data" / "gs_dataset_v2.stats.json"))
    p.add_argument("--n-pairs",   type=int, default=50)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--n-boot",    type=int, default=1000,
                   help="Bootstrap resamples for CI-95 (default 1000)")
    p.add_argument("--out",       default=None)
    p.add_argument("--scheduler", type=str, default=None,
                   help="Dask scheduler address, e.g. tcp://192.168.0.103:8786. "
                        "When set, pairs are evaluated in parallel on the cluster.")
    return p.parse_args()


# ── FreeGSNKE helpers ─────────────────────────────────────────────────────────

def _solve_worker(result_q, eq, paxis, Ip, fvac):
    try:
        profiles = ConstrainPaxisIp(eq, paxis, Ip, fvac)
        NK = NKGSsolver(eq)
        NK.forward_solve(eq, profiles, target_relative_tolerance=SOLVER_TOL,
                         max_solving_iterations=SOLVER_MAXITS,
                         Picard_handover=SOLVER_PICARD_HANDOVER, verbose=False)
        conv = NK.relative_change <= SOLVER_TOL
        n_iter = len(np.array(NK.norm_rel_change)) - 1
        try:
            psi_final = eq.psi().copy()
        except Exception:
            psi_final = None
        result_q.put((n_iter, conv, psi_final))
    except Exception:
        result_q.put((0, False, None))


def _run_solve(eq, paxis, Ip, fvac):
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_solve_worker, args=(q, eq, paxis, Ip, fvac))
    proc.start()
    proc.join(timeout=_SOLVE_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return SOLVER_MAXITS, False, None
    return q.get() if not q.empty() else (0, False, None)


def _seed_psi(eq, psi_pred):
    coil_psi = eq.tokamak.calcPsiFromGreens(eq._pgreen)
    eq._updatePlasmaPsi(psi_pred - coil_psi)


def _psi_consistent(psi_cold, psi_warm, threshold=0.01):
    if psi_cold is None or psi_warm is None:
        return None
    r = float(np.ptp(psi_cold))
    if r == 0:
        return None
    return float(np.max(np.abs(psi_warm - psi_cold))) < threshold * r


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device: str):
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stats = FieldStats.from_dict({
        "field_means": ckpt["field_means"],
        "field_stds":  ckpt["field_stds"],
        "target_mean": ckpt["target_mean"],
        "target_std":  ckpt["target_std"],
    })
    model = MorphGSE(checkpoint_path=None,
                     frozen_backbone=ckpt.get("args", {}).get("frozen_backbone", False),
                     decoder=ckpt.get("args", {}).get("decoder", "upsampling"),
                     device="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(device))
    model.eval()
    ep = ckpt.get("epoch", "?")
    vl = ckpt.get("val_loss", float("nan"))
    mode = ckpt.get("args", {}).get("mode", "?")
    print(f"MorphGSE loaded: mode={mode}  epoch={ep}  val_loss={vl:.4f}")
    return model, stats


def _predict_psi(model: MorphGSE, stats: FieldStats,
                 psi_init, pprime, ffprime, device: str) -> np.ndarray:
    raw = [psi_init, pprime, ffprime]
    x_norm = np.stack([(f.astype(np.float32) - stats.mean[n]) / stats.std[n]
                       for f, n in zip(raw, FIELD_NAMES)])
    x_pad = np.zeros((3, _PAD, _PAD), dtype=np.float32)
    x_pad[:, :NX, :NY] = x_norm
    x_t = torch.from_numpy(x_pad[np.newaxis, :, np.newaxis, np.newaxis]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(x_t)[0].cpu().numpy()
    return pred_norm * stats.target_std + stats.target_mean


# ── Dask task: fully self-contained, no module-level globals ──────────────────

def _dask_solve_pair(pair_idx, idx, shot_id, solver_inputs, psi_pred):
    """Cold + warm GS solve via subprocess with timeout.

    Spawns _gs_solve_worker.py as a fresh process so that hanging FreeGSNKE/BLAS
    calls can be killed via SIGKILL on the whole process group without affecting
    the Dask worker thread.  All imports are local — cloudpickle sees no deps.
    """
    import os
    import pickle
    import signal
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    _root       = Path("/DATALAKE/mast_gs/mt_experiments")
    _helper     = str(_root / "scripts" / "_gs_solve_worker.py")
    _timeout_s  = 120

    _nan = float("nan")
    _fail = {
        "idx": idx, "shot_id": shot_id,
        "iters_cold": -1, "iters_warm": -1,
        "ratio": _nan,
        "t_cold_s": 0.0, "t_warm_s": 0.0,
        "converged_cold": False, "converged_warm": False,
        "converged_both": False, "psi_consistent": "",
    }

    # Write inputs to a temp pickle; use a separate temp file for the result.
    try:
        with tempfile.NamedTemporaryFile(suffix=".in.pkl",  delete=False) as fi:
            in_path = fi.name
            pickle.dump({"solver_inputs": solver_inputs, "psi_pred": psi_pred}, fi)
        with tempfile.NamedTemporaryFile(suffix=".out.pkl", delete=False) as fo:
            out_path = fo.name
        # Redirect stdout/stderr to temp files — pipes stay open after SIGKILL.
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
                pass  # D-state process; move on, it will be reaped eventually
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

    result["idx"]     = idx
    result["shot_id"] = shot_id
    return result


# ── Per-pair task (sequential path) ──────────────────────────────────────────

def validate_pair(
    pair_idx: int,
    idx: int,
    shot_id: int,
    solver_inputs: dict,
    psi_init: np.ndarray,
    pprime_map: np.ndarray,
    ffprime_map: np.ndarray,
    ckpt_path_str: str,
    device: str,
) -> dict:
    """Evaluate one cold/warm pair. Safe to run on a remote Dask worker."""
    global _WORKER_CACHE

    # Load model once per worker process, cache for subsequent tasks.
    if ckpt_path_str not in _WORKER_CACHE:
        _setup_sys_path()
        _WORKER_CACHE[ckpt_path_str] = _load_model(Path(ckpt_path_str), device)
    model, stats = _WORKER_CACHE[ckpt_path_str]

    raw = solver_inputs

    iters_cold, conv_cold, psi_cold_final = -1, False, None
    t_cold = 0.0
    try:
        tok_c = build_machine(raw)
        eq_c  = make_equilibrium(tok_c, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
        t0 = time.perf_counter()
        iters_cold, conv_cold, psi_cold_final = _run_solve(
            eq_c, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]))
        t_cold = time.perf_counter() - t0
    except Exception:
        pass

    psi_pred = None
    try:
        psi_pred = _predict_psi(model, stats, psi_init, pprime_map, ffprime_map, device)
    except Exception:
        pass

    iters_warm, conv_warm, psi_warm_final = -1, False, None
    t_warm = 0.0
    if psi_pred is not None:
        try:
            tok_w = build_machine(raw)
            eq_w  = make_equilibrium(tok_w, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
            _seed_psi(eq_w, psi_pred)
            t0 = time.perf_counter()
            iters_warm, conv_warm, psi_warm_final = _run_solve(
                eq_w, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]))
            t_warm = time.perf_counter() - t0
        except Exception:
            pass

    ratio  = (iters_warm / iters_cold) if (iters_cold > 0 and iters_warm >= 0) else float("nan")
    psi_ok = _psi_consistent(psi_cold_final, psi_warm_final)

    return {
        "idx":            idx,
        "shot_id":        shot_id,
        "iters_cold":     iters_cold,
        "iters_warm":     iters_warm,
        "ratio":          ratio,
        "t_cold_s":       round(t_cold, 3),
        "t_warm_s":       round(t_warm, 3),
        "converged_cold": conv_cold,
        "converged_warm": conv_warm,
        "converged_both": (conv_cold and conv_warm),
        "psi_consistent": ("" if psi_ok is None else str(psi_ok)),
    }


def _setup_sys_path():
    """Ensure imports work on remote Dask workers."""
    for p in [str(_EXP02_SCRIPTS), str(_PROJECT_ROOT / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _bootstrap_median_ci(ratios, n_boot, seed):
    if len(ratios) < 2:
        return float("nan"), float("nan")
    arr  = np.array(ratios)
    rng  = np.random.default_rng(seed)
    boots = np.array([
        np.median(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.weights)
    ds      = GSDatasetV2(args.h5, split="test")
    n_test  = len(ds)
    n_pairs = min(args.n_pairs, n_test) if args.n_pairs > 0 else n_test
    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(n_test, size=n_pairs, replace=False).tolist()
    print(f"Running {n_pairs} pairs (seed={args.seed})")

    ckpt_path_str = str(ckpt_path.resolve())

    if args.scheduler:
        records = _run_dask(args, ds, indices, ckpt_path_str)
    else:
        records = _run_sequential(args, ds, indices, ckpt_path_str)

    # ── aggregate ─────────────────────────────────────────────────────────────
    metric_rows = [r for r in records
                   if r["converged_both"] and r["psi_consistent"] != "False"
                   and not (isinstance(r["ratio"], float) and np.isnan(r["ratio"]))]
    ratios = [r["ratio"] for r in metric_rows]
    median = float(np.median(ratios)) if ratios else float("nan")
    n_conv = sum(1 for r in records if r["converged_both"])

    ci_lo, ci_hi = _bootstrap_median_ci(ratios, args.n_boot, args.seed)

    print(f"\n{'='*50}")
    print(f"n_pairs={n_pairs}  n_converged={n_conv}  n_metric={len(ratios)}")
    print(f"Median ratio: {median:.3f}  CI-95: [{ci_lo:.3f}, {ci_hi:.3f}]  (n_boot={args.n_boot})")
    verdict = "PASS" if (not np.isnan(median) and median <= 0.5) else "FAIL"
    print(f"Verdict: {verdict}  (threshold ≤ 0.5, contract C1)")

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_csv = (Path(args.out) if args.out
               else ckpt_path.parent / f"validate_n{args.n_pairs}_seed{args.seed}.csv")
    fieldnames = ["idx", "shot_id", "iters_cold", "iters_warm", "ratio",
                  "t_cold_s", "t_warm_s",
                  "converged_cold", "converged_warm", "converged_both", "psi_consistent"]
    with open(out_csv, "w", newline="") as f:
        f.write(f"# checkpoint={ckpt_path}\n")
        f.write(f"# n_pairs={n_pairs}, seed={args.seed}\n")
        f.write(f"# median_ratio={median:.4f}  "
                f"bootstrap_ci95=[{ci_lo:.4f}, {ci_hi:.4f}]  "
                f"n_boot={args.n_boot}\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"CSV → {out_csv}")


def _run_sequential(args, ds, indices, ckpt_path_str):
    model, stats = _load_model(Path(args.weights), args.device)
    records = []
    for pair_i, idx in enumerate(indices):
        raw     = ds.get_solver_inputs(idx)
        shot_id = int(ds.shots[idx]) if hasattr(ds, "shots") else -1
        print(f"\n[{pair_i+1}/{len(indices)}] idx={idx}  shot={shot_id}")
        rec = validate_pair(
            pair_i, idx, shot_id, raw,
            ds.psi_init[idx], ds.pprime_map[idx], ds.ffprime_map[idx],
            ckpt_path_str, args.device,
        )
        ratio_str = f"{rec['ratio']:.3f}" if not np.isnan(rec["ratio"]) else "nan"
        print(f"  cold={rec['iters_cold']}({'✓' if rec['converged_cold'] else '✗'})  "
              f"warm={rec['iters_warm']}({'✓' if rec['converged_warm'] else '✗'})  "
              f"ratio={ratio_str}")
        records.append(rec)
    return records


def _run_dask(args, ds, indices, ckpt_path_str):
    from dask.distributed import Client, as_completed

    print(f"Connecting to Dask scheduler: {args.scheduler}")
    client = Client(args.scheduler)
    workers = client.scheduler_info()["workers"]
    print(f"Workers: {len(workers)}  "
          f"threads: {sum(v['nthreads'] for v in workers.values())}")

    # torch is on the client; workers run solver-only tasks via _dask_solve_pair.
    print("Running model inference on client...")
    model, stats = _load_model(Path(args.weights), args.device)
    psi_preds = []
    for idx in indices:
        try:
            pred = _predict_psi(model, stats,
                                ds.psi_init[idx], ds.pprime_map[idx],
                                ds.ffprime_map[idx], args.device)
        except Exception:
            pred = None
        psi_preds.append(pred)
    n_ok = sum(p is not None for p in psi_preds)
    print(f"  predicted {n_ok}/{len(indices)} ψ values")

    pair_data = [
        (i, idx,
         int(ds.shots[idx]) if hasattr(ds, "shots") else -1,
         ds.get_solver_inputs(idx),
         psi_preds[i])
        for i, idx in enumerate(indices)
    ]
    futures = [
        client.submit(_dask_solve_pair, d[0], d[1], d[2], d[3], d[4])
        for d in pair_data
    ]

    records = []
    n_done = 0
    for future in as_completed(futures):
        records.append(future.result())
        n_done += 1
        if n_done % 100 == 0 or n_done == len(futures):
            print(f"  completed {n_done}/{len(futures)}")

    client.close()
    records.sort(key=lambda r: r["idx"])
    return records


if __name__ == "__main__":
    main()
