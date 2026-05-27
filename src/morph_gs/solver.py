"""FreeGSNKE solver wrappers for warm-start evaluation.

Logic extracted from experiments/13_smoke_warmstart/scripts/pair_runs.py
(the version that produced val_loss=0.085 @ ep.42, median ratio=0.561).
The solve logic is unchanged — only wrapped in clean library functions.

Usage::

    from morph_gs import GSDatasetV2, MorphGS, cold_solve, warm_solve, predict_psi

    ds = GSDatasetV2("data/gs_dataset_v2.h5", split="test", field_stats=stats)
    raw = ds.get_solver_inputs(idx=0)

    iters_cold, conv_cold, t_cold = cold_solve(raw)
    psi_pred = predict_psi(model, ds, idx=0)
    iters_warm, conv_warm, t_warm = warm_solve(raw, psi_init_pred=psi_pred)
"""

from __future__ import annotations

import multiprocessing
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import (
    NX, NY, RMIN, RMAX, ZMIN, ZMAX,
    SOLVER_TOL, SOLVER_MAXITS, SOLVER_PICARD_HANDOVER,
)
from .fields import build_machine, make_equilibrium

if TYPE_CHECKING:
    from .model import MorphGS
    from .dataset_v2 import GSDatasetV2


# ── subprocess worker (must be at module level to be picklable) ──────────────

def _solve_worker(
    result_q,
    eq,
    paxis: float,
    Ip: float,
    fvac: float,
    picard_tol: float,
    max_iters: int,
    picard_handover: float,
) -> None:
    """Run FreeGSNKE solve in a subprocess; put (iters, converged) on result_q."""
    try:
        from freegsnke.GSstaticsolver import NKGSsolver
        from freegsnke.jtor_update import ConstrainPaxisIp

        profiles = ConstrainPaxisIp(eq, paxis, Ip, fvac)
        NK = NKGSsolver(eq)
        NK.forward_solve(
            eq, profiles,
            target_relative_tolerance=picard_tol,
            max_solving_iterations=max_iters,
            Picard_handover=picard_handover,
            verbose=False,
        )
        conv = NK.relative_change <= picard_tol
        n = len(np.array(NK.norm_rel_change)) - 1
        result_q.put((n, conv))
    except Exception:
        result_q.put((0, False))


def _run_solve(
    eq,
    paxis: float,
    Ip: float,
    fvac: float,
    picard_tol: float,
    max_iters: int,
    picard_handover: float,
    timeout_s: float,
) -> tuple[int, bool]:
    """Spawn subprocess, return (iters, converged); kill on timeout."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_solve_worker,
        args=(q, eq, paxis, Ip, fvac, picard_tol, max_iters, picard_handover),
    )
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.kill()
        p.join()
        print(f"  [TIMEOUT after {timeout_s:.0f}s — treating as not converged]")
        return max_iters, False
    return q.get() if not q.empty() else (0, False)


def _seed_psi(eq, psi_pred: np.ndarray) -> None:
    """Inject model-predicted psi into equilibrium as warm start."""
    coil_psi = eq.tokamak.calcPsiFromGreens(eq._pgreen)
    eq._updatePlasmaPsi(psi_pred - coil_psi)


# ── public API ────────────────────────────────────────────────────────────────

def cold_solve(
    raw: dict,
    picard_tol: float = SOLVER_TOL,
    max_iters: int = SOLVER_MAXITS,
    picard_handover: float = SOLVER_PICARD_HANDOVER,
    timeout_s: float | None = 120.0,
) -> tuple[int, bool, float]:
    """Run FreeGSNKE solve from default psi initialisation (cold start).

    Args:
        raw:             solver-input dict from GSDatasetV2.get_solver_inputs().
        picard_tol:      relative convergence tolerance.
        max_iters:       maximum Picard / NK iterations.
        picard_handover: residual at which Picard hands over to NK.
        timeout_s:       hard kill timeout in seconds (None = no timeout).

    Returns:
        (iters_used, converged, wall_clock_seconds)
    """
    tok = build_machine(raw)
    eq  = make_equilibrium(tok, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
    t0  = time.perf_counter()
    iters, conv = _run_solve(
        eq,
        float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]),
        picard_tol, max_iters, picard_handover,
        timeout_s if timeout_s is not None else float("inf"),
    )
    return iters, conv, time.perf_counter() - t0


def warm_solve(
    raw: dict,
    psi_init_pred: np.ndarray,
    picard_tol: float = SOLVER_TOL,
    max_iters: int = SOLVER_MAXITS,
    picard_handover: float = SOLVER_PICARD_HANDOVER,
    timeout_s: float | None = 120.0,
) -> tuple[int, bool, float]:
    """Run FreeGSNKE solve from model-predicted psi (warm start).

    Args:
        raw:             solver-input dict from GSDatasetV2.get_solver_inputs().
        psi_init_pred:   (65, 65) float64 in physical Wb units (already
                         denormalised). Use predict_psi() to obtain this.
        picard_tol:      relative convergence tolerance.
        max_iters:       maximum Picard / NK iterations.
        picard_handover: residual at which Picard hands over to NK.
        timeout_s:       hard kill timeout in seconds (None = no timeout).

    Returns:
        (iters_used, converged, wall_clock_seconds)
    """
    tok = build_machine(raw)
    eq  = make_equilibrium(tok, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
    _seed_psi(eq, psi_init_pred)
    t0  = time.perf_counter()
    iters, conv = _run_solve(
        eq,
        float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]),
        picard_tol, max_iters, picard_handover,
        timeout_s if timeout_s is not None else float("inf"),
    )
    return iters, conv, time.perf_counter() - t0


def predict_psi(
    model: "MorphGS",
    dataset: "GSDatasetV2",
    idx: int,
    device: str = "cuda",
) -> np.ndarray:
    """Run model inference for sample idx; return denormalised psi_pred (65,65) float64.

    Pulls normalised UPTF-7 input from dataset[idx], runs model forward pass,
    denormalises using dataset.stats.  Pure convenience wrapper.

    Args:
        model:   MorphGS instance (eval mode set internally).
        dataset: GSDatasetV2 with stats already computed / set.
        idx:     local dataset index (0 … len(dataset)-1).
        device:  torch device string for inference.

    Returns:
        (65, 65) float64 array in physical Wb units.
    """
    dev = torch.device(device)
    sample = dataset[idx]
    x = sample["x"].unsqueeze(0).to(dev)    # (1, 1, F, 1, 1, 72, 72)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred_norm = model(x)[0].cpu().numpy()   # (65, 65) float32
    if was_training:
        model.train()

    return (
        pred_norm * dataset.stats.target_std + dataset.stats.target_mean
    ).astype(np.float64)
