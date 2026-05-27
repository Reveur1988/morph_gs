#!/usr/bin/env python
"""Standalone GS solver: reads inputs from a pickle file, writes result to a pickle file.

Usage:
    python _gs_solve_worker.py <input.pkl> <output.pkl>

Input pickle:  {"solver_inputs": dict, "psi_pred": np.ndarray | None}
Output pickle: result dict (iters_cold, iters_warm, ratio, t_cold_s, t_warm_s,
                            converged_cold, converged_warm, converged_both, psi_consistent)

Spawned as a subprocess by _dask_solve_pair so that hanging FreeGSNKE/BLAS calls
can be killed via SIGKILL on the whole process group without affecting the Dask worker.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

_root = Path("/DATALAKE/mast_gs/mt_experiments")
for _p in [
    str(_root / "src"),
    str(_root / "experiments" / "02_freegsnke_efit_init" / "scripts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # noqa: E402
    NX, NY, RMIN, RMAX, ZMIN, ZMAX,
    SOLVER_TOL, SOLVER_MAXITS, SOLVER_PICARD_HANDOVER,
)
from morph_gs.fields import build_machine, make_equilibrium  # noqa: E402
import freegs4e  # noqa: F401, E402
from freegsnke.GSstaticsolver import NKGSsolver  # noqa: E402
from freegsnke.jtor_update import ConstrainPaxisIp  # noqa: E402


def _run(eq, paxis: float, Ip: float, fvac: float):
    try:
        profiles = ConstrainPaxisIp(eq, paxis, Ip, fvac)
        nk = NKGSsolver(eq)
        nk.forward_solve(
            eq, profiles,
            target_relative_tolerance=SOLVER_TOL,
            max_solving_iterations=SOLVER_MAXITS,
            Picard_handover=SOLVER_PICARD_HANDOVER,
            verbose=False,
        )
        conv   = nk.relative_change <= SOLVER_TOL
        n_iter = len(np.array(nk.norm_rel_change)) - 1
        try:
            psi_f = eq.psi().copy()
        except Exception:
            psi_f = None
        return n_iter, conv, psi_f
    except Exception:
        return SOLVER_MAXITS, False, None


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        data = pickle.load(f)

    raw      = data["solver_inputs"]
    psi_pred = data["psi_pred"]

    iters_cold, conv_cold, psi_cold_final = -1, False, None
    t_cold = 0.0
    try:
        tok_c = build_machine(raw)
        eq_c  = make_equilibrium(tok_c, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
        t0 = time.perf_counter()
        iters_cold, conv_cold, psi_cold_final = _run(
            eq_c, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]))
        t_cold = time.perf_counter() - t0
    except Exception:
        pass

    iters_warm, conv_warm, psi_warm_final = -1, False, None
    t_warm = 0.0
    if psi_pred is not None:
        try:
            tok_w = build_machine(raw)
            eq_w  = make_equilibrium(tok_w, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
            coil_psi = eq_w.tokamak.calcPsiFromGreens(eq_w._pgreen)
            eq_w._updatePlasmaPsi(psi_pred - coil_psi)
            t0 = time.perf_counter()
            iters_warm, conv_warm, psi_warm_final = _run(
                eq_w, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]))
            t_warm = time.perf_counter() - t0
        except Exception:
            pass

    psi_ok = None
    if psi_cold_final is not None and psi_warm_final is not None:
        r = float(np.ptp(psi_cold_final))
        if r > 0:
            psi_ok = float(np.max(np.abs(psi_warm_final - psi_cold_final))) < 0.01 * r

    ratio = (iters_warm / iters_cold) if (iters_cold > 0 and iters_warm >= 0) else float("nan")

    result = {
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

    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
