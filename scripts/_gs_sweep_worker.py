
#!/usr/bin/env python
import pickle, sys, time
from pathlib import Path
import numpy as np

_root = Path("/DATALAKE/mast_gs/mt_experiments")
for _p in [str(_root/"src"), str(_root/"experiments/02_freegsnke_efit_init/scripts")]:
    if _p not in sys.path: sys.path.insert(0, _p)

from config import NX, NY, RMIN, RMAX, ZMIN, ZMAX
from morph_gs.fields import build_machine, make_equilibrium
import freegs4e
from freegsnke.GSstaticsolver import NKGSsolver
from freegsnke.jtor_update import ConstrainPaxisIp

def _run(eq, paxis, Ip, fvac, tol, maxits, handover, max_n_dir, unexpl, max_rel_upd):
    try:
        profiles = ConstrainPaxisIp(eq, paxis, Ip, fvac)
        nk = NKGSsolver(eq)
        nk.forward_solve(eq, profiles,
            target_relative_tolerance=tol,
            max_solving_iterations=maxits,
            Picard_handover=handover,
            max_n_directions=max_n_dir,
            target_relative_unexplained_residual=unexpl,
            max_rel_update_size=max_rel_upd,
            verbose=False)
        conv = nk.relative_change <= tol
        n_iter = len(np.array(nk.norm_rel_change)) - 1
        try: psi_f = eq.psi().copy()
        except: psi_f = None
        return n_iter, conv, psi_f
    except Exception:
        return maxits, False, None

in_path, out_path = sys.argv[1], sys.argv[2]
with open(in_path, "rb") as f:
    data = pickle.load(f)

raw          = data["solver_inputs"]
psi_pred     = data["psi_pred"]
tol          = data.get("tol", 1e-3)
maxits       = data.get("maxits", 100)
handover     = data.get("picard_handover", 0.11)
max_n_dir    = data.get("max_n_directions", 16)
unexpl       = data.get("target_relative_unexplained_residual", 0.2)
max_rel_upd  = data.get("max_rel_update_size", 0.2)

t0 = time.perf_counter()
tok_c = build_machine(raw); eq_c = make_equilibrium(tok_c, RMIN,RMAX,ZMIN,ZMAX,NX,NY)
t0_nk = time.perf_counter()
iters_cold, conv_cold, psi_cf = _run(eq_c, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]),
                                      tol, maxits, handover, max_n_dir, unexpl, max_rel_upd)
t_cold = time.perf_counter() - t0_nk
t_setup = t0_nk - t0

iters_warm, conv_warm, psi_wf = -1, False, None
t_warm = 0.0
if psi_pred is not None:
    tok_w = build_machine(raw); eq_w = make_equilibrium(tok_w, RMIN,RMAX,ZMIN,ZMAX,NX,NY)
    coil_psi = eq_w.tokamak.calcPsiFromGreens(eq_w._pgreen)
    eq_w._updatePlasmaPsi(psi_pred - coil_psi)
    t0_nk = time.perf_counter()
    iters_warm, conv_warm, psi_wf = _run(eq_w, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]),
                                          tol, maxits, handover, max_n_dir, unexpl, max_rel_upd)
    t_warm = time.perf_counter() - t0_nk

psi_ok = None
if psi_cf is not None and psi_wf is not None:
    r = float(np.ptp(psi_cf))
    if r > 0: psi_ok = float(np.max(np.abs(psi_wf - psi_cf))) < 0.01 * r

ratio = (iters_warm / iters_cold) if (iters_cold > 0 and iters_warm >= 0) else float("nan")
result = {"iters_cold": iters_cold, "iters_warm": iters_warm, "ratio": ratio,
          "t_cold_s": round(t_cold,4), "t_warm_s": round(t_warm,4),
          "t_setup_s": round(t_setup,4),
          "converged_cold": conv_cold, "converged_warm": conv_warm,
          "converged_both": bool(conv_cold and conv_warm),
          "psi_consistent": ("" if psi_ok is None else str(psi_ok))}
with open(out_path, "wb") as f:
    pickle.dump(result, f)
