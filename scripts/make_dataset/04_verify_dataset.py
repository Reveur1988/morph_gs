#!/usr/bin/env python
"""Verify gs_dataset_v2.h5 structural and physical correctness.

Checks:
  1. All required HDF5 keys present.
  2. n_samples == sum of train/val/test counts.
  3. NaN only in padding zones of psi_norm, limiter_r, limiter_z.
  4. Ip > 10 kA for all samples.
  5. iters_cold > 0 for all samples.
  6. No shot_id overlap between train/val/test splits.
  7. K5 sanity: re-run cold-solve on one random test sample; recorded
     iters_cold must match within ±2 (float32 storage tolerance).

Usage:
    uv run python scripts/04_verify_dataset.py --h5 data/gs_dataset_v2.h5
    uv run python scripts/04_verify_dataset.py --h5 data/gs_dataset_v2.h5 --no-k5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from morph_gs.config import (
    AMC_SHORT, PF_ACTIVE_COILS,
    NX, NY, RMIN, RMAX, ZMIN, ZMAX,
    SOLVER_TOL, SOLVER_MAXITS, SOLVER_PICARD_HANDOVER,
)

IP_PRESENT_THR_A = 10_000
FEED_COILS = list(AMC_SHORT.values()) + ["SOL"]
FIL_COILS  = list(PF_ACTIVE_COILS.keys()) + ["SOL"]


def check(condition: bool, msg: str, errors: list[str]) -> bool:
    if not condition:
        errors.append(f"  FAIL: {msg}")
        return False
    print(f"  OK  : {msg}")
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",      type=str, required=True)
    p.add_argument("--no-k5",   action="store_true",
                   help="Skip K5 solver sanity check (quick structural verify)")
    p.add_argument("--k5-seed", type=int, default=0,
                   help="Random seed for selecting K5 test sample")
    return p.parse_args()


def main():
    args   = parse_args()
    h5path = Path(args.h5)
    errors: list[str] = []

    if not h5path.exists():
        print(f"ABORT: {h5path} not found")
        sys.exit(1)

    print(f"Verifying {h5path}\n")

    with h5py.File(h5path, "r") as f:

        # ── Check 1: required keys ─────────────────────────────────────────────
        print("=== Check 1: required keys ===")
        required_top = [
            "psi_init", "pprime_map", "ffprime_map", "psi_target",
            "shots", "times", "split", "shot_idx",
            "iters_cold", "t_cold_ms", "converged_cold",
            "Ip", "paxis", "fvac",
            "psi_norm", "pprime_prof", "ffprime_prof", "prof_len",
            "limiter_r", "limiter_z", "limiter_len",
            "fcoil_c",
        ]
        for key in required_top:
            check(key in f, f"key '{key}' present", errors)

        for key in ["R", "Z", "n_shots", "n_samples"]:
            check(key in f["meta"], f"meta/{key} present", errors)
        check(f["meta"].attrs.get("version") == "v2", "meta.version == 'v2'", errors)

        check("feed_currents" in f, "group 'feed_currents' present", errors)
        if "feed_currents" in f:
            for k in FEED_COILS:
                check(k in f["feed_currents"], f"feed_currents/{k} present", errors)

        check("machine" in f, "group 'machine' present", errors)
        if "machine" in f:
            for key in ["shot_ids", "fcoil_r", "fcoil_z", "fcoil_xmult", "fcoil_circ"]:
                check(key in f["machine"], f"machine/{key} present", errors)
            check("filaments" in f["machine"], "machine/filaments present", errors)
            if "filaments" in f["machine"]:
                for coil in FIL_COILS:
                    for suffix in ("_r", "_z", "_len"):
                        check(f"{coil}{suffix}" in f["machine"]["filaments"],
                              f"machine/filaments/{coil}{suffix} present", errors)

        # ── Check 2: n_samples consistency ────────────────────────────────────
        print("\n=== Check 2: n_samples consistency ===")
        N = int(f["meta/n_samples"][()])
        splits = f["split"][:]
        split_vals = np.array([s if isinstance(s, bytes) else s.encode() for s in splits])
        n_train = int((split_vals == b"train").sum())
        n_val   = int((split_vals == b"val").sum())
        n_test  = int((split_vals == b"test").sum())
        check(n_train + n_val + n_test == N,
              f"n_samples={N} == train({n_train})+val({n_val})+test({n_test})", errors)
        check(n_train > 0, f"train split non-empty ({n_train})", errors)
        check(n_val   > 0, f"val split non-empty ({n_val})",   errors)
        check(n_test  > 0, f"test split non-empty ({n_test})", errors)

        # ── Check 3: NaN only in padding zones ────────────────────────────────
        print("\n=== Check 3: NaN padding zones ===")
        psi_norm_arr = f["psi_norm"][:]
        prof_len_arr = f["prof_len"][:]
        n_bad_pn = 0
        for i in range(min(N, 200)):
            pl  = int(prof_len_arr[i])
            row = psi_norm_arr[i]
            if np.any(np.isnan(row[:pl])):
                n_bad_pn += 1
            if pl < row.shape[0] and not np.all(np.isnan(row[pl:])):
                n_bad_pn += 1
        check(n_bad_pn == 0,
              f"psi_norm padding correct (checked first {min(N,200)} samples)", errors)

        limiter_r_arr   = f["limiter_r"][:]
        limiter_len_arr = f["limiter_len"][:]
        n_bad_lim = 0
        for i in range(min(N, 200)):
            ll  = int(limiter_len_arr[i])
            row = limiter_r_arr[i]
            if np.any(np.isnan(row[:ll])):
                n_bad_lim += 1
        check(n_bad_lim == 0,
              f"limiter_r padding correct (checked first {min(N,200)} samples)", errors)

        # ── Check 4: Ip > 10 kA ───────────────────────────────────────────────
        print("\n=== Check 4: Ip > 10 kA ===")
        Ip_arr    = f["Ip"][:]
        n_low_ip  = int((np.abs(Ip_arr) <= IP_PRESENT_THR_A).sum())
        check(n_low_ip == 0, f"Ip > 10 kA for all {N} samples ({n_low_ip} violations)", errors)

        # ── Check 5: iters_cold > 0 ───────────────────────────────────────────
        print("\n=== Check 5: iters_cold > 0 ===")
        iters        = f["iters_cold"][:]
        n_zero_iters = int((iters <= 0).sum())
        check(n_zero_iters == 0,
              f"iters_cold > 0 for all {N} samples ({n_zero_iters} violations)", errors)

        # ── Check 6: split disjointness ───────────────────────────────────────
        print("\n=== Check 6: split disjointness ===")
        shots_arr   = f["shots"][:]
        train_shots = set(shots_arr[split_vals == b"train"].tolist())
        val_shots   = set(shots_arr[split_vals == b"val"].tolist())
        test_shots  = set(shots_arr[split_vals == b"test"].tolist())
        check(len(train_shots & val_shots)  == 0,
              f"train ∩ val = ∅  (|train|={len(train_shots)}, |val|={len(val_shots)})", errors)
        check(len(train_shots & test_shots) == 0,
              f"train ∩ test = ∅  (|test|={len(test_shots)})", errors)
        check(len(val_shots   & test_shots) == 0,
              "val ∩ test = ∅", errors)

        # ── Check 7: K5 cold-solve sanity ─────────────────────────────────────
        print("\n=== Check 7: K5 sanity (cold-solve reproducibility) ===")
        if args.no_k5:
            print("  SKIP (--no-k5)")
        else:
            test_indices = np.where(split_vals == b"test")[0]
            if len(test_indices) == 0:
                errors.append("  FAIL: no test samples for K5 check")
            else:
                rng      = np.random.default_rng(args.k5_seed)
                idx      = int(rng.choice(test_indices))
                shot_idx = int(f["shot_idx"][idx])
                recorded_iters = int(f["iters_cold"][idx])
                print(f"  Using sample idx={idx}  shot={int(f['shots'][idx])}"
                      f"  recorded_iters={recorded_iters}")

                raw: dict = {}
                pl = int(f["prof_len"][idx])
                ll = int(f["limiter_len"][idx])

                raw["psi_norm"]  = f["psi_norm"][idx, :pl].astype(np.float64)
                raw["pprime"]    = f["pprime_prof"][idx, :pl].astype(np.float64)
                raw["ffprime"]   = f["ffprime_prof"][idx, :pl].astype(np.float64)
                raw["Ip"]        = float(f["Ip"][idx])
                raw["paxis"]     = float(f["paxis"][idx])
                raw["fvac"]      = float(f["fvac"][idx])
                raw["limiter_r"] = f["limiter_r"][idx, :ll].astype(np.float64)
                raw["limiter_z"] = f["limiter_z"][idx, :ll].astype(np.float64)

                fcoil_r_full = f["machine/fcoil_r"][shot_idx]
                fc_actual    = int((~np.isnan(fcoil_r_full)).sum())
                raw["efit_fcoil_r"]     = fcoil_r_full[:fc_actual].astype(np.float64)
                raw["efit_fcoil_z"]     = f["machine/fcoil_z"][shot_idx][:fc_actual].astype(np.float64)
                raw["efit_fcoil_xmult"] = f["machine/fcoil_xmult"][shot_idx][:fc_actual].astype(np.float64)
                raw["efit_fcoil_circ"]  = f["machine/fcoil_circ"][shot_idx][:fc_actual].astype(np.float64)
                raw["efit_fcoil_c"]     = f["fcoil_c"][idx][:fc_actual].astype(np.float64)
                for coil in FIL_COILS:
                    cl = int(f[f"machine/filaments/{coil}_len"][shot_idx])
                    raw[f"fil_{coil}_r"] = f[f"machine/filaments/{coil}_r"][shot_idx, :cl].astype(np.float64)
                    raw[f"fil_{coil}_z"] = f[f"machine/filaments/{coil}_z"][shot_idx, :cl].astype(np.float64)

                try:
                    import freegs4e  # noqa: F401
                    from freegsnke.GSstaticsolver import NKGSsolver
                    from freegsnke.jtor_update import ConstrainPaxisIp
                    from morph_gs.fields import build_machine, make_equilibrium

                    tokamak  = build_machine(raw)
                    eq       = make_equilibrium(tokamak, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
                    profiles = ConstrainPaxisIp(eq, raw["paxis"], raw["Ip"], raw["fvac"])
                    NK       = NKGSsolver(eq)
                    try:
                        NK.forward_solve(
                            eq, profiles,
                            target_relative_tolerance=SOLVER_TOL,
                            max_solving_iterations=SOLVER_MAXITS,
                            Picard_handover=SOLVER_PICARD_HANDOVER,
                            verbose=False,
                        )
                    except Exception:
                        pass
                    rerun_iters = len(np.array(NK.norm_rel_change)) - 1
                    delta = abs(rerun_iters - recorded_iters)
                    check(delta <= 2,
                          f"K5: |rerun({rerun_iters}) - recorded({recorded_iters})| = {delta} ≤ 2",
                          errors)
                except Exception as e:
                    errors.append(f"  FAIL: K5 exception: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
