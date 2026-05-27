"""Download and process one MAST shot → per-shot NPZ file.

Downloads EFM + pf_active from FAIR-MAST S3, selects N_TIMES stratified time
slices, runs FreeGSNKE cold-solve, saves converged samples with all fields
needed for offline solver evaluation.

CLI::

    morph-gs-process-shot --shot 30420 --out-dir /path/to/shots --split train

Or use programmatically::

    from morph_gs.process_shot import process_shot
    result = process_shot(shot_id=30420, out_dir=Path("shots"), split="train")
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import (
    MAST_FVAC_FALLBACK,
    NX, NY, RMIN, RMAX, ZMIN, ZMAX,
    SOLVER_MAXITS, SOLVER_PICARD_HANDOVER, SOLVER_TOL,
    AMC_SHORT, PF_ACTIVE_COILS,
)
from .fields import build_machine, build_input_fields, make_equilibrium

S3_BASE          = "https://s3.echo.stfc.ac.uk/mast"
IP_PRESENT_THR_A = 10_000
N_TIMES          = 30

FEED_COILS = list(AMC_SHORT.values()) + ["SOL"]
FIL_COILS  = list(PF_ACTIVE_COILS.keys()) + ["SOL"]

IP_FIELDS = ("plasma_current_c", "plasma_current_x")


# ── FAIR-MAST download ────────────────────────────────────────────────────────

def _open_zarr(url: str, group: str):
    import aiohttp
    import xarray as xr
    timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_connect=30, sock_read=90)
    return xr.open_zarr(
        url, group=group, consolidated=True,
        storage_options={
            "client_kwargs": {"timeout": timeout},
            "skip_instance_cache": True,
        },
    ).load()


def _open_zarr_with_retry(url: str, group: str, max_attempts: int = 10):
    import random
    time.sleep(random.uniform(0, 5))  # spread initial load across workers
    for attempt in range(max_attempts):
        try:
            return _open_zarr(url, group)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep((2 ** attempt) + random.uniform(0, 3))
    raise RuntimeError("unreachable")


def _download_shot(shot: int):
    l1 = f"{S3_BASE}/level1/shots/{shot}.zarr"
    l2 = f"{S3_BASE}/level2/shots/{shot}.zarr"
    efm       = _open_zarr_with_retry(l1, "efm")
    pf_active = _open_zarr_with_retry(l2, "pf_active")
    return efm, pf_active


# ── Time-slice selection ──────────────────────────────────────────────────────

def _select_times(efm, n: int = N_TIMES) -> list[float]:
    ip_field = next((f for f in IP_FIELDS if f in efm), None)
    if ip_field is None:
        return []
    ip   = efm[ip_field].values
    time = efm.time.values

    mask = np.abs(ip) > IP_PRESENT_THR_A
    if not mask.any():
        return []
    t_plasma = time[mask]
    t0, t1   = t_plasma[0], t_plasma[-1]
    dur      = t1 - t0
    if dur <= 0:
        return [float(t0)]

    b = [t0, t0 + 0.20 * dur, t0 + 0.80 * dur, t1]
    fracs_per_phase = [1, n - 2, 1]
    selected: list[float] = []
    for i, count in enumerate(fracs_per_phase):
        phase_times = t_plasma[(t_plasma >= b[i]) & (t_plasma <= b[i + 1])]
        if len(phase_times) == 0:
            continue
        indices = np.round(np.linspace(0, len(phase_times) - 1, count)).astype(int)
        for idx in indices:
            t = float(phase_times[idx])
            if t not in selected:
                selected.append(t)
    return sorted(selected)


# ── Input extraction ──────────────────────────────────────────────────────────

def _build_raw_data(efm, pf, t: float) -> dict:
    efm_s = efm.sel(time=t, method="nearest")
    pf_s  = pf.sel(time=t, method="nearest")

    pprime  = efm_s.pprime.values.astype(np.float64)
    ffprime = efm_s.ffprime.values.astype(np.float64)
    psi_norm = (efm_s.psi_norm.values.astype(np.float64)
                if "psi_norm" in efm_s
                else np.linspace(0.0, 1.0, pprime.shape[0]))

    psi_axis  = float(efm_s.psi_axis)
    psi_bndry = float(efm_s.psi_boundary)
    dpsi  = psi_bndry - psi_axis
    paxis = float(efm_s.ppsi_c.values[0])
    p_phys = abs(-np.trapz(pprime, psi_norm) * dpsi)
    p_norm = abs(-np.trapz(pprime, psi_norm))
    if paxis != 0 and abs(p_phys / paxis - 1.0) > abs(p_norm / paxis - 1.0):
        pprime  = pprime  / dpsi
        ffprime = ffprime / dpsi

    channels = pf_s.current_channel.values
    raw_cur  = pf_s.coil_current.values
    feed_currents = {
        AMC_SHORT[str(ch)]: float(c)
        for ch, c in zip(channels, raw_cur) if str(ch) in AMC_SHORT
    }
    feed_currents["SOL"] = float(pf_s.solenoid_current.values)

    filaments: dict[str, tuple] = {}
    for short, prefix in PF_ACTIVE_COILS.items():
        r = pf[f"{prefix}_r"].values
        z = pf[f"{prefix}_z"].values
        if r.ndim > 1:
            r, z = r[0], z[0]
        filaments[short] = (r.astype(np.float32), z.astype(np.float32))
    sol_r = pf["sol_r"].values
    sol_z = pf["sol_z"].values
    if sol_r.ndim > 1:
        sol_r, sol_z = sol_r[0], sol_z[0]
    filaments["SOL"] = (sol_r.astype(np.float32), sol_z.astype(np.float32))

    fcoil_r    = efm.fcoil_r.values.astype(np.float64)
    fcoil_z    = efm.fcoil_z.values.astype(np.float64)
    fcoil_xm   = efm.fcoil_xmult.values.astype(np.float64)
    fcoil_circ = efm.fcoil_circ.values.astype(np.float64)
    fcoil_c    = efm_s.fcoil_c.values.astype(np.float64)

    fvac = MAST_FVAC_FALLBACK
    for var in ("fvac", "bvac_val", "bphi_vacuum"):
        if var in efm_s and not np.isnan(float(efm_s[var])) and float(efm_s[var]) != 0.0:
            fvac = float(efm_s[var])
            break

    raw: dict = dict(
        time=float(efm_s.time),
        psi_norm=psi_norm,
        pprime=pprime,
        ffprime=ffprime,
        Ip=float(efm_s[next(f for f in IP_FIELDS if f in efm_s)].values),
        paxis=paxis,
        fvac=fvac,
        limiter_r=efm_s.limiterr.values.astype(np.float32),
        limiter_z=efm_s.limiterz.values.astype(np.float32),
        efit_fcoil_r=fcoil_r,
        efit_fcoil_z=fcoil_z,
        efit_fcoil_xmult=fcoil_xm,
        efit_fcoil_circ=fcoil_circ,
        efit_fcoil_c=fcoil_c,
    )
    for name, (r, z) in filaments.items():
        raw[f"fil_{name}_r"] = r
        raw[f"fil_{name}_z"] = z
    for k, v in feed_currents.items():
        raw[f"feed_{k}"] = np.float32(v)
    return raw


# ── Solver ────────────────────────────────────────────────────────────────────

def _run_solve(eq, paxis: float, Ip: float, fvac: float) -> tuple[int, bool, np.ndarray]:
    from freegsnke.GSstaticsolver import NKGSsolver
    from freegsnke.jtor_update import ConstrainPaxisIp

    profiles = ConstrainPaxisIp(eq, paxis, Ip, fvac)
    NK = NKGSsolver(eq)
    try:
        NK.forward_solve(
            eq, profiles,
            target_relative_tolerance=SOLVER_TOL,
            max_solving_iterations=SOLVER_MAXITS,
            Picard_handover=SOLVER_PICARD_HANDOVER,
            verbose=False,
        )
        converged = NK.relative_change <= SOLVER_TOL
    except Exception:
        converged = False
    n_iters   = len(np.array(NK.norm_rel_change)) - 1
    psi_final = eq.psi().copy()
    return n_iters, converged, psi_final


# ── Main worker ───────────────────────────────────────────────────────────────

def process_shot(shot_id: int, out_dir: Path, split: str, n_times: int = N_TIMES) -> dict:
    """Process one MAST shot → per-shot NPZ file.

    Downloads from FAIR-MAST S3, runs cold-solve on n_times time slices, writes
    only converged samples. Atomic write (tmp + rename). Returns result dict.

    Returns:
        dict with keys: shot, n_samples, n_skipped, status, elapsed_s.
    """
    out_dir  = Path(out_dir)
    out_path = out_dir / f"shot_{shot_id}.npz"
    if out_path.exists():
        return {"shot": shot_id, "n_samples": -1, "n_skipped": 0, "status": "skipped"}

    t0_total = time.perf_counter()
    try:
        efm, pf_active = _download_shot(shot_id)
    except Exception as e:
        return {"shot": shot_id, "n_samples": 0, "n_skipped": n_times,
                "status": f"download_fail:{e}"}

    times = _select_times(efm, n_times)
    if not times:
        return {"shot": shot_id, "n_samples": 0, "n_skipped": n_times,
                "status": "no_plasma_times"}

    psi_inits, pprime_maps, ffprime_maps, psi_targets = [], [], [], []
    time_vals, iters_vals, t_cold_vals                = [], [], []
    Ip_vals, paxis_vals, fvac_vals                    = [], [], []
    psi_norm_list, pprime_prof_list, ffprime_prof_list = [], [], []
    limiter_r_list, limiter_z_list                    = [], []
    feed_vals: dict[str, list] = {k: [] for k in FEED_COILS}
    fcoil_c_list: list[np.ndarray] = []
    machine_raw: dict | None = None
    n_skipped = 0

    for t in times:
        try:
            raw    = _build_raw_data(efm, pf_active, t)
            fields = build_input_fields(raw, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
            tok    = build_machine(raw)
            eq     = make_equilibrium(tok, RMIN, RMAX, ZMIN, ZMAX, NX, NY)
        except Exception:
            n_skipped += 1
            continue

        t0_solve = time.perf_counter()
        n_iters, converged, psi_cold = _run_solve(
            eq, float(raw["paxis"]), float(raw["Ip"]), float(raw["fvac"]),
        )
        t_cold_ms = (time.perf_counter() - t0_solve) * 1e3

        if not converged:
            n_skipped += 1
            continue

        psi_inits.append(fields["psi_init"])
        pprime_maps.append(fields["pprime_map"])
        ffprime_maps.append(fields["ffprime_map"])
        psi_targets.append(psi_cold.astype(np.float32))
        time_vals.append(float(raw["time"]))
        iters_vals.append(n_iters)
        t_cold_vals.append(t_cold_ms)
        Ip_vals.append(float(raw["Ip"]))
        paxis_vals.append(float(raw["paxis"]))
        fvac_vals.append(float(raw["fvac"]))
        psi_norm_list.append(raw["psi_norm"].astype(np.float32))
        pprime_prof_list.append(raw["pprime"].astype(np.float32))
        ffprime_prof_list.append(raw["ffprime"].astype(np.float32))
        limiter_r_list.append(raw["limiter_r"].astype(np.float32))
        limiter_z_list.append(raw["limiter_z"].astype(np.float32))
        for k in FEED_COILS:
            feed_vals[k].append(float(raw.get(f"feed_{k}", np.nan)))
        fcoil_c_list.append(raw["efit_fcoil_c"].astype(np.float32))
        if machine_raw is None:
            machine_raw = raw

    n_samples = len(psi_inits)
    if n_samples == 0:
        return {"shot": shot_id, "n_samples": 0, "n_skipped": n_skipped,
                "status": "no_converged"}

    data = dict(
        psi_init    = np.stack(psi_inits).astype(np.float32),
        pprime_map  = np.stack(pprime_maps).astype(np.float32),
        ffprime_map = np.stack(ffprime_maps).astype(np.float32),
        psi_target  = np.stack(psi_targets).astype(np.float32),
        times       = np.array(time_vals, dtype=np.float64),
        iters_cold  = np.array(iters_vals, dtype=np.int32),
        shot        = np.int32(shot_id),
        split       = np.bytes_(split.encode()),
        t_cold_ms   = np.array(t_cold_vals, dtype=np.float32),
        Ip          = np.array(Ip_vals, dtype=np.float32),
        paxis       = np.array(paxis_vals, dtype=np.float32),
        fvac        = np.array(fvac_vals, dtype=np.float32),
        psi_norm    = np.stack(psi_norm_list),
        pprime_prof = np.stack(pprime_prof_list),
        ffprime_prof = np.stack(ffprime_prof_list),
        prof_len    = np.array([len(p) for p in psi_norm_list], dtype=np.int32),
        limiter_r   = np.stack(limiter_r_list),
        limiter_z   = np.stack(limiter_z_list),
        limiter_len = np.array(
            [int((~np.isnan(r)).sum()) for r in limiter_r_list], dtype=np.int32
        ),
        **{f"feed_{k}": np.array(feed_vals[k], dtype=np.float32) for k in FEED_COILS},
        fcoil_c     = np.stack(fcoil_c_list).astype(np.float32),
        fcoil_r     = machine_raw["efit_fcoil_r"].astype(np.float32),
        fcoil_z     = machine_raw["efit_fcoil_z"].astype(np.float32),
        fcoil_xmult = machine_raw["efit_fcoil_xmult"].astype(np.float32),
        fcoil_circ  = machine_raw["efit_fcoil_circ"].astype(np.float32),
        **{k: machine_raw[k].astype(np.float32)
           for k in machine_raw if k.startswith("fil_")},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix="._tmp.npz")
    os.close(fd)
    os.chmod(tmp_path, 0o644)
    try:
        np.savez(tmp_path, **data)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    elapsed = time.perf_counter() - t0_total
    return {"shot": shot_id, "n_samples": n_samples, "n_skipped": n_skipped,
            "status": "ok", "elapsed_s": round(elapsed, 1)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Process one MAST shot → NPZ")
    p.add_argument("--shot",    type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--split",   type=str, default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--n-times", type=int, default=N_TIMES,
                   help="number of time slices to sample per shot")
    args = p.parse_args()
    result = process_shot(args.shot, Path(args.out_dir), args.split, args.n_times)
    print(result)


if __name__ == "__main__":
    main()
