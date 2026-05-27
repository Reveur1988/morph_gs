"""Merge per-shot NPZ files into a single self-contained HDF5 dataset.

Two-pass algorithm:
  Pass 1 — scan all NPZs, determine global max sizes (P_MAX, L_MAX, FC_MAX,
            FIL_MAX per coil).
  Pass 2 — pad variable-length arrays and concatenate.
  Write  — chunked + gzip-compressed HDF5 (same layout as gs_dataset_v2.h5).

CLI::

    morph-gs-build-dataset \\
        --shots-dir /path/to/shots \\
        --out       data/gs_dataset_v2.h5 \\
        --stats-out data/gs_dataset_v2.stats.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .config import (
    AMC_SHORT, PF_ACTIVE_COILS,
    NX, NY, RMIN, RMAX, ZMIN, ZMAX,
    SOLVER_TOL, SOLVER_MAXITS,
)

FIELD_NAMES = ["psi_init", "pprime_map", "ffprime_map"]
FEED_COILS  = list(AMC_SHORT.values()) + ["SOL"]
FIL_COILS   = list(PF_ACTIVE_COILS.keys()) + ["SOL"]
_CHUNK      = 128   # max chunk size; clamped to N at write time


def load_npz(path: Path) -> dict | None:
    try:
        d = np.load(path, allow_pickle=True)
        if d["psi_target"].shape[0] == 0:
            return None
        return dict(d)
    except Exception as e:
        print(f"  WARN: skip {path.name}: {e}")
        return None


def _pad1d(arr: np.ndarray, target: int, fill: float = np.nan) -> np.ndarray:
    if arr.shape[0] >= target:
        return arr[:target]
    out = np.full(target, fill, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def _pad2d_cols(arr: np.ndarray, target_cols: int, fill: float = np.nan) -> np.ndarray:
    rows, cols = arr.shape
    if cols >= target_cols:
        return arr[:, :target_cols]
    out = np.full((rows, target_cols), fill, dtype=arr.dtype)
    out[:, :cols] = arr
    return out


def _compute_stats(psi_init, pprime_map, ffprime_map, psi_target, splits_arr) -> dict:
    mask = splits_arr == b"train"
    stats: dict = {"field_means": {}, "field_stds": {}}
    for name, arr in zip(FIELD_NAMES, [psi_init, pprime_map, ffprime_map]):
        tr = arr[mask]
        stats["field_means"][name] = float(tr.mean())
        stats["field_stds"][name]  = float(tr.std()) or 1.0
    stats["target_mean"] = float(psi_target[mask].mean())
    stats["target_std"]  = float(psi_target[mask].std()) or 1.0
    return stats


def build_dataset(
    shots_dir: Path | str,
    out_path: Path | str,
    stats_out: Path | str | None = None,
    gzip: int = 4,
    dry_run: bool = False,
) -> dict | None:
    """Merge per-shot NPZ files into one HDF5 dataset.

    Args:
        shots_dir: directory containing shot_*.npz files.
        out_path:  path for the output HDF5 file.
        stats_out: optional path for the companion stats JSON.
        gzip:      gzip compression level (0 = no compression).
        dry_run:   if True, scan only — do not write any files.

    Returns:
        stats dict if written, else None.
    """
    shots_dir = Path(shots_dir)
    out_path  = Path(out_path)
    npz_paths = sorted(shots_dir.glob("shot_*.npz"))
    print(f"Found {len(npz_paths)} per-shot NPZ files in {shots_dir}")

    all_data: list[dict] = []
    for path in npz_paths:
        d = load_npz(path)
        if d is not None:
            all_data.append(d)

    if not all_data:
        print("No data loaded. Exiting.")
        return None

    print(f"Loaded {len(all_data)} valid shots")

    P_MAX  = int(max(int(d["prof_len"].max())    for d in all_data))
    L_MAX  = int(max(int(d["limiter_len"].max()) for d in all_data))
    FC_MAX = int(max(d["fcoil_r"].shape[0]       for d in all_data))
    FIL_MAX: dict[str, int] = {}
    for coil in FIL_COILS:
        key = f"fil_{coil}_r"
        FIL_MAX[coil] = int(max(d[key].shape[-1] for d in all_data if key in d))

    print(f"  P_MAX={P_MAX}  L_MAX={L_MAX}  FC_MAX={FC_MAX}")
    print(f"  FIL_MAX: {FIL_MAX}")

    # ── Pass 2: build concatenated arrays ─────────────────────────────────────
    all_psi_init:     list[np.ndarray] = []
    all_pprime:       list[np.ndarray] = []
    all_ffprime:      list[np.ndarray] = []
    all_psi_target:   list[np.ndarray] = []
    all_shots_arr:    list[np.ndarray] = []
    all_times:        list[np.ndarray] = []
    all_iters:        list[np.ndarray] = []
    all_splits:       list[np.ndarray] = []
    all_t_cold:       list[np.ndarray] = []
    all_Ip:           list[np.ndarray] = []
    all_paxis:        list[np.ndarray] = []
    all_fvac:         list[np.ndarray] = []
    all_psi_norm:     list[np.ndarray] = []
    all_pprime_prof:  list[np.ndarray] = []
    all_ffprime_prof: list[np.ndarray] = []
    all_prof_len:     list[np.ndarray] = []
    all_limiter_r:    list[np.ndarray] = []
    all_limiter_z:    list[np.ndarray] = []
    all_limiter_len:  list[np.ndarray] = []
    all_feed: dict[str, list[np.ndarray]] = {k: [] for k in FEED_COILS}
    all_fcoil_c:      list[np.ndarray] = []

    shot_ids_list:    list[int] = []
    shot_fcoil_r:     list[np.ndarray] = []
    shot_fcoil_z:     list[np.ndarray] = []
    shot_fcoil_xmult: list[np.ndarray] = []
    shot_fcoil_circ:  list[np.ndarray] = []
    shot_fil: dict[str, list[np.ndarray]] = {c: [] for c in FIL_COILS}
    sample_shot_idx:  list[np.ndarray] = []

    for shot_idx, d in enumerate(all_data):
        T       = int(d["psi_target"].shape[0])
        shot_id = int(d["shot"])
        shot_ids_list.append(shot_id)

        split_str = d["split"].item()
        if isinstance(split_str, bytes):
            split_str = split_str.decode()

        all_psi_init.append(d["psi_init"])
        all_pprime.append(d["pprime_map"])
        all_ffprime.append(d["ffprime_map"])
        all_psi_target.append(d["psi_target"])
        all_shots_arr.append(np.full(T, shot_id, dtype=np.int32))
        all_times.append(d["times"].astype(np.float32))
        all_iters.append(d["iters_cold"].astype(np.int32))
        all_splits.append(np.array([split_str.encode()] * T))
        all_t_cold.append(d["t_cold_ms"].astype(np.float32))
        all_Ip.append(d["Ip"].astype(np.float32))
        all_paxis.append(d["paxis"].astype(np.float32))
        all_fvac.append(d["fvac"].astype(np.float32))

        all_psi_norm.append(_pad2d_cols(d["psi_norm"].astype(np.float32),    P_MAX))
        all_pprime_prof.append(_pad2d_cols(d["pprime_prof"].astype(np.float32), P_MAX))
        all_ffprime_prof.append(_pad2d_cols(d["ffprime_prof"].astype(np.float32), P_MAX))
        all_prof_len.append(d["prof_len"].astype(np.int32))

        all_limiter_r.append(_pad2d_cols(d["limiter_r"].astype(np.float32), L_MAX))
        all_limiter_z.append(_pad2d_cols(d["limiter_z"].astype(np.float32), L_MAX))
        all_limiter_len.append(d["limiter_len"].astype(np.int32))

        for k in FEED_COILS:
            key = f"feed_{k}"
            if key in d:
                all_feed[k].append(d[key].astype(np.float32))
            else:
                all_feed[k].append(np.full(T, np.nan, dtype=np.float32))

        all_fcoil_c.append(_pad2d_cols(d["fcoil_c"].astype(np.float32), FC_MAX))

        shot_fcoil_r.append(_pad1d(d["fcoil_r"].astype(np.float32),     FC_MAX))
        shot_fcoil_z.append(_pad1d(d["fcoil_z"].astype(np.float32),     FC_MAX))
        shot_fcoil_xmult.append(_pad1d(d["fcoil_xmult"].astype(np.float32), FC_MAX))
        shot_fcoil_circ.append(_pad1d(d["fcoil_circ"].astype(np.float32),   FC_MAX))
        for coil in FIL_COILS:
            key = f"fil_{coil}_r"
            if key in d:
                arr = d[key].astype(np.float32)
                if arr.ndim > 1:
                    arr = arr[0]
                shot_fil[coil].append(_pad1d(arr, FIL_MAX[coil]))
            else:
                shot_fil[coil].append(np.full(FIL_MAX[coil], np.nan, dtype=np.float32))

        sample_shot_idx.append(np.full(T, shot_idx, dtype=np.int32))

    # Concatenate all per-sample arrays
    psi_init     = np.concatenate(all_psi_init,    axis=0)
    pprime_map   = np.concatenate(all_pprime,      axis=0)
    ffprime_map  = np.concatenate(all_ffprime,     axis=0)
    psi_target   = np.concatenate(all_psi_target,  axis=0)
    shots_arr    = np.concatenate(all_shots_arr,   axis=0)
    times_arr    = np.concatenate(all_times,       axis=0)
    iters_arr    = np.concatenate(all_iters,       axis=0)
    splits_arr   = np.concatenate(all_splits,      axis=0)
    t_cold_arr   = np.concatenate(all_t_cold,      axis=0)
    Ip_arr       = np.concatenate(all_Ip,          axis=0)
    paxis_arr    = np.concatenate(all_paxis,       axis=0)
    fvac_arr     = np.concatenate(all_fvac,        axis=0)
    psi_norm_arr    = np.concatenate(all_psi_norm,    axis=0)
    pprime_prof_arr = np.concatenate(all_pprime_prof, axis=0)
    ffprime_prof_arr = np.concatenate(all_ffprime_prof, axis=0)
    prof_len_arr    = np.concatenate(all_prof_len,    axis=0)
    limiter_r_arr   = np.concatenate(all_limiter_r,   axis=0)
    limiter_z_arr   = np.concatenate(all_limiter_z,   axis=0)
    limiter_len_arr = np.concatenate(all_limiter_len, axis=0)
    feed_arr     = {k: np.concatenate(all_feed[k], axis=0) for k in FEED_COILS}
    fcoil_c_arr  = np.concatenate(all_fcoil_c, axis=0)
    shot_idx_arr = np.concatenate(sample_shot_idx, axis=0)

    shot_ids_arr    = np.array(shot_ids_list, dtype=np.int32)
    fcoil_r_arr     = np.stack(shot_fcoil_r)
    fcoil_z_arr     = np.stack(shot_fcoil_z)
    fcoil_xmult_arr = np.stack(shot_fcoil_xmult)
    fcoil_circ_arr  = np.stack(shot_fcoil_circ)
    fil_r_arr: dict[str, np.ndarray] = {c: np.stack(shot_fil[c]) for c in FIL_COILS}
    fil_z_arr: dict[str, np.ndarray] = {}
    for coil in FIL_COILS:
        key = f"fil_{coil}_z"
        tmp = []
        for d in all_data:
            if key in d:
                arr = d[key].astype(np.float32)
                if arr.ndim > 1:
                    arr = arr[0]
                tmp.append(_pad1d(arr, FIL_MAX[coil]))
            else:
                tmp.append(np.full(FIL_MAX[coil], np.nan, dtype=np.float32))
        fil_z_arr[coil] = np.stack(tmp)

    N = int(psi_target.shape[0])
    S = len(shot_ids_list)
    split_counts = {s: int((splits_arr == s).sum()) for s in [b"train", b"val", b"test"]}
    print(f"\n{S} shots, {N} total samples")
    print(f"  train={split_counts.get(b'train',0)}  "
          f"val={split_counts.get(b'val',0)}  "
          f"test={split_counts.get(b'test',0)}")
    nan_count = int(np.isnan(psi_target).sum())
    if nan_count:
        print(f"  WARN: {nan_count} NaN values in psi_target!")

    if dry_run:
        print("\n[dry-run] No files written.")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    R_grid = np.linspace(RMIN, RMAX, NX, dtype=np.float32)
    Z_grid = np.linspace(ZMIN, ZMAX, NY, dtype=np.float32)

    compress  = {"compression": "gzip", "compression_opts": gzip} if gzip else {}
    chunk_n   = min(_CHUNK, N)    # h5py: chunk must not exceed data size
    chunk3d   = (chunk_n, NX, NY)
    chunk2d   = (chunk_n, max(P_MAX, L_MAX, FC_MAX))
    chunk1d   = (min(1024, N),)

    print(f"\nWriting {out_path} …")
    with h5py.File(out_path, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("R",        data=R_grid)
        meta.create_dataset("Z",        data=Z_grid)
        meta.create_dataset("n_shots",  data=np.int32(S))
        meta.create_dataset("n_samples", data=np.int32(N))
        meta.attrs["version"]      = "v2"
        meta.attrs["grid_NX"]      = NX
        meta.attrs["grid_NY"]      = NY
        meta.attrs["solver_tol"]   = float(SOLVER_TOL)
        meta.attrs["solver_maxits"] = int(SOLVER_MAXITS)
        meta.attrs["created_at"]   = datetime.now(timezone.utc).isoformat()

        f.create_dataset("psi_init",    data=psi_init,    chunks=chunk3d, **compress)
        f.create_dataset("pprime_map",  data=pprime_map,  chunks=chunk3d, **compress)
        f.create_dataset("ffprime_map", data=ffprime_map, chunks=chunk3d, **compress)
        f.create_dataset("psi_target",  data=psi_target,  chunks=chunk3d, **compress)

        f.create_dataset("shots",    data=shots_arr,  chunks=chunk1d)
        f.create_dataset("times",    data=times_arr,  chunks=chunk1d)
        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset("split",    data=splits_arr, dtype=dt)
        f.create_dataset("shot_idx", data=shot_idx_arr, chunks=chunk1d)

        f.create_dataset("iters_cold",     data=iters_arr,               chunks=chunk1d)
        f.create_dataset("t_cold_ms",      data=t_cold_arr,              chunks=chunk1d)
        f.create_dataset("converged_cold", data=(iters_arr < SOLVER_MAXITS), chunks=chunk1d)

        f.create_dataset("Ip",    data=Ip_arr,    chunks=chunk1d)
        f.create_dataset("paxis", data=paxis_arr, chunks=chunk1d)
        f.create_dataset("fvac",  data=fvac_arr,  chunks=chunk1d)

        f.create_dataset("psi_norm",     data=psi_norm_arr,    chunks=(chunk_n, P_MAX), **compress)
        f.create_dataset("pprime_prof",  data=pprime_prof_arr, chunks=(chunk_n, P_MAX), **compress)
        f.create_dataset("ffprime_prof", data=ffprime_prof_arr, chunks=(chunk_n, P_MAX), **compress)
        f.create_dataset("prof_len",     data=prof_len_arr,    chunks=chunk1d)

        f.create_dataset("limiter_r",   data=limiter_r_arr,   chunks=(chunk_n, L_MAX), **compress)
        f.create_dataset("limiter_z",   data=limiter_z_arr,   chunks=(chunk_n, L_MAX), **compress)
        f.create_dataset("limiter_len", data=limiter_len_arr, chunks=chunk1d)

        fc_grp = f.create_group("feed_currents")
        for k in FEED_COILS:
            fc_grp.create_dataset(k, data=feed_arr[k], chunks=chunk1d)

        f.create_dataset("fcoil_c", data=fcoil_c_arr, chunks=(chunk_n, FC_MAX), **compress)

        mach = f.create_group("machine")
        mach.create_dataset("shot_ids",    data=shot_ids_arr)
        mach.create_dataset("fcoil_r",     data=fcoil_r_arr,     chunks=(min(64, S), FC_MAX))
        mach.create_dataset("fcoil_z",     data=fcoil_z_arr,     chunks=(min(64, S), FC_MAX))
        mach.create_dataset("fcoil_xmult", data=fcoil_xmult_arr, chunks=(min(64, S), FC_MAX))
        mach.create_dataset("fcoil_circ",  data=fcoil_circ_arr,  chunks=(min(64, S), FC_MAX))
        fil_grp = mach.create_group("filaments")
        for coil in FIL_COILS:
            fm = FIL_MAX[coil]
            fil_grp.create_dataset(f"{coil}_r", data=fil_r_arr[coil], chunks=(min(64, S), fm))
            fil_grp.create_dataset(f"{coil}_z", data=fil_z_arr[coil], chunks=(min(64, S), fm))
            fil_grp.create_dataset(
                f"{coil}_len",
                data=np.array([int((~np.isnan(fil_r_arr[coil][i])).sum())
                               for i in range(S)], dtype=np.int32),
            )

    size_mb = out_path.stat().st_size / 1e6
    print(f"Written: {out_path}  ({size_mb:.1f} MB)")

    stats = _compute_stats(psi_init, pprime_map, ffprime_map, psi_target, splits_arr)
    stats_path = (Path(stats_out) if stats_out
                  else out_path.with_suffix("").with_name(out_path.stem + ".stats.json"))
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats  → {stats_path}")
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Merge per-shot NPZs into HDF5 dataset")
    p.add_argument("--shots-dir", type=str, required=True)
    p.add_argument("--out",       type=str, required=True)
    p.add_argument("--stats-out", type=str, default=None)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--gzip",      type=int, default=4)
    args = p.parse_args()
    build_dataset(
        shots_dir=args.shots_dir,
        out_path=args.out,
        stats_out=args.stats_out,
        gzip=args.gzip,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
