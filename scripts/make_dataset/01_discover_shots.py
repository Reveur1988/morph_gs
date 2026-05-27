#!/usr/bin/env python
"""Discover valid MAST shots from FAIR-MAST S3 and produce a stratified shot manifest.

Improvements over v1:
  * Scan a wider candidate pool (--step 1) with random subsampling for unbiased coverage
  * Heuristic disruption-tail removal (dIp/dt threshold) before duration / Ip statistics
  * Compute peak_ip on plasma_mask (not valid_mask); compute duration as plasma_mask.sum() * dt
  * Require minimum number of valid slices per shot (downstream pair generation needs them)
  * Verify EFM ↔ pf_active time-grid overlap (not just availability)
  * Two-axis stratified sampling: Ip bin × MAST era bin
  * Explicit per-bin quotas with logged shortfall (no silent top-up)
  * Train / val / test split is shot-level and asserted disjoint
  * Adversarial flag: lowest Ip decile of the test set
  * Classified rejection reasons + diagnostic log
  * Filesystem cache of scan results (resume / re-stratify without re-scanning S3)
  * Manifest with full reproducibility metadata (seed, filters, git sha, timestamp,
    versions, full valid pool, rejection counts)
  * Deterministic order (sort by shot before sampling) despite parallel scanning

Usage (from project root):
    uv run python experiments/07_dataset_generation/scripts/01_discover_shots_v2.py \\
        --out shot_manifest.json \\
        --n-shots 500 \\
        --workers 32

    # Quick test:
    uv run python .../01_discover_shots_v2.py \\
        --start 28000 --end 30500 --n-candidates 200 --n-shots 50 \\
        --out shot_manifest_test.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_BASE = "https://s3.echo.stfc.ac.uk/mast"
IP_FIELDS = ("plasma_current_c", "plasma_current_x")

# Quality filters
MIN_PEAK_IP_KA = 50.0          # 50 kA minimum peak plasma current
MIN_DURATION_MS = 100.0        # 100 ms minimum plasma duration (post-disruption-trim)
IP_PRESENT_THR_A = 10_000      # |Ip| > 10 kA → plasma present (§С9 of contract)
MIN_VALID_SLICES_DEFAULT = 30  # need >= this many valid slices per shot for downstream

# Disruption heuristic: declare disruption start where |dIp/dt| exceeds
# DISRUPTION_DIPDT_FACTOR × median(|dIp/dt|) during the flat-top, and Ip is decreasing
# in magnitude. Conservative; missed disruptions are flagged for follow-up.
DISRUPTION_DIPDT_FACTOR = 8.0

# Stratification: |Ip| bins in kA (peak per shot)
IP_BINS_KA = [50.0, 200.0, 400.0, 600.0, float("inf")]

# MAST era bins (shot-ID intervals) — coarse proxy for "different campaigns / configurations"
# Will be filled at runtime from --start / --end.
N_ERA_BINS = 3

# Adversarial flag: bottom-decile peak_ip within test set
ADVERSARIAL_FRAC = 0.10


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def l1_url(shot: int) -> str:
    return f"{S3_BASE}/level1/shots/{shot}.zarr"


def l2_url(shot: int) -> str:
    return f"{S3_BASE}/level2/shots/{shot}.zarr"


# ---------------------------------------------------------------------------
# Single-shot scan: returns either a result dict or {"shot": ..., "reject": <reason>}
# ---------------------------------------------------------------------------

def _trim_disruption(time_vals: np.ndarray, ip_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Remove the post-disruption tail using a dIp/dt heuristic.

    Strategy:
      1. Identify the flat-top window (|Ip| ≥ 0.5 × peak_|Ip|).
      2. Inside that window compute median |dIp/dt| as a robust baseline.
      3. Walk forward from the global peak; mark the first index where
         |dIp/dt| > DISRUPTION_DIPDT_FACTOR × baseline AND Ip is decreasing in magnitude.
      4. Discard that index and everything after it.

    Returns (time_trimmed, ip_trimmed, disrupted_flag).
    """
    if len(time_vals) < 5:
        return time_vals, ip_vals, False

    abs_ip = np.abs(ip_vals)
    peak = np.nanmax(abs_ip)
    if peak <= 0 or not np.isfinite(peak):
        return time_vals, ip_vals, False

    flat_top_mask = abs_ip >= 0.5 * peak
    if flat_top_mask.sum() < 3:
        return time_vals, ip_vals, False

    dt = np.diff(time_vals)
    dt[dt == 0] = np.nan
    dip = np.diff(ip_vals) / dt
    abs_dip = np.abs(dip)

    # Baseline: median |dIp/dt| over the flat-top, excluding the very edges.
    flat_top_idx = np.where(flat_top_mask)[0]
    flat_top_dip = abs_dip[flat_top_idx[:-1]]
    flat_top_dip = flat_top_dip[np.isfinite(flat_top_dip)]
    if len(flat_top_dip) < 3:
        return time_vals, ip_vals, False
    baseline = np.median(flat_top_dip)
    if baseline <= 0:
        return time_vals, ip_vals, False

    peak_idx = int(np.nanargmax(abs_ip))
    # Search after the global peak for a fast decrease in |Ip|.
    for i in range(peak_idx, len(dip)):
        if not np.isfinite(abs_dip[i]):
            continue
        decreasing = abs_ip[i + 1] < abs_ip[i]
        if abs_dip[i] > DISRUPTION_DIPDT_FACTOR * baseline and decreasing:
            return time_vals[: i + 1], ip_vals[: i + 1], True

    return time_vals, ip_vals, False


def _scan_shot(shot: int) -> dict:
    """Scan one shot. Always returns a dict; on rejection it carries a 'reject' key."""
    try:
        # Open EFM (level 1)
        try:
            ds_l1 = xr.open_zarr(
                fsspec.get_mapper(l1_url(shot)), group="efm", consolidated=True,
            )
        except Exception as exc:  # noqa: BLE001
            return {"shot": shot, "reject": "no_efm", "detail": type(exc).__name__}

        ip_field = next((f for f in IP_FIELDS if f in ds_l1), None)
        if ip_field is None:
            return {"shot": shot, "reject": "no_ip_field"}

        ip_raw = ds_l1[ip_field].values
        time_raw = ds_l1.time.values
        if ip_raw.size < 5 or time_raw.size != ip_raw.size:
            return {"shot": shot, "reject": "too_few_samples"}

        # Drop NaN entries jointly
        valid_mask = ~np.isnan(ip_raw)
        if valid_mask.sum() < 5:
            return {"shot": shot, "reject": "all_nan_ip"}
        time_v = time_raw[valid_mask]
        ip_v = ip_raw[valid_mask]

        # Disruption trim BEFORE peak/duration statistics
        time_v, ip_v, disrupted = _trim_disruption(time_v, ip_v)

        plasma_mask = np.abs(ip_v) > IP_PRESENT_THR_A
        n_valid_slices = int(plasma_mask.sum())
        if n_valid_slices == 0:
            return {"shot": shot, "reject": "no_plasma_slices", "disrupted": disrupted}

        # Peak Ip / signed peak, computed on plasma_mask
        plasma_ip = ip_v[plasma_mask]
        plasma_t = time_v[plasma_mask]
        peak_abs_a = float(np.max(np.abs(plasma_ip)))
        peak_signed_a = float(plasma_ip[np.argmax(np.abs(plasma_ip))])
        peak_ip_ka = peak_abs_a / 1e3

        if peak_ip_ka < MIN_PEAK_IP_KA:
            return {"shot": shot, "reject": "peak_ip_low", "peak_ip_ka": round(peak_ip_ka, 1)}

        # Duration: count plasma samples × median dt (robust to re-strikes)
        dt_samples = np.diff(plasma_t)
        if dt_samples.size == 0:
            return {"shot": shot, "reject": "single_plasma_sample"}
        median_dt = float(np.median(dt_samples[dt_samples > 0])) if np.any(dt_samples > 0) else 0.0
        duration_ms = float(n_valid_slices * median_dt * 1e3) if median_dt > 0 else 0.0
        if duration_ms < MIN_DURATION_MS:
            return {"shot": shot, "reject": "duration_short", "duration_ms": round(duration_ms, 1)}

        # Check pf_active availability AND time-grid overlap with EFM
        try:
            ds_l2 = xr.open_zarr(
                fsspec.get_mapper(l2_url(shot)), group="pf_active", consolidated=True,
            )
        except Exception as exc:  # noqa: BLE001
            return {"shot": shot, "reject": "no_pf_active", "detail": type(exc).__name__}

        pf_time = ds_l2.time.values if "time" in ds_l2.coords else None
        if pf_time is None or pf_time.size == 0:
            return {"shot": shot, "reject": "pf_active_no_time"}
        # Overlap check: at least one plasma slice falls inside [pf_time.min, pf_time.max]
        if not (pf_time.min() <= plasma_t.max() and pf_time.max() >= plasma_t.min()):
            return {"shot": shot, "reject": "pf_efm_time_disjoint"}

        return {
            "shot": shot,
            "peak_ip_ka": round(peak_ip_ka, 1),
            "peak_ip_signed_ka": round(peak_signed_a / 1e3, 1),
            "duration_ms": round(duration_ms, 1),
            "n_valid_slices": n_valid_slices,
            "median_dt_ms": round(median_dt * 1e3, 3),
            "disrupted_trimmed": bool(disrupted),
        }

    except Exception as exc:  # noqa: BLE001
        return {"shot": shot, "reject": "unhandled_exception", "detail": type(exc).__name__}


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

def _ip_bin(peak_ip_ka: float) -> int:
    for i in range(len(IP_BINS_KA) - 1):
        if IP_BINS_KA[i] <= peak_ip_ka < IP_BINS_KA[i + 1]:
            return i
    return len(IP_BINS_KA) - 2


def _era_bin(shot: int, era_edges: list[int]) -> int:
    for i in range(len(era_edges) - 1):
        if era_edges[i] <= shot < era_edges[i + 1]:
            return i
    return len(era_edges) - 2


# ---------------------------------------------------------------------------
# Sampling with explicit per-bin quotas
# ---------------------------------------------------------------------------

def stratified_sample(
    valid: list[dict],
    n_shots: int,
    era_edges: list[int],
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Sample n_shots from valid, balanced across (ip_bin × era_bin).

    Strategy: compute a uniform target per non-empty cell, sample up to that target
    from each, then redistribute leftover quota to non-empty cells that still have
    candidates. Log any shortfall.
    """
    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for c in valid:
        key = (_ip_bin(c["peak_ip_ka"]), _era_bin(c["shot"], era_edges))
        cells[key].append(c)

    non_empty = list(cells.keys())
    if not non_empty:
        return [], {"shortfall": n_shots, "per_cell": {}}

    # Deterministic order: sort cells; sort shots inside cells by shot-ID; then shuffle.
    selected: list[dict] = []
    per_cell_report: dict[str, dict] = {}
    base_target = max(1, n_shots // len(non_empty))

    leftover = n_shots
    for key in sorted(non_empty):
        shots_in_cell = sorted(cells[key], key=lambda d: d["shot"])
        rng.shuffle(shots_in_cell)
        take = min(base_target, len(shots_in_cell), leftover)
        selected.extend(shots_in_cell[:take])
        per_cell_report[f"ip{key[0]}_era{key[1]}"] = {
            "available": len(shots_in_cell),
            "target": base_target,
            "taken": take,
        }
        cells[key] = shots_in_cell[take:]
        leftover -= take

    # Distribute remaining quota across cells that still have candidates
    while leftover > 0:
        cells_with_left = [k for k in sorted(non_empty) if cells[k]]
        if not cells_with_left:
            break
        for key in cells_with_left:
            if leftover == 0:
                break
            take = min(1, len(cells[key]))
            if take == 0:
                continue
            selected.append(cells[key][0])
            per_cell_report[f"ip{key[0]}_era{key[1]}"]["taken"] += 1
            cells[key] = cells[key][1:]
            leftover -= 1

    return selected, {
        "shortfall": leftover,
        "per_cell": per_cell_report,
        "n_cells_filled": sum(1 for r in per_cell_report.values() if r["taken"] > 0),
        "n_cells_empty_in_pool": (len(IP_BINS_KA) - 1) * (len(era_edges) - 1) - len(non_empty),
    }


# ---------------------------------------------------------------------------
# Train / val / test split — shot-level, stratified, asserted disjoint
# ---------------------------------------------------------------------------

def make_split(
    selected: list[dict],
    era_edges: list[int],
    rng: random.Random,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_per_cell_per_split: int = 1,
) -> dict[str, list[int]]:
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for s in selected:
        key = (_ip_bin(s["peak_ip_ka"]), _era_bin(s["shot"], era_edges))
        cells[key].append(s["shot"])

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for key in sorted(cells.keys()):
        shots = sorted(cells[key])
        rng.shuffle(shots)
        n = len(shots)
        n_test = max(min_per_cell_per_split, round(n * test_frac)) if n >= 3 else (1 if n >= 1 else 0)
        n_val = max(min_per_cell_per_split, round(n * val_frac)) if n - n_test >= 2 else 0
        n_val = min(n_val, max(0, n - n_test - 1))  # leave at least one for train

        test.extend(shots[:n_test])
        val.extend(shots[n_test: n_test + n_val])
        train.extend(shots[n_test + n_val:])

    # Disjointness asserts (defensive — should hold by construction)
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert not (s_train & s_val), "train ∩ val is non-empty"
    assert not (s_train & s_test), "train ∩ test is non-empty"
    assert not (s_val & s_test), "val ∩ test is non-empty"
    assert s_train | s_val | s_test == {s["shot"] for s in selected}, "split coverage mismatch"

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def mark_adversarial(test_shots: list[int], all_selected: list[dict]) -> list[int]:
    """Return the subset of test_shots in the bottom-decile of peak_ip_ka."""
    by_shot = {d["shot"]: d for d in all_selected}
    test_records = [by_shot[s] for s in test_shots if s in by_shot]
    if not test_records:
        return []
    test_records.sort(key=lambda d: d["peak_ip_ka"])
    cutoff_idx = max(1, int(len(test_records) * ADVERSARIAL_FRAC))
    return sorted([d["shot"] for d in test_records[:cutoff_idx]])


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "fsspec": fsspec.__version__,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="shot_manifest.json")
    p.add_argument("--start", type=int, default=10_000)
    p.add_argument("--end", type=int, default=30_473)
    p.add_argument("--n-candidates", type=int, default=8_000,
                   help="how many shot-IDs to scan (random subsample of [start, end))")
    p.add_argument("--n-shots", type=int, default=500,
                   help="target number of selected shots in the manifest")
    p.add_argument("--min-valid-slices", type=int, default=MIN_VALID_SLICES_DEFAULT,
                   help="reject shots with fewer plasma slices than this")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", type=str, default=None,
                   help="optional path to cache scan results (re-stratify without re-scanning S3)")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    t0 = time.time()

    # Build candidate pool: random subsample (without replacement) from [start, end)
    full_range = list(range(args.start, args.end))
    if args.n_candidates >= len(full_range):
        candidates = full_range
    else:
        candidates = rng.sample(full_range, args.n_candidates)
    candidates.sort()  # deterministic dispatch order

    print(f"Candidate pool: {len(candidates)} shot-IDs randomly sampled "
          f"from [{args.start}, {args.end})")

    # Era edges (3 equal-width bins over the full [start, end) range)
    era_edges = list(np.linspace(args.start, args.end, N_ERA_BINS + 1, dtype=int))
    era_edges[-1] = args.end + 1  # right-open boundary inclusive of the upper edge
    print(f"Era bins (shot-ID): {era_edges}")

    # Load cache if present
    scan_results: list[dict] = []
    cached_shots: set[int] = set()
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            scan_results.extend(cached)
            cached_shots = {r["shot"] for r in cached}
            print(f"Loaded {len(cached_shots)} cached scan records from {cache_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to load cache ({type(exc).__name__}); continuing")

    to_scan = [s for s in candidates if s not in cached_shots]
    print(f"Scanning {len(to_scan)} new shot-IDs with {args.workers} threads …")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_scan_shot, s): s for s in to_scan}
        for i, fut in enumerate(as_completed(futures), start=1):
            scan_results.append(fut.result())
            if i % 100 == 0:
                kept = sum(1 for r in scan_results if "reject" not in r)
                print(f"  {i}/{len(to_scan)} scanned, {kept} accepted so far")

    # Save cache
    if cache_path:
        cache_path.write_text(json.dumps(scan_results, indent=2))
        print(f"Cache written → {cache_path}")

    # Split into kept vs rejected; deterministic order
    scan_results.sort(key=lambda r: r["shot"])
    kept = [r for r in scan_results if "reject" not in r]
    rejected = [r for r in scan_results if "reject" in r]

    # Apply --min-valid-slices filter (this is a soft filter we may want to tune)
    kept_before_slices = len(kept)
    kept = [r for r in kept if r["n_valid_slices"] >= args.min_valid_slices]
    n_dropped_for_slices = kept_before_slices - len(kept)
    if n_dropped_for_slices > 0:
        for r in scan_results:
            if "reject" not in r and r["n_valid_slices"] < args.min_valid_slices:
                r_copy = dict(r)
                r_copy["reject"] = "too_few_valid_slices"
                rejected.append(r_copy)

    reject_counts = Counter(r["reject"] for r in rejected)
    print(f"\nScan summary: {len(kept)} accepted, {len(rejected)} rejected")
    for reason, cnt in reject_counts.most_common():
        print(f"  {reason:30s}  {cnt}")

    # Distribution diagnostics on kept pool
    print("\nIp bin distribution (peak |Ip|):")
    ip_dist = Counter(_ip_bin(r["peak_ip_ka"]) for r in kept)
    for b in sorted(ip_dist):
        lo, hi = IP_BINS_KA[b], IP_BINS_KA[b + 1]
        print(f"  [{lo:>5.0f}, {hi if hi < 1e9 else '∞':>5}) kA: {ip_dist[b]}")
    print("Era bin distribution (shot-ID):")
    era_dist = Counter(_era_bin(r["shot"], era_edges) for r in kept)
    for b in sorted(era_dist):
        print(f"  era {b} [{era_edges[b]}, {era_edges[b + 1]}): {era_dist[b]}")

    if len(kept) < args.n_shots:
        print(f"\nWARNING: only {len(kept)} valid shots in pool (requested {args.n_shots})")

    # Stratified sample (ip_bin × era_bin)
    selected, sample_report = stratified_sample(kept, args.n_shots, era_edges, rng)
    print(f"\nSampling: selected {len(selected)} / requested {args.n_shots}, "
          f"shortfall={sample_report['shortfall']}")

    # Split + adversarial flag
    split = make_split(selected, era_edges, rng)
    adversarial = mark_adversarial(split["test"], selected)
    print(f"Split: train={len(split['train'])}  val={len(split['val'])}  "
          f"test={len(split['test'])}  adversarial(in test)={len(adversarial)}")

    # Manifest
    manifest = {
        "meta": {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "versions": _versions(),
            "args": vars(args),
            "filters": {
                "MIN_PEAK_IP_KA": MIN_PEAK_IP_KA,
                "MIN_DURATION_MS": MIN_DURATION_MS,
                "IP_PRESENT_THR_A": IP_PRESENT_THR_A,
                "MIN_VALID_SLICES": args.min_valid_slices,
                "DISRUPTION_DIPDT_FACTOR": DISRUPTION_DIPDT_FACTOR,
                "IP_BINS_KA": IP_BINS_KA,
                "ERA_EDGES": era_edges,
                "ADVERSARIAL_FRAC": ADVERSARIAL_FRAC,
            },
            "duration_s": round(time.time() - t0, 1),
        },
        "counts": {
            "scanned": len(scan_results),
            "kept_pool": len(kept),
            "rejected_total": len(rejected),
            "rejection_reasons": dict(reject_counts),
            "selected": len(selected),
            "sampling": sample_report,
        },
        "selected_shots": sorted(selected, key=lambda d: d["shot"]),
        "split": split,
        "adversarial_test_shots": adversarial,
        "valid_pool": sorted(kept, key=lambda d: d["shot"]),
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(manifest, indent=2, cls=_NumpyEncoder))
    print(f"\nManifest saved → {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    # Optional sibling: rejection log
    rej_path = out_path.with_name(out_path.stem + "_rejected.json")
    rej_path.write_text(json.dumps(sorted(rejected, key=lambda d: d["shot"]), indent=2))
    print(f"Rejection log saved → {rej_path}")


if __name__ == "__main__":
    main()
