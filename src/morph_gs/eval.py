"""Cold vs warm-start solver pair runs — evaluation of a trained MorphGS.

Loads a trained MorphGS checkpoint, runs cold_solve and warm_solve for N test
samples, saves results to CSV.

CLI::

    morph-gs-eval-pairs \\
        --h5       data/gs_dataset_v2.h5 \\
        --weights  results/morphgs_ft1_lora_best.pth \\
        --n-pairs  10 --seed 42 \\
        --out      results/pair_runs.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

from .dataset_v2 import GSDatasetV2, FieldStats
from .model import MorphGS
from .solver import cold_solve, warm_solve, predict_psi
from .config import SOLVER_TOL, SOLVER_MAXITS, SOLVER_PICARD_HANDOVER


def _load_model_and_stats(ckpt_path: Path, device: str) -> tuple[MorphGS, FieldStats]:
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stats = FieldStats.from_dict({
        "field_means": ckpt["field_means"],
        "field_stds":  ckpt["field_stds"],
        "target_mean": ckpt["target_mean"],
        "target_std":  ckpt["target_std"],
    })
    model = MorphGS.from_checkpoint(ckpt_path, device=device)
    ep    = ckpt.get("epoch", "?")
    vl    = ckpt.get("val_loss", float("nan"))
    print(f"Model loaded: epoch={ep}  val_loss={vl:.4f}")
    return model, stats


def eval_pairs(
    h5:       str | Path,
    weights:  str | Path,
    out:      str | Path,
    n_pairs:  int  = 10,
    seed:     int  = 42,
    device:   str  = "cpu",
) -> list[dict]:
    """Run cold/warm solve pairs for N test samples; write CSV.

    Args:
        h5:      path to HDF5 dataset.
        weights: path to trained MorphGS checkpoint.
        out:     path for output CSV.
        n_pairs: number of test samples to evaluate.
        seed:    RNG seed for sample selection.
        device:  torch device for model inference.

    Returns:
        list of result dicts (one per pair).
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    ckpt_path = Path(weights)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, stats = _load_model_and_stats(ckpt_path, device)
    ds = GSDatasetV2(h5, split="test", field_stats=stats)
    print(f"Test split: {len(ds)} samples")
    print(f"Solver: tol={SOLVER_TOL}, maxits={SOLVER_MAXITS}, "
          f"picard_handover={SOLVER_PICARD_HANDOVER}")

    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_pairs, len(ds)), replace=False).tolist()
    print(f"Evaluating {len(indices)} pairs: {indices}")

    records = []
    for pair_i, idx in enumerate(indices):
        raw     = ds.get_solver_inputs(idx)
        shot_id = int(ds.shots[idx])
        print(f"\n[{pair_i+1}/{len(indices)}] idx={idx}  shot={shot_id}")

        try:
            iters_cold, conv_cold, t_cold = cold_solve(raw)
        except Exception as e:
            print(f"  cold solve FAILED: {e}")
            iters_cold, conv_cold, t_cold = -1, False, 0.0

        psi_pred = None
        try:
            psi_pred = predict_psi(model, ds, idx, device=device)
        except Exception as e:
            print(f"  inference FAILED: {e}")

        iters_warm, conv_warm, t_warm = -1, False, 0.0
        if psi_pred is not None:
            try:
                iters_warm, conv_warm, t_warm = warm_solve(raw, psi_pred)
            except Exception as e:
                print(f"  warm solve FAILED: {e}")

        ratio     = (iters_warm / iters_cold) if (iters_cold > 0 and iters_warm >= 0) else float("nan")
        conv_both = conv_cold and conv_warm

        ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "nan"
        print(f"  cold={iters_cold}({'OK' if conv_cold else 'NO'}) t={t_cold:.1f}s  "
              f"warm={iters_warm}({'OK' if conv_warm else 'NO'}) t={t_warm:.1f}s  "
              f"ratio={ratio_str}")

        records.append({
            "idx":              idx,
            "shot_id":          shot_id,
            "iters_cold":       iters_cold,
            "iters_warm":       iters_warm,
            "ratio":            ratio,
            "t_cold_s":         round(t_cold, 3),
            "t_warm_s":         round(t_warm, 3),
            "converged_cold":   conv_cold,
            "converged_warm":   conv_warm,
            "converged_both":   conv_both,
        })

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid_ratios = [r["ratio"] for r in records if not np.isnan(r["ratio"])]
    median_ratio = float(np.median(valid_ratios)) if valid_ratios else float("nan")

    with open(out_path, "w", newline="") as f:
        f.write(f"# picard_tol={SOLVER_TOL}, max_iters={SOLVER_MAXITS}, "
                f"picard_handover={SOLVER_PICARD_HANDOVER}\n")
        f.write(f"# n_pairs={len(records)}, seed={seed}, checkpoint={weights}\n")
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'='*60}")
    print(f"Results: {len(records)} pairs, "
          f"{sum(r['converged_both'] for r in records)} both converged")
    print(f"Median ratio (warm/cold): {median_ratio:.3f}")
    verdict = "помогает" if median_ratio < 0.95 else "не помогает"
    print(f"Verdict: модель {verdict} (threshold 0.95)")
    print(f"CSV → {out_path}")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Cold/warm solver pair evaluation")
    p.add_argument("--h5",       type=str, required=True)
    p.add_argument("--weights",  type=str, required=True)
    p.add_argument("--out",      type=str, required=True)
    p.add_argument("--n-pairs",  type=int, default=10)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   type=str, default="cpu")
    a = p.parse_args()
    eval_pairs(h5=a.h5, weights=a.weights, out=a.out,
               n_pairs=a.n_pairs, seed=a.seed, device=a.device)


if __name__ == "__main__":
    main()
