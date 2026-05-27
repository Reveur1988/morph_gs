#!/usr/bin/env python3
"""
Compute values for vkr_v41.md placeholders.
Outputs: Table 2.1, §2.5 cluster analysis, Table 2.4, §2.6 values.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

CONFIGS = ["bilinear_frozen", "bilinear_warmup", "upsamp_frozen", "upsamp_warmup"]
CONFIG_LABELS = {
    "bilinear_frozen":  "A1: `bilinear_frozen`",
    "bilinear_warmup":  "A2: `bilinear_warmup`",
    "upsamp_frozen":    "B1: `upsamp_frozen`",
    "upsamp_warmup":    "B2: `upsamp_warmup`",
}
SEEDS = [0, 1, 42]
N_BOOT = 2000
RNG_SEED = 42


def load_csv(config: str, seeds=SEEDS) -> pd.DataFrame:
    dfs = []
    for seed in seeds:
        path = RESULTS / config / "pretrained" / "N6000" / f"seed{seed}" / "validate_n1000_seed42.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        df = pd.read_csv(path, comment="#")
        df["model_seed"] = seed
        df["config"] = config
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return np.percentile(boots, [2.5, 97.5])


def primary(df: pd.DataFrame) -> pd.DataFrame:
    """Primary subset: converged_both=True AND psi_consistent != False."""
    return df[(df["converged_both"] == True) & (df["psi_consistent"] != False)]


def main():
    # ── TABLE 2.1 ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("TABLE 2.1 — Results for 4 configurations at N=6000, seed=42 eval")
    print("=" * 72)

    config_data: dict[str, pd.DataFrame] = {}
    config_results: dict[str, dict] = {}
    all_dfs = []

    for cfg in CONFIGS:
        df = load_csv(cfg)
        if df.empty:
            continue
        config_data[cfg] = df
        all_dfs.append(df)

        prim = primary(df)
        n_prim = len(prim)
        mean_cold = prim["iters_cold"].mean()
        mean_warm = prim["iters_warm"].mean()
        mean_ratio = prim["ratio"].mean()
        ci_lo, ci_hi = bootstrap_ci_mean(prim["ratio"].values)

        config_results[cfg] = dict(
            n_prim=n_prim,
            mean_cold=mean_cold,
            mean_warm=mean_warm,
            mean_ratio=mean_ratio,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )

    print(f"\n{'Config':<27} {'n_prim':>7} {'cold':>8} {'warm':>8} {'mean_ratio':>11} {'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 80)
    for cfg in CONFIGS:
        r = config_results[cfg]
        print(
            f"{CONFIG_LABELS[cfg]:<27} {r['n_prim']:>7} {r['mean_cold']:>8.2f} "
            f"{r['mean_warm']:>8.2f} {r['mean_ratio']:>11.4f} {r['ci_lo']:>8.4f} {r['ci_hi']:>8.4f}"
        )

    # ── §2.5 CLUSTER ANALYSIS — all 4 configs combined ────────────────────────
    print("\n" + "=" * 72)
    print("§2.5 CLUSTER ANALYSIS — all 4 configs combined (primary: conv_both+psi_ok)")
    print("=" * 72)

    all_df = pd.concat(all_dfs, ignore_index=True)
    N = len(all_df)

    # Cluster 1: cold converged, warm did not
    c1 = all_df[(all_df["converged_cold"] == True) & (all_df["converged_warm"] == False)]
    n1, p1 = len(c1), 100 * len(c1) / N
    shot_ex = c1["shot_id"].value_counts().head(4).index.tolist()

    # Cluster 2: both converged but psi_inconsistent
    c2 = all_df[(all_df["converged_both"] == True) & (all_df["psi_consistent"] == False)]
    n2, p2 = len(c2), 100 * len(c2) / N

    prim_all = primary(all_df)

    # Cluster 3: primary subset, ratio > 1.0 (warm slower)
    c3 = prim_all[prim_all["ratio"] > 1.0]
    n3, p3 = len(c3), 100 * len(c3) / N

    # Cluster 4: primary subset, 0.9 <= ratio <= 1.0 (near-neutral)
    c4 = prim_all[(prim_all["ratio"] >= 0.9) & (prim_all["ratio"] <= 1.0)]
    n4, p4 = len(c4), 100 * len(c4) / N

    # Useful: primary subset, ratio < 0.9
    useful = prim_all[prim_all["ratio"] < 0.9]
    n_useful, p_useful = len(useful), 100 * len(useful) / N

    p_harmful = p1 + p2
    p_neutral  = p3 + p4
    # Note: p_useful + p_neutral + p_harmful may not sum to 100 if some rows are unaccounted
    unaccounted = N - (n1 + n2 + n3 + n4 + n_useful)

    print(f"Total evaluations: {N}  (4 configs × 3 model-seeds × ~1000 pairs)")
    print(f"\nCluster 1 — cold OK, warm failed:           n={n1:>4},  p={p1:>5.1f}%")
    print(f"  Characteristic shot_ids: {shot_ex}")
    print(f"\nCluster 2 — both converged, psi-inconsist.: n={n2:>4},  p={p2:>5.1f}%")
    print(f"\nCluster 3 — primary, ratio > 1.0 (slower): n={n3:>4},  p={p3:>5.1f}%")
    print(f"\nCluster 4 — primary, 0.9 ≤ ratio ≤ 1.0:   n={n4:>4},  p={p4:>5.1f}%")
    print(f"\nUseful    — primary, ratio < 0.9:           n={n_useful:>4},  p={p_useful:>5.1f}%")
    print(f"\n(Unaccounted rows: {unaccounted})")
    print(f"\n--- Three-area summary ---")
    print(f"p_useful  = {p_useful:.1f}%")
    print(f"p_neutral = {p_neutral:.1f}%   (clusters 3+4)")
    print(f"p_harmful = {p_harmful:.1f}%   (clusters 1+2)")
    print(f"Sum check = {p_useful + p_neutral + p_harmful:.1f}% (should be ~100%)")

    # ── TABLE 2.4 — Top-5 worst ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TABLE 2.4 — Top-5 worst by ratio (primary subset, ratio > 1.0)")
    print("=" * 72)
    worst5 = prim_all[prim_all["ratio"] > 1.0].nlargest(5, "ratio")
    if len(worst5) < 5:
        print("  NOTE: fewer than 5 examples with ratio > 1.0, showing all:")
        worst5 = prim_all.nlargest(5, "ratio")
    print(f"\n{'№':>2}  {'shot_id':>8}  {'config':<25}  {'seed':>4}  {'cold':>6}  {'warm':>6}  {'ratio':>6}")
    for i, (_, row) in enumerate(worst5.iterrows(), 1):
        print(
            f"{i:>2}  {int(row['shot_id']):>8}  {row['config']:<25}  {int(row['model_seed']):>4}  "
            f"{int(row['iters_cold']):>6}  {int(row['iters_warm']):>6}  {row['ratio']:>6.3f}"
        )
    print("\nMarkdown rows for Table 2.4:")
    for i, (_, row) in enumerate(worst5.iterrows(), 1):
        label = CONFIG_LABELS.get(row["config"], row["config"])
        print(
            f"| {i} | {int(row['shot_id'])} | {label} | "
            f"{int(row['model_seed'])} | {int(row['iters_cold'])} | "
            f"{int(row['iters_warm'])} | {row['ratio']:.3f} | — |"
        )

    # ── §2.6 — bilinear_warmup summary ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("§2.6 SUMMARY — bilinear_warmup (best by metric)")
    print("=" * 72)
    bw = config_results.get("bilinear_warmup", {})
    if bw:
        m, lo, hi = bw["mean_ratio"], bw["ci_lo"], bw["ci_hi"]
        red = (1 - m) * 100
        print(f"N_test    = 1 000")
        print(f"m         = {m:.4f}  (mean ratio, primary subset, 3 model-seeds)")
        print(f"95% ДИ   = [{lo:.4f}; {hi:.4f}]")
        print(f"reduction ≈ {red:.0f}%")

    # ── FORMATTED MARKDOWN ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("READY-TO-PASTE VALUES FOR vkr_v41.md")
    print("=" * 72)

    print("\n[Table 2.1] Header fix: '6 000' → '1 000' in all 4 rows")
    print("\n[Table 2.1] Full rows:")
    for cfg in CONFIGS:
        r = config_results[cfg]
        label = CONFIG_LABELS[cfg]
        ci_str = f"[{r['ci_lo']:.4f}; {r['ci_hi']:.4f}]"
        mean_ratio_str = f"{r['mean_ratio']:.4f}"
        print(
            f"| {label} | 1 000 | {r['n_prim']} | {r['mean_cold']:.1f} | "
            f"{r['mean_warm']:.1f} | {mean_ratio_str} | {ci_str} | Порог достигнут |"
        )

    print(f"\n[Line 320] bilinear_warmup value: {config_results['bilinear_warmup']['mean_ratio']:.4f}")

    print(f"\n[§2.5 line 354 cluster substitutions]")
    print(f"  p1={p1:.1f}%, n1={n1}, shot_examples={shot_ex[:4]}")
    print(f"  p2={p2:.1f}%")
    print(f"  p3={p3:.1f}%")
    print(f"  p4={p4:.1f}%")

    print(f"\n[§2.5 line 368 three-area summary]")
    print(f"  p_useful={p_useful:.1f}%, p_neutral={p_neutral:.1f}%, p_harmful={p_harmful:.1f}%")

    if bw:
        m, lo, hi = bw["mean_ratio"], bw["ci_lo"], bw["ci_hi"]
        red = (1 - m) * 100
        print(f"\n[§2.6] N_test=1 000, m={m:.4f}, lo={lo:.4f}, hi={hi:.4f}, reduction≈{red:.0f}%")


if __name__ == "__main__":
    main()
