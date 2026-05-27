#!/usr/bin/env python
"""Train MorphGSE — L2a: frozen backbone, bilinear decoder.

Usage (from project root):
    # pretrained backbone (frozen) + bilinear decoder:
    uv run python scripts/bilinear_frozen/train.py \\
        --ckpt models/morph-Ti-FM-max_ar1_ep225.pth \\
        --device cuda:0 --seed 42 --max-samples 10000
    # → results/bilinear_frozen/pretrained/N10000/seed42/

    # random backbone (frozen) + bilinear decoder:
    uv run python scripts/bilinear_frozen/train.py \\
        --device cuda:0 --seed 42 --max-samples 10000
    # → results/bilinear_frozen/random/N10000/seed42/
"""
import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from morph_gs import GSDatasetV2, FieldStats, MorphGSE

_DEFAULT_H5    = str(_PROJECT_ROOT / "data" / "gs_dataset_v2.h5")
_DEFAULT_STATS = str(_PROJECT_ROOT / "data" / "gs_dataset_v2.stats.json")


def parse_args():
    p = argparse.ArgumentParser(description="Train MorphGSE L2a: frozen backbone, bilinear decoder")
    p.add_argument("--h5",           type=str,   default=_DEFAULT_H5)
    p.add_argument("--stats",        type=str,   default=_DEFAULT_STATS)
    p.add_argument("--ckpt",         type=str,   default=None,
                   help="MORPH FM checkpoint (.pth). Omit for random init.")
    p.add_argument("--lr-decoder",   type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--max-samples",  type=int,   default=0)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--out-dir",      type=str,   default=None,
                   help="Output dir (default: results/bilinear_frozen/{pretrained|random}/N{N}/seed{seed}/)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--early-stopping-patience",   type=int, default=20)
    p.add_argument("--early-stopping-min-epochs", type=int, default=10)
    return p.parse_args()


def _fix_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _grad_norm(model: MorphGSE) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def main():
    args = parse_args()
    _fix_seeds(args.seed)
    device = torch.device(args.device)

    # ── data ──────────────────────────────────────────────────────────────────
    stats_dict = json.loads(Path(args.stats).read_text())
    stats      = FieldStats.from_dict(stats_dict)
    train_ds   = GSDatasetV2(args.h5, split="train", field_stats=stats)
    val_ds     = GSDatasetV2(args.h5, split="val",   field_stats=stats)

    if args.max_samples > 0 and args.max_samples < len(train_ds):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_ds), size=args.max_samples, replace=False)
        train_ds = Subset(train_ds, idx.tolist())
        print(f"Dataset: {len(train_ds)} train (subset) / {len(val_ds)} val samples")
    else:
        print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    n_train = len(train_ds)
    nw = 0 if args.device == "cpu" else 4
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=False, num_workers=nw,
                          pin_memory=(args.device != "cpu"),
                          generator=torch.Generator().manual_seed(args.seed))
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=(args.device != "cpu"))

    # ── model ─────────────────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    model = MorphGSE(checkpoint_path=ckpt_path,
                     frozen_backbone=True,
                     decoder="bilinear",
                     device=str(device))

    n_dec  = model.n_decoder_params()
    n_bb   = model.n_backbone_params()
    n_tr   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot  = sum(p.numel() for p in model.parameters())
    pretrain = "pretrained" if args.ckpt else "random"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        n_label = f"N{args.max_samples}" if args.max_samples > 0 else "N_full"
        out_dir = (
            _PROJECT_ROOT / "results"
            / "bilinear_frozen"
            / pretrain
            / n_label
            / f"seed{args.seed}"
        )
    print(f"[frozen/{pretrain}/bilinear] decoder={n_dec:,}  backbone={n_bb:,}  "
          f"trainable={n_tr:,} / total={n_tot:,}")

    opt = model.configure_optimizer(
        lr_decoder=args.lr_decoder,
        lr_backbone=0.0,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-7)

    # ── output ────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "morphgse_best.pth"
    log_path  = out_dir / "train_log.json"

    meta = {
        "model":            "MorphGSE",
        "mode":             "frozen",
        "decoder":          "bilinear",
        "frozen_backbone":  True,
        "pretrain":         pretrain,
        "seed":             args.seed,
        "n_decoder_params": n_dec,
        "n_backbone_params": n_bb,
        "n_trainable":      n_tr,
        "cli_args":         vars(args),
        "started_at":       datetime.now(timezone.utc).isoformat(),
        "finished_at":      None,
        "early_stop":       None,
    }

    best_val          = math.inf
    best_epoch        = 0
    epochs_since_best = 0
    epoch_logs        = []
    early_stopped     = False

    def _write_log():
        log_path.write_text(json.dumps({"meta": meta, "epochs": epoch_logs}, indent=2))

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        gnorms     = []
        t0         = time.perf_counter()

        for batch in train_dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            gnorms.append(_grad_norm(model))
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            opt.step()
            train_loss += loss.item() * x.shape[0]

        train_loss /= n_train
        epoch_time  = time.perf_counter() - t0
        scheduler.step()

        lr_dec = opt.param_groups[-1]["lr"]

        model.eval()
        val_loss = 0.0
        val_se_phys = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                bs   = x.shape[0]
                val_loss    += F.mse_loss(pred, y).item() * bs
                p_ph = pred * stats.target_std + stats.target_mean
                y_ph = y    * stats.target_std + stats.target_mean
                val_se_phys += F.mse_loss(p_ph, y_ph).item() * bs
                n_val += bs
        val_loss     /= n_val
        val_rmse_phys = (val_se_phys / n_val) ** 0.5

        entry = dict(epoch=epoch, train_loss=round(train_loss, 6),
                     val_loss=round(val_loss, 6),
                     val_rmse_phys=round(float(val_rmse_phys), 6),
                     lr_decoder=lr_dec,
                     epoch_time_sec=round(epoch_time, 3),
                     grad_norm_mean=round(float(np.mean(gnorms)) if gnorms else 0.0, 6))
        epoch_logs.append(entry)

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"rmse_phys={val_rmse_phys:.4f} Wb  "
              f"gnorm={entry['grad_norm_mean']:.4f}  t={epoch_time:.1f}s")

        if val_loss < best_val:
            best_val          = val_loss
            best_epoch        = epoch
            epochs_since_best = 0
            torch.save({
                "epoch":      epoch,
                "val_loss":   val_loss,
                "state_dict": model.state_dict(),
                "args": {
                    "mode":            "frozen",
                    "decoder":         "bilinear",
                    "frozen_backbone": True,
                    "lr_decoder":      args.lr_decoder,
                },
                "field_means": dict(stats.mean),
                "field_stds":  dict(stats.std),
                "target_mean": stats.target_mean,
                "target_std":  stats.target_std,
            }, best_ckpt)
        else:
            epochs_since_best += 1

        _write_log()

        if (epoch >= args.early_stopping_min_epochs and
                epochs_since_best >= args.early_stopping_patience):
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best val={best_val:.6f} at epoch {best_epoch}.")
            early_stopped = True
            meta["early_stop"] = {
                "triggered":     True,
                "stop_epoch":    epoch,
                "best_epoch":    best_epoch,
                "best_val_loss": round(best_val, 6),
            }
            _write_log()
            break

    if not early_stopped:
        meta["early_stop"] = {
            "triggered":     False,
            "stop_epoch":    args.epochs,
            "best_epoch":    best_epoch,
            "best_val_loss": round(best_val, 6),
        }

    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_log()
    print(f"\nBest val loss: {best_val:.6f} (epoch {best_epoch})  →  {best_ckpt}")


if __name__ == "__main__":
    main()
