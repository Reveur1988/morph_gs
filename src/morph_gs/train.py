"""Train MorphGS with LoRA fine-tuning.

CLI::

    morph-gs-train \\
        --h5         data/gs_dataset_v2.h5 \\
        --stats      data/gs_dataset_v2.stats.json \\
        --ckpt       models/morph-Ti-FM-max_ar1_ep225.pth \\
        --ft-level   1 \\
        --lora-r-attn 16 --lora-r-mlp 16 --lora-alpha 32 --lora-p 0.05 \\
        --lr-head    1e-3 --lr-lora 5e-4 --weight-decay 1e-2 \\
        --epochs     50 --batch-size 32 \\
        --device     cuda --seed 42 \\
        --out-dir    results/run1 --run-id run1
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import MorphGS
from .dataset_v2 import GSDatasetV2, FieldStats


# ── helpers ───────────────────────────────────────────────────────────────────

def _fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _cuda_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(idx)
    return "cpu"


def _load_fm_weights(model: MorphGS, ckpt_path: Path) -> dict:
    """Load FM backbone weights (not a fine-tuned checkpoint)."""
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    if next(iter(state)).startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    bad_missing    = [k for k in missing    if not k.endswith((".A", ".B"))]
    bad_unexpected = [k for k in unexpected if ".A" not in k and ".B" not in k]
    if bad_missing:
        raise RuntimeError(f"Non-LoRA missing keys: {bad_missing}")
    if bad_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {bad_unexpected}")
    return {
        "n_loaded_keys":          len(state),
        "n_missing":              len(missing),
        "n_unexpected":           len(unexpected),
        "all_missing_are_lora":   all(k.endswith((".A", ".B")) for k in missing),
    }


def _set_lora_lr(opt: torch.optim.AdamW, model: MorphGS, lr_lora: float) -> bool:
    param_id_to_name = {id(p): n for n, p in model.named_parameters()}
    found = False
    for group in opt.param_groups:
        names = [param_id_to_name.get(id(p), "") for p in group["params"]]
        if any(".A" in nm or ".B" in nm for nm in names):
            group["lr"] = lr_lora
            found = True
    return found


def _get_lora_lr(opt: torch.optim.AdamW, model: MorphGS) -> float:
    param_id_to_name = {id(p): n for n, p in model.named_parameters()}
    for group in opt.param_groups:
        names = [param_id_to_name.get(id(p), "") for p in group["params"]]
        if any(".A" in nm or ".B" in nm for nm in names):
            return group["lr"]
    return 0.0


def _grad_norms_split(model: MorphGS):
    lora_norms, other_norms = [], []
    for n, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        norm = p.grad.detach().norm(2).item()
        if ".A" in n or ".B" in n:
            lora_norms.append(norm)
        else:
            other_norms.append(norm)
    return lora_norms, other_norms


# ── main training function ────────────────────────────────────────────────────

def train(
    h5:                        str | Path,
    stats:                     str | Path,
    out_dir:                   str | Path,
    ckpt:                      str | Path | None = None,
    ft_level:                  int   = 1,
    lora_r_attn:               int   = 0,
    lora_r_mlp:                int   = 0,
    lora_alpha:                float | None = None,
    lora_p:                    float = 0.0,
    lr_head:                   float = 1e-3,
    lr_backbone:               float = 1e-4,
    lr_lora:                   float = 5e-4,
    weight_decay:              float = 1e-2,
    epochs:                    int   = 50,
    batch_size:                int   = 32,
    device:                    str   = "cpu",
    variant:                   str   = "Ti",
    seed:                      int   = 42,
    run_id:                    str   = "",
    early_stopping_patience:   int   = 15,
    early_stopping_min_epochs: int   = 30,
) -> Path:
    """Run training loop; return path to best checkpoint."""
    _fix_seeds(seed)
    dev       = torch.device(device)
    started   = datetime.now(timezone.utc).isoformat()
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_dict   = json.loads(Path(stats).read_text())
    field_stats  = FieldStats.from_dict(stats_dict)
    train_ds     = GSDatasetV2(h5, split="train", field_stats=field_stats)
    val_ds       = GSDatasetV2(h5, split="val",   field_stats=field_stats)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val")

    num_workers = 0 if device == "cpu" else 4
    gen         = torch.Generator().manual_seed(seed)
    train_dl    = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             drop_last=False, num_workers=num_workers,
                             pin_memory=(device != "cpu"), generator=gen)
    val_dl      = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device != "cpu"))

    ckpt_path = Path(ckpt) if ckpt else None
    if ckpt_path is not None and not ckpt_path.exists():
        print(f"WARNING: checkpoint not found at {ckpt_path} — random init")
        ckpt_path = None

    model = MorphGS(
        checkpoint_path=None, variant=variant, ft_level=ft_level,
        device="cpu", lora_r_attn=lora_r_attn, lora_r_mlp=lora_r_mlp,
        lora_alpha=lora_alpha, lora_p=lora_p,
    )
    ckpt_info = None
    if ckpt_path is not None:
        ckpt_info = _load_fm_weights(model, ckpt_path)
        print(f"FM weights loaded: {ckpt_info['n_loaded_keys']} keys, "
              f"missing={ckpt_info['n_missing']}, unexpected={ckpt_info['n_unexpected']}")
    model.to(dev)

    opt = model.configure_optimizer(lr_head=lr_head, lr_backbone=lr_backbone,
                                    weight_decay=weight_decay)
    _set_lora_lr(opt, model, lr_lora)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    lora_alpha_actual = lora_alpha if lora_alpha is not None else (
        max(lora_r_attn, lora_r_mlp) or 1
    ) * 2
    lora_suffix = "_lora" if (lora_r_attn > 0 or lora_r_mlp > 0) else ""
    best_ckpt   = out_dir / f"morphgs_ft{ft_level}{lora_suffix}_best.pth"
    log_path    = out_dir / "train_log.json"

    meta: dict = {
        "run_id": run_id, "seed": seed, "ft_level": ft_level,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "n_trainable_params": trainable, "n_total_params": total,
        "lora": {"r_attn": lora_r_attn, "r_mlp": lora_r_mlp,
                 "alpha": lora_alpha_actual, "dropout": lora_p},
        "device": str(dev), "cuda_device_name": _cuda_device_name(dev),
        "git_commit_sha": _git_sha(),
        "started_at": started, "finished_at": None, "early_stop": None,
    }
    if ckpt_info:
        meta["checkpoint_load"] = ckpt_info

    best_val        = math.inf
    best_epoch      = 0
    epochs_since_best = 0
    epoch_logs      = []
    early_stopped   = False

    def _write_log():
        log_path.write_text(json.dumps({"meta": meta, "epochs": epoch_logs}, indent=2))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        lora_norms_all, other_norms_all = [], []
        t_start = time.perf_counter()

        for batch in train_dl:
            x = batch["x"].to(dev)
            y = batch["y"].to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            lns, ons = _grad_norms_split(model)
            lora_norms_all.extend(lns)
            other_norms_all.extend(ons)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            opt.step()
            train_loss += loss.item() * x.shape[0]

        train_loss /= len(train_ds)
        epoch_time  = time.perf_counter() - t_start
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_se_phys = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x"].to(dev)
                y = batch["y"].to(dev)
                pred = model(x)
                bs   = x.shape[0]
                val_loss    += F.mse_loss(pred, y).item() * bs
                pred_phys    = pred * field_stats.target_std + field_stats.target_mean
                y_phys       = y    * field_stats.target_std + field_stats.target_mean
                val_se_phys += F.mse_loss(pred_phys, y_phys).item() * bs
                n_val       += bs
        val_loss      /= n_val
        val_rmse_phys  = (val_se_phys / n_val) ** 0.5

        grad_lora  = float(np.mean(lora_norms_all))  if lora_norms_all  else 0.0
        grad_other = float(np.mean(other_norms_all)) if other_norms_all else 0.0

        epoch_logs.append(dict(
            epoch=epoch, train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
            val_rmse_phys=round(float(val_rmse_phys), 6),
            lr_head=opt.param_groups[-1]["lr"], lr_lora=_get_lora_lr(opt, model),
            epoch_time_sec=round(epoch_time, 3),
            grad_norm_lora_mean=round(grad_lora, 6),
            grad_norm_other_mean=round(grad_other, 6),
        ))
        print(f"Epoch {epoch:4d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"rmse_phys={val_rmse_phys:.4f} Wb  t={epoch_time:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "epoch": epoch, "val_loss": val_loss,
                "state_dict": model.state_dict(),
                "args": {
                    "variant": variant, "ft_level": ft_level,
                    "lora_r_attn": lora_r_attn, "lora_r_mlp": lora_r_mlp,
                    "lora_alpha": lora_alpha_actual, "lora_p": lora_p,
                },
                "field_means": dict(field_stats.mean),
                "field_stds":  dict(field_stats.std),
                "target_mean": field_stats.target_mean,
                "target_std":  field_stats.target_std,
            }, best_ckpt)
        else:
            epochs_since_best += 1

        _write_log()

        if (epoch >= early_stopping_min_epochs and
                epochs_since_best >= early_stopping_patience):
            print(f"\nEarly stopping at epoch {epoch}. Best val={best_val:.6f} @ ep{best_epoch}.")
            early_stopped = True
            meta["early_stop"] = {"triggered": True, "stop_epoch": epoch,
                                  "best_epoch": best_epoch,
                                  "best_val_loss": round(best_val, 6)}
            _write_log()
            break

    if not early_stopped:
        meta["early_stop"] = {"triggered": False, "stop_epoch": epochs,
                              "best_epoch": best_epoch,
                              "best_val_loss": round(best_val, 6)}
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_log()

    print(f"\nBest val_loss: {best_val:.6f} (epoch {best_epoch})  →  {best_ckpt}")
    return best_ckpt


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Train MorphGS with LoRA")
    p.add_argument("--h5",           type=str, required=True)
    p.add_argument("--stats",        type=str, required=True)
    p.add_argument("--out-dir",      type=str, required=True)
    p.add_argument("--ckpt",         type=str, default=None)
    p.add_argument("--ft-level",     type=int, default=1, choices=[0, 1, 2, 4])
    p.add_argument("--lora-r-attn",  type=int,   default=0)
    p.add_argument("--lora-r-mlp",   type=int,   default=0)
    p.add_argument("--lora-alpha",   type=float, default=None)
    p.add_argument("--lora-p",       type=float, default=0.0)
    p.add_argument("--lr-lora",      type=float, default=5e-4)
    p.add_argument("--lr-head",      type=float, default=1e-3)
    p.add_argument("--lr-backbone",  type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--variant",      type=str,   default="Ti")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--run-id",       type=str,   default="")
    p.add_argument("--early-stopping-patience",   type=int, default=15)
    p.add_argument("--early-stopping-min-epochs", type=int, default=30)
    a = p.parse_args()
    train(
        h5=a.h5, stats=a.stats, out_dir=a.out_dir, ckpt=a.ckpt,
        ft_level=a.ft_level, lora_r_attn=a.lora_r_attn, lora_r_mlp=a.lora_r_mlp,
        lora_alpha=a.lora_alpha, lora_p=a.lora_p,
        lr_head=a.lr_head, lr_backbone=a.lr_backbone, lr_lora=a.lr_lora,
        weight_decay=a.weight_decay, epochs=a.epochs, batch_size=a.batch_size,
        device=a.device, variant=a.variant, seed=a.seed, run_id=a.run_id,
        early_stopping_patience=a.early_stopping_patience,
        early_stopping_min_epochs=a.early_stopping_min_epochs,
    )


if __name__ == "__main__":
    main()
