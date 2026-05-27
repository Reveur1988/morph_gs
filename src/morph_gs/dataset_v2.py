"""GSDatasetV2 — unified HDF5 dataset for ML training AND solver evaluation.

Modes:
  - ML training:   __getitem__(idx) returns {x, y, shot, time} where x is a
                   normalised, padded UPTF-7 tensor (1, F=3, 1, 1, 72, 72).
  - Solver eval:   get_solver_inputs(idx) returns raw dict for build_machine.

Usage::

    from morph_gs import GSDatasetV2, FieldStats

    train_ds = GSDatasetV2("data/gs_dataset_v2.h5", split="train")
    val_ds   = GSDatasetV2("data/gs_dataset_v2.h5", split="val",
                           field_stats=train_ds.stats)
    # stats are also accessible as train_ds.stats (FieldStats instance)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import AMC_SHORT, PF_ACTIVE_COILS

FIELD_NAMES = ["psi_init", "pprime_map", "ffprime_map"]

_GRID = 65
_PAD  = 72

FEED_COILS = list(AMC_SHORT.values()) + ["SOL"]
FIL_COILS  = list(PF_ACTIVE_COILS.keys()) + ["SOL"]


class FieldStats:
    """Per-field global mean/std for normalisation.

    Always compute from the *training* split, then pass to val/test so
    normalisation is consistent and leak-free.

    Serde via as_dict() / from_dict() for checkpoint storage.

    Note: no __init__ that reads a dataset file. Stats are either computed
    inside GSDatasetV2._compute_stats() or restored from a dict via from_dict().
    """

    mean:        dict[str, float]
    std:         dict[str, float]
    target_mean: float
    target_std:  float

    def normalise_fields(
        self,
        fields: torch.Tensor,
        names: list[str] = FIELD_NAMES,
    ) -> torch.Tensor:
        """Normalise each field channel; returns new tensor."""
        out = []
        for i, name in enumerate(names):
            out.append((fields[i] - self.mean[name]) / self.std[name])
        return torch.stack(out)

    def normalise_target(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.target_mean) / self.target_std

    def denormalise_target(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.target_std + self.target_mean

    def as_dict(self) -> dict:
        """Serialisable snapshot — save in checkpoint for inference."""
        return {
            "field_means": dict(self.mean),
            "field_stds":  dict(self.std),
            "target_mean": self.target_mean,
            "target_std":  self.target_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldStats":
        """Restore from a checkpoint dict (no dataset access needed)."""
        obj = cls.__new__(cls)
        obj.mean        = d["field_means"]
        obj.std         = d["field_stds"]
        obj.target_mean = d["target_mean"]
        obj.target_std  = d["target_std"]
        return obj


class GSDatasetV2(Dataset):
    """Unified HDF5 dataset for ML training AND solver evaluation.

    Modes:
      - ML training: __getitem__(idx) returns {x, y, shot, time} where
                     x is normalised, padded UPTF-7 tensor (1, F=3, 1, 1, 72, 72).
      - Solver eval: get_solver_inputs(idx) returns raw dict for build_machine.

    Args:
        h5_path:     path to gs_dataset_v*.h5
        split:       "train", "val", or "test"
        field_stats: FieldStats for normalisation; if None, computed from this split.
                     Always compute from "train" then pass to val/test.
        pad:         spatial padding target (must be divisible by 8). Default 72.
    """

    RAM_LIMIT_BYTES = 2 * 10 ** 9   # 2 GB

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        field_stats: Optional[FieldStats] = None,
        pad: int = _PAD,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.split   = split
        self.pad     = pad
        self._lazy   = False

        with h5py.File(self.h5_path, "r") as f:
            # Determine split mask
            splits_raw = f["split"][:]
            splits = np.array([
                s if isinstance(s, bytes) else s.encode() for s in splits_raw
            ])
            mask = splits == split.encode()
            self._indices = np.where(mask)[0]
            N = len(self._indices)

            # ML fields — RAM limit check (4 fields × float32 × grid)
            sample_bytes = N * 4 * _GRID * _GRID * 4
            if N > 0 and sample_bytes < self.RAM_LIMIT_BYTES:
                self.psi_init    = f["psi_init"][mask]
                self.pprime_map  = f["pprime_map"][mask]
                self.ffprime_map = f["ffprime_map"][mask]
                self.psi_target  = f["psi_target"][mask]
            else:
                self._lazy = True
                self.psi_init    = None
                self.pprime_map  = None
                self.ffprime_map = None
                self.psi_target  = None

            # Metadata (always in RAM — small)
            self.shots    = f["shots"][self._indices]     # (N,) i32
            self.times    = f["times"][self._indices]     # (N,) f32
            self.shot_idx = f["shot_idx"][self._indices]  # (N,) i32 → machine/ row

            # Diagnostics
            self.iters_cold = f["iters_cold"][self._indices]
            self.t_cold_ms  = f["t_cold_ms"][self._indices]

            # Solver scalars
            self.Ip    = f["Ip"][self._indices]
            self.paxis = f["paxis"][self._indices]
            self.fvac  = f["fvac"][self._indices]

            # Profiles
            self.psi_norm     = f["psi_norm"][self._indices]
            self.pprime_prof  = f["pprime_prof"][self._indices]
            self.ffprime_prof = f["ffprime_prof"][self._indices]
            self.prof_len     = f["prof_len"][self._indices]

            # Limiter
            self.limiter_r   = f["limiter_r"][self._indices]
            self.limiter_z   = f["limiter_z"][self._indices]
            self.limiter_len = f["limiter_len"][self._indices]

            # Feed currents (per-slice)
            self.feed_currents: dict[str, np.ndarray] = {}
            for k in FEED_COILS:
                if k in f["feed_currents"]:
                    self.feed_currents[k] = f["feed_currents"][k][self._indices]

            # fcoil_c per-slice
            self.fcoil_c = f["fcoil_c"][self._indices]

            # Per-shot machine config (indexed by shot_idx, not sample idx)
            S = f["machine/shot_ids"].shape[0]
            self._shot_ids   = f["machine/shot_ids"][:]
            self.fcoil_r     = f["machine/fcoil_r"][:]       # (S, FC_MAX)
            self.fcoil_z     = f["machine/fcoil_z"][:]
            self.fcoil_xmult = f["machine/fcoil_xmult"][:]
            self.fcoil_circ  = f["machine/fcoil_circ"][:]
            self.fc_len = np.array(
                [(~np.isnan(self.fcoil_r[i])).sum() for i in range(S)], dtype=np.int32
            )

            self.filaments_r:   dict[str, np.ndarray] = {}
            self.filaments_z:   dict[str, np.ndarray] = {}
            self.filaments_len: dict[str, np.ndarray] = {}
            fil = f["machine/filaments"]
            for coil in FIL_COILS:
                if f"{coil}_r" in fil:
                    self.filaments_r[coil]   = fil[f"{coil}_r"][:]    # (S, FIL_MAX)
                    self.filaments_z[coil]   = fil[f"{coil}_z"][:]
                    self.filaments_len[coil] = fil[f"{coil}_len"][:]  # (S,)

        self.stats: FieldStats = (
            field_stats if field_stats is not None else self._compute_stats()
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        """Return ML training dict: {x, y, shot, time}.

        x shape: (1, F=3, 1, 1, pad, pad) — normalised, padded UPTF-7 tensor.
        y shape: (65, 65)                  — normalised psi_target.
        """
        if self._lazy:
            return self._getitem_lazy(idx)

        fields_raw = torch.stack([
            torch.from_numpy(self.psi_init[idx]),
            torch.from_numpy(self.pprime_map[idx]),
            torch.from_numpy(self.ffprime_map[idx]),
        ])                                                       # (F, 65, 65)

        fields_norm = self.stats.normalise_fields(fields_raw)   # (F, 65, 65)

        H, W = fields_norm.shape[-2:]
        ph, pw = self.pad - H, self.pad - W
        if ph > 0 or pw > 0:
            fields_norm = torch.nn.functional.pad(fields_norm, (0, pw, 0, ph))

        x = fields_norm[None, :, None, None, :, :]              # (1, F, 1, 1, pad, pad)

        target_raw  = torch.from_numpy(self.psi_target[idx])
        target_norm = self.stats.normalise_target(target_raw)   # (65, 65)

        return {
            "x":    x,
            "y":    target_norm,
            "shot": int(self.shots[idx]),
            "time": float(self.times[idx]),
        }

    def _getitem_lazy(self, idx: int) -> dict:
        h5_idx = int(self._indices[idx])
        with h5py.File(self.h5_path, "r") as f:
            psi_init    = f["psi_init"][h5_idx]
            pprime_map  = f["pprime_map"][h5_idx]
            ffprime_map = f["ffprime_map"][h5_idx]
            psi_target  = f["psi_target"][h5_idx]

        fields_raw = torch.stack([
            torch.from_numpy(psi_init.astype(np.float32)),
            torch.from_numpy(pprime_map.astype(np.float32)),
            torch.from_numpy(ffprime_map.astype(np.float32)),
        ])
        fields_norm = self.stats.normalise_fields(fields_raw)

        H, W = fields_norm.shape[-2:]
        ph, pw = self.pad - H, self.pad - W
        if ph > 0 or pw > 0:
            fields_norm = torch.nn.functional.pad(fields_norm, (0, pw, 0, ph))

        x = fields_norm[None, :, None, None, :, :]

        target_norm = self.stats.normalise_target(
            torch.from_numpy(psi_target.astype(np.float32))
        )
        return {
            "x":    x,
            "y":    target_norm,
            "shot": int(self.shots[idx]),
            "time": float(self.times[idx]),
        }

    def _compute_stats(self) -> FieldStats:
        """Compute FieldStats from this split (reads from HDF5)."""
        with h5py.File(self.h5_path, "r") as f:
            splits_raw = f["split"][:]
            splits = np.array([
                s if isinstance(s, bytes) else s.encode() for s in splits_raw
            ])
            mask = splits == self.split.encode()

            psi_init    = f["psi_init"][mask]
            pprime_map  = f["pprime_map"][mask]
            ffprime_map = f["ffprime_map"][mask]
            psi_target  = f["psi_target"][mask]

        obj = FieldStats.__new__(FieldStats)
        obj.mean = {}
        obj.std  = {}
        for name, arr in zip(FIELD_NAMES, [psi_init, pprime_map, ffprime_map]):
            obj.mean[name] = float(arr.mean())
            obj.std[name]  = float(arr.std()) or 1.0
        obj.target_mean = float(psi_target.mean())
        obj.target_std  = float(psi_target.std()) or 1.0
        return obj

    @classmethod
    def from_stats_json(
        cls,
        h5_path: str | Path,
        stats_json: str | Path,
        split: str = "train",
        pad: int = _PAD,
    ) -> "GSDatasetV2":
        """Convenience constructor: load FieldStats from companion JSON file."""
        stats = FieldStats.from_dict(json.loads(Path(stats_json).read_text()))
        return cls(h5_path, split=split, field_stats=stats, pad=pad)

    def get_solver_inputs(self, idx: int) -> dict[str, Any]:
        """Return raw dict compatible with morph_gs.fields.build_machine.

        Args:
            idx: local dataset index (0 … len(self)-1).

        Returns:
            dict with all fields needed to call build_machine + make_equilibrium
            + cold_solve / warm_solve without any S3 downloads.
        """
        si = int(self.shot_idx[idx])     # index into machine/ arrays
        pl = int(self.prof_len[idx])
        ll = int(self.limiter_len[idx])
        fc = int(self.fc_len[si])

        raw: dict[str, Any] = dict(
            # per-slice scalars
            time  = float(self.times[idx]),
            Ip    = float(self.Ip[idx]),
            paxis = float(self.paxis[idx]),
            fvac  = float(self.fvac[idx]),
            # profiles (strip NaN padding)
            psi_norm = self.psi_norm[idx, :pl].astype(np.float64),
            pprime   = self.pprime_prof[idx, :pl].astype(np.float64),
            ffprime  = self.ffprime_prof[idx, :pl].astype(np.float64),
            # limiter (strip NaN padding)
            limiter_r = self.limiter_r[idx, :ll].astype(np.float64),
            limiter_z = self.limiter_z[idx, :ll].astype(np.float64),
            # EFIT coil model (per-shot, strip padding)
            efit_fcoil_r     = self.fcoil_r[si, :fc].astype(np.float64),
            efit_fcoil_z     = self.fcoil_z[si, :fc].astype(np.float64),
            efit_fcoil_xmult = self.fcoil_xmult[si, :fc].astype(np.float64),
            efit_fcoil_circ  = self.fcoil_circ[si, :fc].astype(np.float64),
            # fcoil_c (per-slice, strip padding to fc length)
            efit_fcoil_c = self.fcoil_c[idx, :fc].astype(np.float64),
        )

        # Filament geometry (per-coil, per-shot, strip padding)
        for coil in FIL_COILS:
            if coil in self.filaments_r:
                cl = int(self.filaments_len[coil][si])
                raw[f"fil_{coil}_r"] = self.filaments_r[coil][si, :cl].astype(np.float64)
                raw[f"fil_{coil}_z"] = self.filaments_z[coil][si, :cl].astype(np.float64)

        # Feed currents (per-slice)
        raw["feed_currents"] = {
            k: float(self.feed_currents[k][idx])
            for k in FEED_COILS if k in self.feed_currents
        }
        # Also expose as flat keys for code that reads feed_{k} directly
        for k, v in raw["feed_currents"].items():
            raw[f"feed_{k}"] = np.float32(v)

        return raw
