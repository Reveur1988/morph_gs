"""MorphGS: MORPH Foundation Model adapted for Grad-Shafranov warm-start.

Architecture summary
--------------------
Input  → UPTF-7 tensor (B, T=1, F=3, C=1, D=1, 72, 72)
           ↓
       ViT3DRegression backbone  (MORPH-Ti or MORPH-S)
           ↓
       output (B, F=3, 1, 1, 72, 72)  →  crop [:, :, :, :, :65, :65]
           ↓
       (B, 3, 65, 65)  →  Conv2d(3→1, kernel=1) + scale + bias
           ↓
       ψ_pred (B, 65, 65)   [in normalised space during training]

Key design choices
------------------
T=1:        Static GS equation → no temporal autoregression needed.
            Time-axial attention with seq_len=1 is a trivial no-op.
max_fields=3: FM checkpoint was trained with max_fields=3; using the same value
            avoids shape mismatch in the decoder linear layer. SimpleDecoder
            truncates to actual F via x[..., :out_ch].
pad 65→72:  MORPH requires H%8==0 (patch_size=8). 72=9×8.
field_combine: lightweight 1×1 convolution maps the 3 backbone output channels
            to a single ψ prediction. Acts as a learned weighted average.

Fine-tuning levels (ft_level)
------------------------------
0   Only Conv2d head + scale/bias train.  Backbone frozen.      ~10 params
1   + LayerNorms + positional encodings.                         ~50 K params
2   + conv encoder + cross-attention layers.                     ~1 M params
4   All backbone parameters.                                     ~7 M params
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_morph_path_env = os.environ.get("MORPH_PATH")
_MORPH_ROOT = Path(_morph_path_env) if _morph_path_env else Path(__file__).resolve().parents[2] / "MORPH"
if not _MORPH_ROOT.exists():
    raise ImportError(
        f"MORPH repository not found at {_MORPH_ROOT}. "
        f"Clone https://github.com/lanl/MORPH there or set MORPH_PATH env var."
    )
if str(_MORPH_ROOT) not in sys.path:
    sys.path.insert(0, str(_MORPH_ROOT))

from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters

_MORPH_CONFIGS = {
    "Ti": dict(conv_filter=8, dim=256, heads=4, depth=4, mlp_dim=1024),
    "S":  dict(conv_filter=8, dim=512, heads=8, depth=4, mlp_dim=2048),
}

_N_FIELDS    = 3   # psi_init, pprime_map, ffprime_map
_GRID        = 65
_PAD         = 72
_H_PATCHES   = _W_PATCHES = _PAD // 8  # = 9

_MORPH_CFG_E = dict(conv_filter=8, dim=256, depth=4, heads=4, heads_xa=32,
                    mlp_dim=1024, max_components=3, max_ar=1,
                    max_patches=4096, max_fields=3, dropout=0.0, emb_dropout=0.0)


class MorphGS(nn.Module):
    """MORPH Foundation Model receiving real 2D Grad-Shafranov physics fields.

    Args:
        checkpoint_path: Path to FM .pth file containing "model_state_dict".
                         Pass None for random initialisation (testing only).
        variant:         "Ti" (~7 M params) or "S" (~30 M params).
        ft_level:        Backbone fine-tuning level; see module docstring.
        device:          Torch device string.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        variant: str = "Ti",
        ft_level: int = 0,
        device: str = "cpu",
        lora_r_attn: int = 0,
        lora_r_mlp: int = 0,
        lora_alpha: Optional[float] = None,
        lora_p: float = 0.0,
    ):
        super().__init__()
        cfg = _MORPH_CONFIGS[variant]
        self.ft_level    = ft_level
        self.lora_r_attn = lora_r_attn
        self.lora_r_mlp  = lora_r_mlp
        self.lora_alpha  = lora_alpha if lora_alpha is not None else (max(lora_r_attn, lora_r_mlp) or 1) * 2
        self.lora_p      = lora_p

        self.backbone = ViT3DRegression(
            patch_size=8,
            dim=cfg["dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            heads_xa=32,
            mlp_dim=cfg["mlp_dim"],
            max_components=3,
            conv_filter=cfg["conv_filter"],
            max_ar=1,
            max_patches=4096,
            max_fields=3,   # must match FM checkpoint; decoder truncates to actual F
            dropout=0.0,
            emb_dropout=0.0,
            lora_r_attn=lora_r_attn,
            lora_r_mlp=lora_r_mlp,
            lora_alpha=self.lora_alpha,
            lora_p=lora_p,
        )

        if checkpoint_path is not None:
            ckpt  = torch.load(str(checkpoint_path), map_location="cpu",
                               weights_only=True)
            state = ckpt["model_state_dict"]
            if next(iter(state)).startswith("module."):
                state = {k[len("module."):]: v for k, v in state.items()}
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            _check_keys(missing, unexpected)

        # Learnable head: combine F=3 output channels → single ψ
        self.field_combine = nn.Conv2d(_N_FIELDS, 1, kernel_size=1, bias=True)
        self.log_scale     = nn.Parameter(torch.zeros(1))
        self.bias_param    = nn.Parameter(torch.zeros(1))

        self.to(torch.device(device))

    # ── optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizer(
        self,
        lr_head:      float = 1e-3,
        lr_backbone:  float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> torch.optim.AdamW:
        """Build AdamW with frozen/unfrozen backbone based on ft_level."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        backbone_groups: list[dict] = []
        if self.ft_level > 0:
            args = _FTArgs(
                ft_level1      = self.ft_level >= 1,
                ft_level2      = self.ft_level >= 2,
                ft_level3      = self.ft_level >= 3,
                ft_level4      = self.ft_level >= 4,
                lr_level4      = lr_backbone,
                wd_level4      = weight_decay,
                rank_lora_attn = self.lora_r_attn,
                rank_lora_mlp  = self.lora_r_mlp,
                lora_p         = self.lora_p,
            )
            backbone_groups = SelectFineTuningParameters(
                self.backbone, args,
            ).configure_levels().param_groups

        head_params = (
            list(self.field_combine.parameters()) +
            [self.log_scale, self.bias_param]
        )
        return torch.optim.AdamW(
            backbone_groups + [
                {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
            ]
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, F=3, 1, 1, 72, 72) UPTF-7 tensor (padded, normalised).

        Returns:
            ψ_pred of shape (B, 65, 65) in normalised space.
        """
        _, _, out = self.backbone(x)              # (B, F, 1, 1, 72, 72)
        out = out[:, :, 0, 0, :_GRID, :_GRID]    # (B, F, 65, 65)
        psi = self.field_combine(out)[:, 0]       # (B, 65, 65)
        return self.log_scale.exp() * psi + self.bias_param

    # ── checkpoint I/O ────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: str = "cpu",
    ) -> "MorphGS":
        """Load a trained MorphGS from a saved checkpoint.

        Checkpoint must contain keys: state_dict, args (with ft_level, variant,
        lora_r_attn, lora_r_mlp, lora_alpha, lora_p).

        Returns model in eval mode on the requested device.
        """
        import torch as _torch
        ckpt = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        a = ckpt.get("args", {})
        model = cls(
            checkpoint_path=None,
            variant=a.get("variant", "Ti"),
            ft_level=a.get("ft_level", 1),
            lora_r_attn=a.get("lora_r_attn", 0),
            lora_r_mlp=a.get("lora_r_mlp", 0),
            lora_alpha=a.get("lora_alpha", None),
            lora_p=a.get("lora_p", 0.0),
            device="cpu",
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(_torch.device(device))
        model.eval()
        return model

    # ── convenience ───────────────────────────────────────────────────────────

    def predict_normalised(self, x_np: np.ndarray) -> np.ndarray:
        """Inference from a single UPTF-7 numpy array (no batch dim).

        Args:
            x_np: (1, F, 1, 1, 72, 72) float32, already normalised + padded.

        Returns:
            (65, 65) float32 in normalised ψ space.
        """
        device       = next(self.parameters()).device
        was_training = self.training
        self.eval()
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)   # (1,1,F,1,1,72,72)
        with torch.no_grad():
            out = self.forward(x)[0].cpu().numpy()
        if was_training:
            self.train()
        return out


# ── helpers ───────────────────────────────────────────────────────────────────

class _FTArgs:
    """Minimal args namespace expected by SelectFineTuningParameters."""
    def __init__(self, ft_level1=False, ft_level2=False,
                 ft_level3=False, ft_level4=False,
                 lr_level4=1e-4, wd_level4=0.0,
                 rank_lora_attn=0, rank_lora_mlp=0, lora_p=0.0):
        self.ft_level1      = ft_level1
        self.ft_level2      = ft_level2
        self.ft_level3      = ft_level3
        self.ft_level4      = ft_level4
        self.rank_lora_attn = rank_lora_attn
        self.rank_lora_mlp  = rank_lora_mlp
        self.lora_p         = lora_p
        self.lr_level4      = lr_level4
        self.wd_level4      = wd_level4


def _check_keys(missing: list[str], unexpected: list[str]) -> None:
    bad_missing    = [k for k in missing    if not k.endswith((".A", ".B"))]
    bad_unexpected = [k for k in unexpected if ".A" not in k and ".B" not in k]
    if bad_missing:
        raise RuntimeError(f"Missing backbone keys: {bad_missing}")
    if bad_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {bad_unexpected}")


# ── MorphGSE: backbone + upsampling convolutional decoder ─────────────────────

class UpsamplingDecoder(nn.Module):
    """3-stage ConvTranspose decoder: (B, 256, 9, 9) → (B, 1, 72, 72)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # stage 1: 9→18
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            # stage 2: 18→36
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            # stage 3: 36→72
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            # final projection
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1, 72, 72)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BilinearDecoder(nn.Module):
    """Bilinear upsampling + learned Conv2d refinement: (B, 256, 9, 9) → (B, 1, 72, 72).

    Upsampling is fixed bilinear interpolation (no learned parameters for that step);
    refinement is learned Conv2d. Roughly ~388K params vs ~883K for UpsamplingDecoder.
    """

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 9→18
        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 18→36
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 36→72
        x = self.stage3(x)
        return self.out(x)  # (B, 1, 72, 72)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MorphGSE(nn.Module):
    """MORPH backbone + UpsamplingDecoder head.

    The backbone's internal SimpleDecoder still runs during forward but its
    output is discarded — transformer tokens z are reshaped and fed to
    UpsamplingDecoder instead.

    Args:
        checkpoint_path: FM .pth file with "model_state_dict". None = random backbone.
        frozen_backbone: If True, backbone params are frozen (only decoder trains).
        device:          Torch device string.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        frozen_backbone: bool = False,
        device: str = "cpu",
        decoder: str = "upsampling",
    ):
        super().__init__()
        self.frozen_backbone = frozen_backbone
        self.decoder_type    = decoder

        self.backbone = ViT3DRegression(patch_size=8, **_MORPH_CFG_E)

        if checkpoint_path is not None:
            ckpt  = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            state = ckpt["model_state_dict"]
            if next(iter(state)).startswith("module."):
                state = {k[len("module."):]: v for k, v in state.items()}
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            bad = [k for k in missing if not k.endswith((".A", ".B"))]
            if bad:
                raise RuntimeError(f"Missing backbone keys: {bad}")
            print(f"Backbone loaded: {len(state)} keys")

        if frozen_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("→ Backbone FROZEN (only decoder trains)")
        else:
            print("→ Backbone UNFROZEN (полное дообучение)")

        if decoder == "bilinear":
            self.decoder = BilinearDecoder()
        else:
            self.decoder = UpsamplingDecoder()
        self.log_scale  = nn.Parameter(torch.zeros(1))
        self.bias_param = nn.Parameter(torch.zeros(1))

        self.to(torch.device(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 3, 1, 1, 72, 72)  →  ψ_pred (B, 65, 65) normalised."""
        B = x.shape[0]
        _, z, _ = self.backbone(x)           # z: (B, t=1, n=81, dim=256)
        tokens = z[:, 0]                     # (B, 81, 256)
        tokens = (tokens
                  .view(B, _H_PATCHES, _W_PATCHES, 256)
                  .permute(0, 3, 1, 2)       # (B, 256, 9, 9)
                  .contiguous())
        out = self.decoder(tokens)           # (B, 1, 72, 72)
        psi = out[:, 0, :_GRID, :_GRID]     # (B, 65, 65)
        return self.log_scale.exp() * psi + self.bias_param

    def configure_optimizer(
        self,
        lr_decoder:  float = 1e-3,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-2,
    ) -> torch.optim.AdamW:
        decoder_params = (
            list(self.decoder.parameters()) +
            [self.log_scale, self.bias_param]
        )
        groups = [{"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay}]
        if not self.frozen_backbone:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            groups = [
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            ] + groups
        return torch.optim.AdamW(groups)

    def configure_optimizer_warmup(
        self,
        lr_decoder: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> torch.optim.AdamW:
        """Phase 1: train ONLY decoder + log_scale + bias_param. Backbone frozen externally."""
        decoder_params = (
            list(self.decoder.parameters()) +
            [self.log_scale, self.bias_param]
        )
        return torch.optim.AdamW(
            [{"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay}]
        )

    def configure_optimizer_llrd(
        self,
        lr_decoder: float = 1e-3,
        lr_backbone_top: float = 1e-5,
        llrd_factor: float = 0.75,
        weight_decay: float = 1e-2,
    ) -> torch.optim.AdamW:
        """Phase 2: full fine-tune with layer-wise LR decay on transformer blocks."""
        try:
            n_blocks = len(self.backbone.transformer_blocks)
        except AttributeError:
            block_ids: set[int] = set()
            for name, _ in self.backbone.named_parameters():
                m = re.match(r'transformer_blocks\.(\d+)\.', name)
                if m:
                    block_ids.add(int(m.group(1)))
            n_blocks = len(block_ids) if block_ids else 1

        block_params: dict[int, list] = {i: [] for i in range(n_blocks)}
        other_params: list = []

        for name, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            matched = False
            for i in range(n_blocks):
                if name.startswith(f'transformer_blocks.{i}.'):
                    block_params[i].append(p)
                    matched = True
                    break
            if not matched:
                other_params.append(p)

        groups = []
        for i in range(n_blocks):
            lr_i = lr_backbone_top * (llrd_factor ** (n_blocks - 1 - i))
            params = block_params[i] + (other_params if i == 0 else [])
            if params:
                groups.append({"params": params, "lr": lr_i, "weight_decay": weight_decay})

        decoder_params = (
            list(self.decoder.parameters()) +
            [self.log_scale, self.bias_param]
        )
        groups.append({"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay})
        return torch.optim.AdamW(groups)

    def n_decoder_params(self) -> int:
        return self.decoder.n_params() + 2  # +log_scale +bias_param

    def n_backbone_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())
